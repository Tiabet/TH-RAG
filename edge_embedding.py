import os
import json
import networkx as nx
import numpy as np
import faiss
from typing import List, Tuple, Dict, Set
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if "SSL_CERT_FILE" in os.environ:
    print("⚠️ Removing problematic SSL_CERT_FILE:", os.environ["SSL_CERT_FILE"])
    os.environ.pop("SSL_CERT_FILE")

Edge = Tuple[str, str, str, str, str]  # id, source, target, label, sentence

# === Configuration ===
GEXF_PATH       = "hotpotQA/graph_v1.gexf"
EMBEDDING_MODEL = "text-embedding-3-small"
# EMBEDDING_MODEL = "gemini-embedding-exp-03-07"  # 더 큰 모델을 사용하여 임베딩 품질 향상
INDEX_PATH      = "hotpotQA/edge_index_v1.faiss"
PAYLOAD_PATH    = "hotpotQA/edge_payloads_v1.npy"
MAX_WORKERS     = 30

# OpenAI API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("환경 변수 OPENAI_API_KEY를 설정해야 합니다.")

import re

def _normalize_sentence(s: str) -> str:
    """대소문자·공백·구두점 최소 정규화."""
    s = re.sub(r"\s+", " ", s.strip())   # 줄바꿈·다중 공백 → 1칸
    return s.lower()

class EdgeEmbedderFAISS:
    """Embed each edge sentence separately into FAISS index with corresponding payloads."""

    def __init__(
        self,
        gexf_path: str,
        embedding_model: str,
        openai_api_key: str,
        index_path: str,
        payload_path: str,
    ) -> None:
        # Load graph and initialize
        self.graph = nx.read_gexf(gexf_path)
        self.embedding_model = embedding_model
        # self.openai = OpenAI(api_key=openai_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.openai = OpenAI(api_key=openai_api_key)
        self.index_path = index_path
        self.payload_path = payload_path

        # Prepare edges: split by sentence and assign unique ids
        self.edges: List[Edge] = []
        for src, dst, data in self.graph.edges(data=True):
            label = data.get("label") or data.get("relation_type", "")
            sentence_block = data.get("sentence", "")
            if sentence_block:
                sentences = [s.strip() for s in sentence_block.split("/") if s.strip()]
                for i, sentence in enumerate(sentences):
                    eid = f"{src}-{dst}-{label}".replace(" ", "_") + f"#{i}"
                    self.edges.append((eid, src, dst, label, sentence))

        # Placeholders for index and payload
        self.index: faiss.IndexFlatIP
        self.payloads: List[Dict] = []


    def _embed(self, text: str) -> np.ndarray:
        resp = self.openai.embeddings.create(
            input=[text], model=self.embedding_model
        )
        emb = np.array(resp.data[0].embedding, dtype="float32")
        return emb / np.linalg.norm(emb)

    def build_index(self) -> None:
        # Determine embedding dimension
        dim = self._embed("test").shape[0]
        self.index = faiss.IndexFlatIP(dim)

        
        seen: set[str] = set()           # ✅ 이미 임베딩한 문장 저장
        vecs, payloads = [], []

        # Worker function to embed and generate payload
        def worker(edge: Edge):
            eid, src, dst, lbl, sent = edge
            key = _normalize_sentence(sent)
            if key in seen:
                return None
            emb = self._embed(sent)
            payload = {
                "edge_id": eid,
                "source": src,
                "target": dst,
                "label": lbl,
                "sentence": sent
            }
            return emb, payload

        # Embed all edge sentences in parallel
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for res in tqdm(executor.map(worker, self.edges), total=len(self.edges), desc="Embedding edges"):
                if res is None:
                    continue
                emb, payload = res
                vecs.append(emb)
                payloads.append(payload)

        self.payloads = payloads
        self.index.add(np.vstack(vecs))

        faiss.write_index(self.index, self.index_path)
        np.save(self.payload_path, np.array(self.payloads, dtype=object))

    def load_index(self) -> None:
        self.index = faiss.read_index(self.index_path)
        self.payloads = np.load(self.payload_path, allow_pickle=True).tolist()

    def search(self, query: str, top_k: int = 5, filter_entities: Set[str] = None) -> List[Dict]:
        q_emb = self._embed(query).reshape(1, -1)
        if filter_entities:
            # payloads 필터링
            filtered = [(i, p) for i, p in enumerate(self.payloads)
                        if p["source"] in filter_entities or p["target"] in filter_entities]
            idxs, payloads = zip(*filtered) if filtered else ([], [])
            embs = np.vstack([self.index.reconstruct(int(i)) for i in idxs])
            sims = (embs @ q_emb.T).flatten()
            order = np.argsort(-sims)[:min(len(sims), top_k)]
            results = []
            for rank, idx in enumerate(order, start=1):
                p = payloads[idx]
                results.append({
                    "edge_id": p["edge_id"],
                    "source": p["source"],
                    "target": p["target"],
                    "label": p.get("label"),
                    "sentence": p.get("sentence"),
                    "score": float(sims[idx]),
                    "rank": rank
                })
            return results
        else:
            # 기존 로직 유지
            D, I = self.index.search(q_emb, top_k)
            results = []
            for rank, idx in enumerate(I[0], start=1):
                p = self.payloads[idx]
                results.append({
                    "edge_id": p["edge_id"],
                    "source": p["source"],
                    "target": p["target"],
                    "label": p.get("label"),
                    "sentence": p.get("sentence"),
                    "score": float(D[0][rank-1]),
                    "rank": rank
                })
            return results


embedder = EdgeEmbedderFAISS(
    gexf_path=GEXF_PATH,
    embedding_model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    index_path=INDEX_PATH,
    payload_path=PAYLOAD_PATH,
)

if not os.path.exists(INDEX_PATH):
    embedder.build_index()
    print("FAISS index & payloads 생성 완료.")
# else:
#     embedder.load_index()
#     print("FAISS index & payloads 로드 완료.")

# # 예시 검색
# results = embedder.search("Bee is good")
# for edge in results:
#     print(edge)
