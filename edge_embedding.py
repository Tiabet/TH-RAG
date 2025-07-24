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
import argparse

GEXF_PATH = "UltraDomain/Agriculture/graph_v1.gexf"
INDEX_PATH = "UltraDomain/Agriculture/edge_index_v1.faiss"
PAYLOAD_PATH = "UltraDomain/Agriculture/edge_payloads_v1.npy"
JSON_PATH = "UltraDomain/Agriculture/graph_v1.json"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_WORKERS     = 50

# Load environment variables
load_dotenv()

if "SSL_CERT_FILE" in os.environ:
    print("⚠️ Removing problematic SSL_CERT_FILE:", os.environ["SSL_CERT_FILE"])
    os.environ.pop("SSL_CERT_FILE")

Edge = Tuple[str, str, str, str, str]  # id, source, target, label, sentence

# === Configuration ===

# OpenAI API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("환경 변수 OPENAI_API_KEY를 설정해야 합니다.")

import re

def build_sent2chunk(graph_json_path: str) -> Dict[str, int]:
    with open(graph_json_path, encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for idx, block in enumerate(data):
        if not isinstance(block, dict):
            print(f"⚠️ Skipping non-dict block at index {idx}: {block}")
            continue

        chunk_id = block.get("chunk_id")
        triples = block.get("triples", [])

        if not isinstance(triples, list):
            continue

        for item in triples:
            if not isinstance(item, dict):
                continue

            sentence = item.get("sentence")

            if isinstance(sentence, str):
                mapping[sentence] = chunk_id
            elif isinstance(sentence, list):
                for s in sentence:
                    if isinstance(s, str):
                        mapping[s] = chunk_id

    return mapping

class EdgeEmbedderFAISS:


    def __init__(
        self,
        gexf_path: str,
        embedding_model: str,
        openai_api_key: str,
        index_path: str,
        payload_path: str,
        json_path: str,
    ) -> None:
        # Load graph and initialize
        self.graph = nx.read_gexf(gexf_path)
        self.embedding_model = embedding_model
        # self.openai = OpenAI(api_key=openai_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.openai = OpenAI(api_key=openai_api_key)
        self.index_path = index_path
        self.payload_path = payload_path
        self.json_path = json_path

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
        self.sent2cid = build_sent2chunk(self.json_path)


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
            key = sent
            if key in seen:
                return None
            emb = self._embed(sent)
            cid = self.sent2cid.get(key)
            payload = {
            "edge_id": eid,
            "source": src,
            "target": dst,
            "label": lbl,
            "sentence": sent,
            "chunk_id": cid,  # ✅ 추가됨
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

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_entities: Set[str] | None = None,
        overretrieve: int = 5,          # 필터용 여유 검색 개수(top_k*overretrieve)
    ) -> List[Dict]:
        """
        query         : 검색할 문장
        top_k         : 최종으로 돌려줄 결과 수
        filter_entities : {"E1", "E2"} 형태. 있으면 source/target 기준으로 필터
        overretrieve  : 필터가 있을 때 넉넉히 가져올 배수
        """
        # 1️⃣ 쿼리 임베딩
        q_vec = self._embed(query).reshape(1, -1)

        # 2️⃣ FAISS 검색 (필터 O → 더 많이, 필터 X → 정확히 top_k)
        k = top_k * overretrieve if filter_entities else top_k
        D, I = self.index.search(q_vec, k)

        # 3️⃣ 결과 후처리 (필터 적용 + top_k 슬라이스)
        results = []
        for rank, idx in enumerate(I[0], start=1):
            if idx < 0:                 # padding 값(-1) 방어
                continue
            p = self.payloads[idx]

            # 엔티티 필터링
            if filter_entities and (
                p["source"] not in filter_entities
                and p["target"] not in filter_entities
            ):
                continue

            results.append({
                "edge_id" : p["edge_id"],
                "source"  : p["source"],
                "target"  : p["target"],
                "label"   : p.get("label"),
                "sentence": p.get("sentence"),
                "chunk_id": p.get("chunk_id"),
                "score"   : float(D[0][rank - 1]),
                "rank"    : len(results) + 1,
            })

            if len(results) == top_k:   # 원하는 개수 채우면 종료
                break

        return results


# ── 명령줄에서 직접 실행될 경우만 ────────────────────────────────────────
if __name__ == "__main__":

    embedder = EdgeEmbedderFAISS(
        gexf_path=GEXF_PATH,
        json_path=JSON_PATH,
        embedding_model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        index_path=INDEX_PATH,
        payload_path=PAYLOAD_PATH,
    )

    if not os.path.exists(INDEX_PATH):
        embedder.build_index()
        print("FAISS index & payloads 생성 완료.")