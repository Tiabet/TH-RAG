import os
import networkx as nx
import numpy as np
import faiss
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI

Edge = Tuple[str, str, str, str, str]  # id, source, target, label, sentence

# === Configuration ===
GEXF_PATH       = "DB/graph_v7.gexf"
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_PATH      = "DB/edge_index.faiss"
PAYLOAD_PATH    = "DB/edge_payloads.npy"
MAX_WORKERS     = 10

# OpenAI API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("환경 변수 OPENAI_API_KEY를 설정해야 합니다.")

class EdgeEmbedderFAISS:
    """Embed edge sentences into a local FAISS index with payloads in parallel."""

    def __init__(
        self,
        gexf_path: str,
        embedding_model: str,
        openai_api_key: str,
        index_path: str,
        payload_path: str,
    ) -> None:
        # Load graph and init
        self.graph = nx.read_gexf(gexf_path)
        self.embedding_model = embedding_model
        self.openai = OpenAI(api_key=openai_api_key)
        self.index_path = index_path
        self.payload_path = payload_path

        # Prepare edges
        self.edges: List[Edge] = []
        for src, dst, data in self.graph.edges(data=True):
            sent = data.get("sentence")
            lbl  = data.get("label") or data.get("relation_type", "")
            if sent:
                eid = f"{src}-{dst}-{lbl}".replace(" ", "_")
                self.edges.append((eid, src, dst, lbl, sent))

        # placeholders for FAISS
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

        # Worker to generate embedding and payload
        def worker(edge: Edge):
            eid, src, dst, lbl, sent = edge
            emb = self._embed(sent)
            payload = {
                "edge_id": eid,
                "source": src,
                "target": dst,
                "label": lbl,
                "sentence": sent
            }
            return emb, payload

        # Parallel embedding with progress bar
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for emb, payload in tqdm(
                executor.map(worker, self.edges),
                total=len(self.edges),
                desc="Embedding edges"
            ):
                results.append((emb, payload))

        vecs = np.vstack([r[0] for r in results])
        self.payloads = [r[1] for r in results]
        self.index.add(vecs)

        # Persist index and payloads
        faiss.write_index(self.index, self.index_path)
        np.save(self.payload_path, np.array(self.payloads, dtype=object))

    def load_index(self) -> None:
        self.index = faiss.read_index(self.index_path)
        self.payloads = np.load(self.payload_path, allow_pickle=True).tolist()

    def search(self, query: str, top_k: int = 5) -> List[Edge]:
        q_emb = self._embed(query).reshape(1, -1)
        _, indices = self.index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            p = self.payloads[idx]
            results.append((p["edge_id"], p["source"], p["target"], p["label"], p["sentence"]))
        return results

# Usage example (no main guard):
# instantiate and build or load the index directly
embedder = EdgeEmbedderFAISS(
    gexf_path=GEXF_PATH,
    embedding_model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    index_path=INDEX_PATH,
    payload_path=PAYLOAD_PATH,
)

# Build index if not exists, otherwise load
if not os.path.exists(INDEX_PATH):
    embedder.build_index()
    print("FAISS index & payloads 생성 완료.")
else:
    embedder.load_index()
    print("FAISS index & payloads 로드 완료.")

# 검색 예시
results = embedder.search("Bee is good")
for edge in results:
    print(edge)
