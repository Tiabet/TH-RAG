import os
import faiss
import numpy as np
import openai
from openai import OpenAI
import networkx as nx
from typing import List, Dict
from edge_topic import extract_topics_subtopics

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()



# 설정
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
GRAPH_PATH = os.getenv("GRAPH_PATH", "DB/graph_v7.gexf")
INDEX_PATH = os.getenv("INDEX_PATH", "DB/edge_index.faiss")
PAYLOAD_PATH = os.getenv("PAYLOAD_PATH", "DB/edge_payloads.npy")

class GraphRAG:
    def __init__(self,
                 graph_path: str = GRAPH_PATH,
                 index_path: str = INDEX_PATH,
                 payload_path: str = PAYLOAD_PATH,
                 embed_model: str = EMBED_MODEL,
                 chat_model: str = CHAT_MODEL):
        # 모델 및 클라이언트 설정
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # 그래프 로드 및 라벨 매핑
        print("Loading graph and index...")
        self.graph = nx.read_gexf(graph_path)
        self.label_map = {
            data.get("label"): n
            for n, data in self.graph.nodes(data=True)
            if data.get("label") is not None
        }
        print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        # FAISS 인덱스 및 페이로드 로드
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} edges.")
        self.payloads = np.load(payload_path, allow_pickle=True)
        print(f"Loaded {len(self.payloads)} edge payloads.")

    def embed_query(self, query: str) -> np.ndarray:
        """Query 임베딩 생성"""
        resp = openai.embeddings.create(
            model=self.embed_model,
            input=query
        )
        # print(resp)
        return np.array(resp.data[0].embedding, dtype="float32")[None, :]

    def retrieve(self, query: str, top_n: int = 50) -> List[Dict]:
        print("1. Extracting topics...")
        topics = extract_topics_subtopics(query, self.client)
        print(f" - Topics extracted: {topics}")

        terms = [t["topic"] for t in topics] + [sub for t in topics for sub in t["subtopics"]]
        print(f" - Total terms extracted: {len(terms)}")
        matched_nodes = [self.label_map[t] for t in terms if t in self.label_map]
        print(f" - Matched nodes: {len(matched_nodes)}")

        filtered_idxs = [
            i for i, p in enumerate(self.payloads)
            if p["source"] in matched_nodes or p["target"] in matched_nodes
        ]
        print(f"2. Filtered edges: {len(filtered_idxs)}")

        q_emb = self.embed_query(query)
        print("3. Query embedded.")

        embs = np.vstack([self.index.reconstruct(int(i)) for i in filtered_idxs])
        print("4. Embeddings reconstructed.")

        sims = (embs @ q_emb.T).flatten()
        print("5. Similarities computed.")

        top_n = min(len(sims), top_n)
        order = np.argsort(-sims)[:top_n]

        results = []
        for rank, idx in enumerate(order, start=1):
            p = self.payloads[filtered_idxs[idx]]
            p["score"] = float(sims[idx])
            p["rank"] = rank
            results.append(p)
        print("6. Retrieval complete.")
        return results

    def compose_context(self, docs: List[Dict]) -> str:
        """검색된 엣지를 컨텍스트 문자열로 결합"""
        result = "\n\n".join(
            f"Edge {d['edge_id']} ({d['source']} -> {d['target']}, label={d['label']}):\n{d['sentence']}"
            for d in docs
        )

        # print(len(docs))
        return result

    def answer(self, query: str) -> str:
        docs = self.retrieve(query)
        print("=== Query ===")
        print(query)
        context = self.compose_context(docs)
        print("=== Context ===")
        print(context)
        prompt = (
            f"You are a helpful AI assistant. Use the following graph edges as context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:")
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a graph-aware assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        print("=== Response ===")
        return resp.choices[0].message.content.strip()

# 실행용 인터페이스
query = "What are the steps involved in extracting and handling honey, and why is it important to strain the honey after extraction?"
rag = GraphRAG()
print(rag.answer(query))
