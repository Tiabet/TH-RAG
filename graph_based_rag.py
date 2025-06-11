import os
import faiss
import numpy as np
import openai
from openai import OpenAI
import networkx as nx
from typing import List, Dict
from edge_topic import extract_topics_subtopics
# import certifi
# os.environ["SSL_CERT_FILE"] = certifi.where()

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()



# 설정
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
GRAPH_PATH = os.getenv("GRAPH_PATH", "graph.gexf")
INDEX_PATH = os.getenv("INDEX_PATH", "edge_index.faiss")
PAYLOAD_PATH = os.getenv("PAYLOAD_PATH", "edge_payloads.npy")

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
        self.graph = nx.read_gexf(graph_path)
        self.label_map = {
            data.get("label"): n
            for n, data in self.graph.nodes(data=True)
            if data.get("label") is not None
        }
        # FAISS 인덱스 및 페이로드 로드
        self.index = faiss.read_index(index_path)
        self.payloads = np.load(payload_path, allow_pickle=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Query 임베딩 생성"""
        resp = openai.Embedding.create(
            model=self.embed_model,
            input=[query]
        )
        return np.array(resp["data"][0]["embedding"], dtype="float32")[None, :]

    # def retrieve(self, query: str, top_n: int = 50) -> List[Dict]:
    #     """
    #     1) 질문에서 토픽/서브토픽 추출
    #     2) 매칭되는 노드 기반 엣지 필터링
    #     3) 필터링된 엣지를 query 임베딩과 유사도 계산하여 상위 N개 선택
    #     """
    #     # 토픽/서브토픽 추출
    #     topics = extract_topics_subtopics(query, self.client)
    #     terms = [t["topic"] for t in topics] + [sub for t in topics for sub in t["subtopics"]]
    #     # 노드 라벨 매핑
    #     matched_nodes = [self.label_map[t] for t in terms if t in self.label_map]
    #     # 엣지 필터링
    #     filtered_idxs = [
    #         i for i, p in enumerate(self.payloads)
    #         if p.item()["source"] in matched_nodes or p.item()["target"] in matched_nodes
    #     ]
    #     # query 임베딩 및 유사도 계산
    #     q_emb = self.embed_query(query)
    #     embs = np.vstack([self.index.reconstruct(int(i)) for i in filtered_idxs])
    #     sims = (embs @ q_emb.T).flatten()
    #     # 상위 N개 선택
    #     top_n = min(len(sims), top_n)
    #     order = np.argsort(-sims)[:top_n]
    #     results = []
    #     for rank, idx in enumerate(order, start=1):
    #         p = self.payloads[filtered_idxs[idx]].item()
    #         p["score"] = float(sims[idx])
    #         p["rank"] = rank
    #         results.append(p)
    #     return results

    def retrieve(self, query: str, top_n: int = 50) -> List[Dict]:
        topics = extract_topics_subtopics(query, self.client)
        terms = [t["topic"] for t in topics] + [sub for t in topics for sub in t["subtopics"]]
        matched_nodes = [self.label_map[t] for t in terms if t in self.label_map]

        filtered_idxs = [
            i for i, p in enumerate(self.payloads)
            if p["source"] in matched_nodes or p["target"] in matched_nodes
        ]

        q_emb = self.embed_query(query)
        embs = np.vstack([self.index.reconstruct(int(i)) for i in filtered_idxs])
        sims = (embs @ q_emb.T).flatten()
        top_n = min(len(sims), top_n)
        order = np.argsort(-sims)[:top_n]

        results = []
        for rank, idx in enumerate(order, start=1):
            p = self.payloads[filtered_idxs[idx]]
            p["score"] = float(sims[idx])
            p["rank"] = rank
            results.append(p)
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
        context = self.compose_context(docs)
        prompt = (
            f"You are a helpful AI assistant. Use the following graph edges as context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:")
        resp = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a graph-aware assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content.strip()

# 실행용 인터페이스
query = input("What are the steps involved in extracting and handling honey, and why is it important to strain the honey after extraction?")
rag = GraphRAG()
print("\n=== Answer ===")
print(rag.answer(query))
