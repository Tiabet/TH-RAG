import os
import networkx as nx
from typing import List, Dict, Set
import openai
from dotenv import load_dotenv

# from Retriever import Retriever  # Retriever.py에 정의된 클래스
from Retriever_v1 import Retriever  # retriever_test.py에 정의된 클래스
from prompt.answer import ANSWER_PROMPT

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY environment variable required.")

# 모델 및 경로 설정
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL    = os.getenv("CHAT_MODEL", "gpt-4o-mini")
GRAPH_PATH    = os.getenv("GRAPH_PATH", "hotpotQA/graph_v1.gexf")
INDEX_PATH    = os.getenv("INDEX_PATH", "hotpotQA/edge_index.faiss")
PAYLOAD_PATH  = os.getenv("PAYLOAD_PATH", "hotpotQA/edge_payloads.npy")

class GraphRAG:
    def __init__(
        self,
        gexf_path: str = GRAPH_PATH,
        index_path: str = INDEX_PATH,
        payload_path: str = PAYLOAD_PATH,
        embed_model: str = EMBED_MODEL,
        chat_model: str = CHAT_MODEL,
    ):
        # Retriever 초기화
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = Retriever(
            gexf_path=gexf_path,
            embedding_model=embed_model,
            openai_api_key=OPENAI_API_KEY,
            index_path=index_path,
            payload_path=payload_path,
            client = self.client,
        )
        # Chat client
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.chat_model = chat_model

    def compose_context(self, entity_sentences: Dict[str, List[str]], faiss_results: List[Dict]) -> str:
        parts = []
        # 엔티티 문장
        for ent, sents in entity_sentences.items():
            for s in sents:
                parts.append(f"[Entity: {ent}] {s}")
        # FAISS 결과
        for hit in faiss_results:
            parts.append(f"(Edge {hit['edge_id']} score={hit['score']:.3f}): {hit['sentence']}")
        return "\n".join(parts)

    def answer(self, query: str) -> str:
        # topic_infos = extract_topics_subtopics(query, self.client)
        # Retriever로부터 entity_sentences와 faiss_results 얻기
        outputs = self.retriever.retrieve(query, top_n=50)  # query 인자 추가
        entity_sentences = outputs.get("entity_sentences", {})
        faiss_results = outputs.get("faiss_results", [])

        # 컨텍스트 구성
        context = self.compose_context(entity_sentences, faiss_results)     
        prompt = ANSWER_PROMPT.replace("{question}", query).replace("{context}", context)
        print(prompt)
        print(len(context), "characters in context.")

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a graph-aware assistant, capable of understanding complex relationships."},
                {"role": "user", "content": prompt}
            ]
        )

        
        return resp.choices[0].message.content.strip()

# if __name__ == "__main__":
#     rag = GraphRAG()
#     question = "What recurring tasks are essential for successful hive management throughout the bee season?"
#     answer = rag.answer(question)
#     print("\n=== Answer ===")
#     print(answer)
