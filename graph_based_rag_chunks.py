import os, openai, networkx as nx
from typing import List, Dict
from dotenv import load_dotenv

from Retriever_v4 import Retriever        # ← 청크 기반 버전
from prompt.answer_infiniteqa import ANSWER_PROMPT
import time
import tiktoken

# ── 환경변수 및 경로 ───────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env var required.")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

GEXF_PATH       = "hotpotQA/graph_v1.gexf"
CHUNKS_PATH     = "hotpotQA/chunks_v1.txt"
GRAPH_JSON_PATH = "hotpotQA/graph_v1.json"
INDEX_PATH      = "hotpotQA/edge_index_v1.faiss"
PAYLOAD_PATH    = "hotpotQA/edge_payloads_v1.npy"
# ─────────────────────────────────────────────────────────────────────

class GraphRAG:
    def __init__(
        self,
        gexf_path: str        = GEXF_PATH,
        chunks_path: str      = CHUNKS_PATH,
        graph_json_path: str  = GRAPH_JSON_PATH,
        index_path: str       = INDEX_PATH,
        payload_path: str     = PAYLOAD_PATH,
        embed_model: str      = EMBED_MODEL,
        chat_model: str       = CHAT_MODEL,
    ):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # 청크‑기반 Retriever
        self.retriever = Retriever(
            gexf_path       = gexf_path,
            chunks_path     = chunks_path,
            graph_json_path = graph_json_path,
            index_path      = index_path,
            payload_path    = payload_path,
            embedding_model = embed_model,
            openai_api_key  = OPENAI_API_KEY,
            client          = self.client,     # 같은 OpenAI 인스턴스 공유
        )


        self.chat_model = chat_model

    # ------------------------------------------------------------------
    def compose_context(self, chunks: List[str], edges_meta: List[Dict]) -> str:
        """
        chunks : 검색된 청크 원문 리스트
        edges_meta : 문장/점수 등 메타 (원하면 프롬프트에 포함)
        """
        parts: List[str] = []

        # 1) 청크 본문
        for i, chk in enumerate(chunks, 1):
            parts.append(f"[Chunk {i}] {chk}")

        # 2) (옵션) 문장 메타 – 프롬프트 디버그용
        for hit in edges_meta:
            parts.append(f"(Edge {hit['edge_id']} score={hit['score']:.3f}): {hit['sentence']}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    def answer(self, query: str, top_k: int = 10) -> str:
        # Retriever 실행 → chunks + edges
        
        start_time = time.time()
        out = self.retriever.retrieve(query, top_k=top_k)
        end_time = time.time()
        spent_time = end_time - start_time
        chunks: List[str]       = out.get("chunks", [])
        edges_meta: List[Dict]  = out.get("edges", [])

        if not chunks:
            return "죄송합니다. 관련 정보를 찾지 못했습니다."

        # 컨텍스트 조립
        context = self.compose_context(chunks, edges_meta)
        prompt  = ANSWER_PROMPT.replace("{question}", query).replace("{context}", context)
        # print(f"{len(context)} characters in context.")
        tokenizer = tiktoken.encoding_for_model(self.chat_model)
        context_tokens = tokenizer.encode(context, disallowed_special=())

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a graph‑aware assistant, capable of understanding complex relationships."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip(), spent_time, len(context_tokens)


# ── 예시 실행 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = GraphRAG()
    q = 'Which American comedian born on March 21, 1962, appeared in the movie "Sleepless in Seattle"?'
    ans = rag.answer(q, top_k=5)
    print("\n=== Answer ===")
    print(ans)
