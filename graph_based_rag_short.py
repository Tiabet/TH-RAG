import os, openai, networkx as nx
from typing import List, Dict
from dotenv import load_dotenv

from Retriever import Retriever        # ← 청크 기반 버전
from prompt.answer_short import ANSWER_PROMPT
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
CHUNKS_PATH     = "hotpotQA/chunks.txt"
GRAPH_JSON_PATH = "hotpotQA/graph_v1.json"
INDEX_PATH      = "hotpotQA/edge_index_v1.faiss"
PAYLOAD_PATH    = "hotpotQA/edge_payloads_v1.npy"

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

    def compose_context(self, chunks: List[str], edges_meta: List[Dict]) -> str:
        """
        chunks : top_k2개의 청크 본문
        edges_meta : top_k1개의 전체 엣지 정보
        """
        parts: List[str] = []

        # 1) 청크 본문
        for i, chk in enumerate(chunks, 1):
            parts.append(f"[Chunk {i}] {chk}")

        # # 2) 전체 edge 정보 (50개) 포함
        for i, hit in enumerate(edges_meta, 1):
            source = hit.get("source", "?")
            label  = hit.get("label", "?")
            target = hit.get("target", "?")
            score  = hit.get("score", 0.0)
            rank   = hit.get("rank", "?")
            sent   = hit.get("sentence", "")
            cid    = hit.get("chunk_id", "?")

            parts.append(
                f"(Edge {i} | rank={rank} score={score:.3f})\n"
                f"[{source}] --{label}→ [{target}]\n"
                f"{sent}"
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    def answer(self, query: str, top_k1: int = 50, top_k2 : int = 10) -> str:
        # Retriever 실행 → chunks + edges
        
        start_time = time.time()
        out = self.retriever.retrieve(query, top_k1=top_k1, top_k2=top_k2)
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

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a graph‑aware assistant,and an expert that always gives detailed, comprehensive answers."},
                {"role": "user",   "content": prompt},
            ],
            temperature=1.0,
            max_tokens=16384,
            response_format={"type": "text"},
        )
        return resp.choices[0].message.content.strip(), spent_time, context


# ── 예시 실행 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = GraphRAG()
    q = 'Which American comedian born on March 21, 1962, appeared in the movie "Sleepless in Seattle"?'
    ans = rag.answer(q, top_k1=10, top_k2=5)
    print("\n=== Answer ===")
    print(ans)
