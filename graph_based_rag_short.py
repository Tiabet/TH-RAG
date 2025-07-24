import os, openai, json
from typing import List, Dict
from dotenv import load_dotenv
from Retriever import Retriever
from prompt.answer_short_v2 import ANSWER_PROMPT
import time

# ── 환경변수 및 경로 ───────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env var required.")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

GEXF_PATH        = "MultihopRAG/graph_v1.gexf"
JSON_PATH        = "MultihopRAG/graph_v1.json"
KV_JSON_PATH     = "MultihopRAG/kv_store_text_chunks.json"
INDEX_PATH       = "MultihopRAG/edge_index_v1.faiss"
PAYLOAD_PATH     = "MultihopRAG/edge_payloads_v1.npy"

class GraphRAG:
    def __init__(self,
        gexf_path: str        = GEXF_PATH,
        kv_json_path: str     = KV_JSON_PATH,
        index_path: str       = INDEX_PATH,
        payload_path: str     = PAYLOAD_PATH,
        embed_model: str      = EMBED_MODEL,
        chat_model: str       = CHAT_MODEL,
    ):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # kv-store json 로딩 (chunk_id → 본문)
        with open(kv_json_path, encoding="utf-8") as f:
            kv_data = json.load(f)
        self.chunk_map: Dict[str, str] = {
            chunk_id: obj["content"] for chunk_id, obj in kv_data.items()
        }

        # 청크‑기반 Retriever
        self.retriever = Retriever(
            gexf_path       = gexf_path,
            json_path       = JSON_PATH,
            kv_json_path    = kv_json_path,
            index_path      = index_path,
            payload_path    = payload_path,
            embedding_model = embed_model,
            openai_api_key  = OPENAI_API_KEY,
            client          = self.client,
        )

        self.chat_model = chat_model

    def compose_context(self, chunk_ids: List[str], edges_meta: List[Dict]) -> str:
        """
        chunk_ids : top_k2개의 chunk-id
        edges_meta : top_k1개의 전체 엣지 정보
        """
        parts: List[str] = []

        # 1) 청크 본문 (chunk-id → 실제 텍스트)
        for i, cid in enumerate(chunk_ids, 1):
            text = self.chunk_map.get(cid, "(missing)")
            parts.append(f"[Chunk {i}] {text}")

        # 2) 전체 edge 정보
        for i, hit in enumerate(edges_meta, 1):
            source = hit.get("source", "?")
            label  = hit.get("label", "?")
            target = hit.get("target", "?")
            # score  = hit.get("score", 0.0)
            # rank   = hit.get("rank", "?")
            sent   = hit.get("sentence", "")
            # cid  = hit.get("chunk_id", "?")  # 필요시 포함 가능

            parts.append(
                # f"(Edge {i} | rank={rank} score={score:.3f})\n"
                f"[{source}] --{label}→ [{target}]\n"
                f"{sent}"
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    def answer(self, query: str, top_k1: int = 50, top_k2: int = 10) -> str:
        # Retriever 실행 → chunk-ids + edges
        start_time = time.time()
        out = self.retriever.retrieve(query, top_k1=top_k1, top_k2=top_k2)
        end_time = time.time()
        spent_time = end_time - start_time

        chunk_ids: List[str] = out.get("chunks", [])
        edges_meta: List[Dict] = out.get("edges", [])

        chunk_ids: List[str] = out.get("chunks", [])
        edges_meta: List[Dict] = out.get("edges", [])

        # ✅ sentence들이 들어있던 모든 chunk-id 수집
        all_sentence_chunk_ids = []
        seen_chunk_ids = set()
        for edge in edges_meta:
            raw_id = edge.get("chunk_id")
            chunk_id = None
            if isinstance(raw_id, int):
                if 0 <= raw_id < len(self.retriever.chunk_id_list):
                    chunk_id = self.retriever.chunk_id_list[raw_id]
            elif isinstance(raw_id, str) and raw_id.isdigit():
                idx = int(raw_id)
                if 0 <= idx < len(self.retriever.chunk_id_list):
                    chunk_id = self.retriever.chunk_id_list[idx]
            elif isinstance(raw_id, str) and raw_id.startswith("chunk-"):
                chunk_id = raw_id
            if chunk_id and chunk_id not in seen_chunk_ids:
                all_sentence_chunk_ids.append(chunk_id)
                seen_chunk_ids.add(chunk_id)

        self.last_chunk_ids = chunk_ids  # top-k2
        self.all_sentence_chunk_ids = all_sentence_chunk_ids  # ✅ 새로 추가
        
        if not chunk_ids:
            return "죄송합니다. 관련 정보를 찾지 못했습니다."

        # 컨텍스트 조립
        context = self.compose_context(chunk_ids, edges_meta)
        prompt  = ANSWER_PROMPT.replace("{question}", query).replace("{context}", context)

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a graph‑aware assistant, and an expert that always gives detailed, comprehensive answers."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "text"},
        )
        return resp.choices[0].message.content.strip(), spent_time, context

# ── 예시 실행 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, numpy as np, faiss, os
    with open(KV_JSON_PATH, encoding="utf-8") as f:          # ← 인코딩 명시
        kv_data = json.load(f)
    print("📝 kv‑store chunks :", len(kv_data))

    payload = np.load(PAYLOAD_PATH, allow_pickle=True)   # mmap_mode 빼고, allow_pickle=True
    print("📦 payload entries:", len(payload))

    index = faiss.read_index(INDEX_PATH)
    print("🔍 faiss index    :", index.ntotal)
    
    rag = GraphRAG()
    q = "Are the films Gasland from 2010 and Pumping Iron from 1977 both documentaries?"
    ans = rag.answer(q, top_k1=10, top_k2=5)
    print("\n=== Answer ===")
    print(ans)