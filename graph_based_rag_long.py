import os, openai, json, time
from typing import List, Dict
from dotenv import load_dotenv
from Retriever import Retriever
from prompt.answer import ANSWER_PROMPT
import tiktoken

# ── 환경변수 및 경로 ───────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env var required.")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

GEXF_PATH        = "UltraDomain/Mix/graph_v1.gexf"
JSON_PATH        = "UltraDomain/Mix/graph_v1.json"
KV_JSON_PATH     = "UltraDomain/Mix/kv_store_text_chunks.json"
INDEX_PATH       = "UltraDomain/Mix/edge_index_v1.faiss"
PAYLOAD_PATH     = "UltraDomain/Mix/edge_payloads_v1.npy"

class GraphRAG:
    def __init__(
        self,
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
        parts: List[str] = []

        for i, cid in enumerate(chunk_ids, 1):
            text = self.chunk_map.get(cid, "(missing)")
            parts.append(f"[Chunk {i}] {text}")

        for i, hit in enumerate(edges_meta, 1):
            source = hit.get("source", "?")
            label  = hit.get("label", "?")
            target = hit.get("target", "?")
            sent   = hit.get("sentence", "")

            parts.append(
                f"[{source}] --{label}→ [{target}]\n{sent}"
            )

        return "\n".join(parts)

    def answer(self, query: str, top_k1: int = 50, top_k2: int = 10) -> str:
        start_time = time.time()
        out = self.retriever.retrieve(query, top_k1=top_k1, top_k2=top_k2)
        end_time = time.time()
        spent_time = end_time - start_time

        chunk_ids: List[str] = out.get("chunks", [])
        edges_meta: List[Dict] = out.get("edges", [])

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

        self.last_chunk_ids = chunk_ids
        self.all_sentence_chunk_ids = all_sentence_chunk_ids

        if not chunk_ids:
            return "죄송합니다. 관련 정보를 찾지 못했습니다."

        context = self.compose_context(chunk_ids, edges_meta)
        prompt  = ANSWER_PROMPT.replace("{question}", query).replace("{context}", context)

        tokenizer = tiktoken.encoding_for_model(self.chat_model)
        context_tokens = tokenizer.encode(context, disallowed_special=())

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system",
                 "content": (
                     "Your responses should be **long, comprehensive, diverse, and empowering**, drawing deeply from the provided information. "
                     "Always aim to synthesize the information meaningfully, avoiding surface-level summaries. "
                     "Be analytical and thoughtful. Your goal is not just to answer, but to **educate and inform** using the full richness of the input data."
                 )},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=16384,
            response_format={"type": "text"},
        )
        return resp.choices[0].message.content.strip(), spent_time, len(context_tokens)

# ── 예시 실행 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = GraphRAG()
    q = "What recurring tasks are essential for successful hive management throughout the bee season?"
    ans = rag.answer(q, top_k1=25, top_k2=5)
    print("\n=== Answer ===")
    print(ans)
