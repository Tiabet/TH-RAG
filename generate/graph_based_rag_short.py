import os, openai, json, sys, time
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Retriever import Retriever
from prompt.answer_short import ANSWER_PROMPT

# ── 환경변수 및 경로 ───────────────────────────────────────────────────
load_dotenv()
import sys
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 설정 import
from config import get_config
config = get_config()

OPENAI_API_KEY = config.openai_api_key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env var required.")

EMBED_MODEL = config.embed_model
CHAT_MODEL  = config.chat_model

class GraphRAG:
    def __init__(self,
        dataset_name: str = None,
        gexf_path: str = None,
        json_path: str = None,
        kv_json_path: str = None,
        index_path: str = None,
        payload_path: str = None,
        embed_model: str = EMBED_MODEL,
        chat_model: str = CHAT_MODEL,
    ):
        # 설정 객체 생성
        if dataset_name:
            config = get_config(dataset_name)
            self.gexf_path = gexf_path or str(config.get_graph_gexf_file())
            self.json_path = json_path or str(config.get_graph_json_file())
            self.kv_json_path = kv_json_path or str(config.get_kv_store_file())
            self.index_path = index_path or str(config.get_edge_index_file())
            self.payload_path = payload_path or str(config.get_edge_payload_file())
        else:
            # 기본값 설정 (호환성을 위해)
            self.gexf_path = gexf_path or "hotpotQA/graph_v1.gexf"
            self.json_path = json_path or "hotpotQA/graph_v1.json"
            self.kv_json_path = kv_json_path or "hotpotQA/kv_store_text_chunks.json"
            self.index_path = index_path or "hotpotQA/edge_index_v1.faiss"
            self.payload_path = payload_path or "hotpotQA/edge_payloads_v1.npy"
        
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # kv-store json 로딩 (chunk_id → 본문)
        with open(self.kv_json_path, encoding="utf-8") as f:
            kv_data = json.load(f)
        self.chunk_map: Dict[str, str] = {
            chunk_id: obj["content"] for chunk_id, obj in kv_data.items()
        }

        # 청크‑기반 Retriever
        self.retriever = Retriever(
            gexf_path       = self.gexf_path,
            json_path       = self.json_path,
            kv_json_path    = self.kv_json_path,
            index_path      = self.index_path,
            payload_path    = self.payload_path,
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
    # 테스트용 예제
    import json, numpy as np, faiss, os
    
    # 기본 설정으로 테스트
    config = get_config("hotpotQA")  # 기본 데이터셋
    
    kv_json_path = str(config.get_kv_store_file())
    payload_path = str(config.get_edge_payload_file())
    index_path = str(config.get_edge_index_file())
    
    if os.path.exists(kv_json_path):
        with open(kv_json_path, encoding="utf-8") as f:
            kv_data = json.load(f)
        print("📝 kv‑store chunks :", len(kv_data))
    
    if os.path.exists(payload_path):
        payload = np.load(payload_path, allow_pickle=True)
        print("📦 payload entries:", len(payload))
    
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print("🔍 faiss index    :", index.ntotal)
        
        rag = GraphRAG(dataset_name="hotpotQA")
        q = "Which OpenAI figure rose with ChatGPT, promoted AI agents, and faced board controversy per Fortune and TechCrunch?"
        ans = rag.answer(q, top_k1=50, top_k2=5)
        print("\n=== Answer ===")
        print(ans)
    else:
        print("❌ Index files not found. Please run indexing first.")