
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from index.edge_embedding import EdgeEmbedderFAISS
from index.topic_choice import choose_topics_from_graph
from index.subtopic_choice import choose_subtopics_for_topic

load_dotenv()

# ---------- Hardcoded/Paths ----------
GEXF_PATH = "MultihopRAG/graph_v1.gexf"
KV_JSON_PATH = "MultihopRAG/kv_store_text_chunks.json"
GRAPH_JSON_PATH = "MultihopRAG/graph_v1.json"
INDEX_PATH = "MultihopRAG/edge_index_v1.faiss"
PAYLOAD_PATH = "MultihopRAG/edge_payloads_v1.npy"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_WORKERS = 10
# -----------------------------------

# ... 생략된 import 및 상수 정의는 그대로 ...

class Retriever:
    def __init__(
        self,
        gexf_path: str,
        json_path: str,
        kv_json_path: str,
        index_path: str,
        payload_path: str,
        embedding_model: str,
        openai_api_key: str,
        client: OpenAI | None = None,
        *,
        thread_workers: int = 10,
    ) -> None:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        print("📖  loading graph …", end=" ")
        self.graph = nx.read_gexf(gexf_path)
        print(f"done ({self.graph.number_of_nodes()} nodes)")

        # kv-store 로딩 (chunk id map도 여기서 준비)
        with open(kv_json_path, encoding="utf-8") as f:
            kv_data = json.load(f)

        self.chunk_map: Dict[str, str] = {k: v["content"] for k, v in kv_data.items()}
        self.chunk_id_list: List[str] = list(kv_data.keys())  # ← index → chunk_id 변환용
        print(f"📚  {len(self.chunk_map)} chunks loaded")

        self.client = client or OpenAI(api_key=openai_api_key)

        self.embedder = EdgeEmbedderFAISS(
            gexf_path=gexf_path,
            json_path=json_path,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            index_path=index_path,
            payload_path=payload_path,
        )
        if os.path.exists(index_path):
            self.embedder.load_index()
            print("✅  FAISS index loaded\n")

        self.topic_lbl2nid = {
            d["label"]: n for n, d in self.graph.nodes(data=True) if d.get("type") == "topic"
        }
        self.sub_lbl2nid = {
            d["label"]: n for n, d in self.graph.nodes(data=True) if d.get("type") == "subtopic"
        }

        self.thread_workers = thread_workers

    def retrieve(self, query: str, top_k1: int = None, top_k2: int = None) -> Dict[str, List[str]]:
        # 기본값 설정
        if top_k1 is None:
            from config import get_config
            config = get_config()
            top_k1 = config.top_k1
        if top_k2 is None:
            from config import get_config
            config = get_config()
            top_k2 = config.top_k2
            
        print("=== Retrieval ===")
        topics = choose_topics_from_graph(query, self.graph, self.client)
        print("topics:", topics)

        chosen_subtopics: dict[str, List[str]] = defaultdict(list)
        entities: Set[str] = set()

        def _process_topic(t: str):
            t_id = self.topic_lbl2nid.get(t)
            if t_id is None:
                return t, [], set()

            subs_dict = choose_subtopics_for_topic(
                question=query,
                topic_nid=t_id,
                graph=self.graph,
                client=self.client,
            )
            # print(f"Subtopics for {t}:", subs_dict)
            subs = subs_dict

            ent_set: Set[str] = set()
            for sub_lbl in subs:
                sub_id = self.sub_lbl2nid.get(sub_lbl)
                if sub_id:
                    ent_set |= {
                        nb
                        for nb in self.graph.neighbors(sub_id)
                        if self.graph.nodes[nb].get("type") == "entity"
                    }
            return t, subs, ent_set

        max_workers = min(self.thread_workers, len(topics))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_topic, t) for t in topics]
            for fut in as_completed(futures):
                t, subs, ent_set = fut.result()
                chosen_subtopics[t] = subs
                entities |= ent_set

        if not entities:
            print("🚫 no entities → abort")
            return {}

        edges = self.embedder.search(query, top_k=top_k1, filter_entities=entities)

        chunk_ids: List[str] = []
        seen: Set[str] = set()

        for e in edges:
            raw_id = e.get("chunk_id")
            chunk_id = None

            if isinstance(raw_id, int):
                if 0 <= raw_id < len(self.chunk_id_list):
                    chunk_id = self.chunk_id_list[raw_id]
            elif isinstance(raw_id, str) and raw_id.isdigit():
                idx = int(raw_id)
                if 0 <= idx < len(self.chunk_id_list):
                    chunk_id = self.chunk_id_list[idx]
            elif isinstance(raw_id, str) and raw_id.startswith("chunk-"):
                chunk_id = raw_id

            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
                if len(chunk_ids) == top_k2:
                    break

        print(f"🗂  returning {len(chunk_ids)} chunk IDs\n")

        simplified_edges = []
        for e in edges:
            chunk_id = None
            raw_id = e.get("chunk_id")
            if isinstance(raw_id, int):
                if 0 <= raw_id < len(self.chunk_id_list):
                    chunk_id = self.chunk_id_list[raw_id]
            elif isinstance(raw_id, str) and raw_id.isdigit():
                idx = int(raw_id)
                if 0 <= idx < len(self.chunk_id_list):
                    chunk_id = self.chunk_id_list[idx]
            elif isinstance(raw_id, str) and raw_id.startswith("chunk-"):
                chunk_id = raw_id

            simplified_edges.append({
                "source": e.get("source"),
                "target": e.get("target"),
                "sentence": e.get("sentence"),
                "score": e.get("score"),
                "rank": e.get("rank"),
                "chunk_id": chunk_id
            })

        return {
            "chunks": chunk_ids,
            "edges": simplified_edges,
            "topics": topics,
            "subtopics": chosen_subtopics,
        }

if __name__ == "__main__":
    retriever = Retriever(
        gexf_path=GEXF_PATH,
        kv_json_path=KV_JSON_PATH,
        json_path=GRAPH_JSON_PATH,
        index_path=INDEX_PATH,
        payload_path=PAYLOAD_PATH,
        embedding_model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        client=OpenAI(api_key=OPENAI_API_KEY),
    )
    with open("MultihopRAG/qa.json", encoding="utf-8") as f:
        qa_list = json.load(f)

    for i in range(10, 15):
        query = qa_list[i]["query"]
        print(f"\nQuery {i+1}: {query}")
        res = retriever.retrieve(
            query,
            top_k1=50,
            top_k2=10,
        )
        print(res["subtopics"])