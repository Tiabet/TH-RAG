# retriever_chunk.py  ── 핵심 부분만
from __future__ import annotations
import json, os, re
from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx
from openai import OpenAI
from edge_embedding import EdgeEmbedderFAISS
from topic_choice    import choose_topics_from_graph
from subtopic_choice import choose_subtopics_for_topic
from dotenv import load_dotenv

load_dotenv()

# ---------- 하드코딩/경로 ----------
GEXF_PATH        = "hotpotQA/graph_v1.gexf"
CHUNKS_PATH      = "hotpotQA/chunks_v1.txt"      # 한 줄‑한 청크
GRAPH_JSON_PATH  = "hotpotQA/graph_v1.json"      # sentence + chunk_id
INDEX_PATH       = "hotpotQA/edge_index_v1.faiss"
PAYLOAD_PATH     = "hotpotQA/edge_payloads_v1.npy"
EMBEDDING_MODEL  = "text-embedding-3-small"
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
# -----------------------------------

# ── util ──────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
        return " ".join(re.sub(r"\s+", " ", s.strip()).split()).lower()

def build_sent2chunk(path: str) -> Dict[str, int]:
    """
    JSON 구조:
    [
      { "triples":[ {...}, {...} ], "chunk_id": 0 },
      { "triples":[ {...}, ... ],   "chunk_id": 1 },
      ...
    ]
    ⇒  sentence → chunk_id 매핑 반환
    """
    import json, re


    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, int] = {}

    for block in data:
        if not isinstance(block, dict):
            continue
        cid = block.get("chunk_id")
        triples = block.get("triples")
        if cid is None or not isinstance(triples, list):
            continue                       # chunk_id 없으면 건너뜀
        for item in triples:
            sent = item.get("sentence")
            if isinstance(sent, str):
                mapping[normalize(sent)] = cid

    if not mapping:
        raise ValueError("❌  No sentence→chunk_id pairs found.")

    return mapping
# ──────────────────────────────────────────────────────────────────────


class Retriever:
    def __init__(
        self,
        gexf_path:       str,
        chunks_path:     str,
        graph_json_path: str,
        index_path:      str,
        payload_path:    str,
        embedding_model: str,
        openai_api_key:  str,
        client: OpenAI | None = None,
    ) -> None:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        # 0) 로드: 그래프, 청크, 문장→청크
        print("📖  loading graph …", end=" ")
        self.graph = nx.read_gexf(gexf_path)
        print(f"done ({self.graph.number_of_nodes()} nodes)")

        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = [ln.rstrip("\n") for ln in f]
        print(f"📚  {len(self.chunks)} chunks loaded")

        self.sent2cid = build_sent2chunk(graph_json_path)
        print(f"🔗  sentence→chunk map: {len(self.sent2cid)} entries")

        self.client = client

        # FAISS
        self.embedder = EdgeEmbedderFAISS(
            gexf_path=gexf_path,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            index_path=index_path,
            payload_path=payload_path,
        )
        if os.path.exists(index_path):
            self.embedder.load_index()
            print("✅  FAISS index loaded\n")

        # 미리 topic/subtopic 인덱스
        self.topic_lbl2nid = {d["label"]: n for n, d in self.graph.nodes(data=True) if d.get("type") == "topic"}
        self.sub_lbl2nid   = {d["label"]: n for n, d in self.graph.nodes(data=True) if d.get("type") == "subtopic"}

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k1: int = 50, top_k2: int = 5) -> Dict[str, List[str]]:
        print("=== Retrieval ===") 
        topics = choose_topics_from_graph(query, self.graph, self.client)
        print("topics:", topics)

        chosen_subtopics: dict[str, List[str]] = defaultdict(list)
        entities: Set[str] = set()
        for t in topics:
            t_id = self.topic_lbl2nid.get(t)
            if t_id is None: continue
            subs = choose_subtopics_for_topic(question = query, topic_nid =t_id, graph = self.graph, client = self.client)
            chosen_subtopics[t] = subs
            for sub_lbl in subs:
                sub_id = self.sub_lbl2nid.get(sub_lbl)
                if sub_id:
                    entities |= {
                        nb for nb in self.graph.neighbors(sub_id)
                        if self.graph.nodes[nb].get("type") == "entity"
                    }

        if not entities:
            print("🚫 no entities → abort")
            return {}

        # 🔍 Step 1: FAISS에서 top_k1개 문장 검색
        edges = self.embedder.search(query, top_k=top_k1, filter_entities=entities)

        # 🧠 Step 2: 상위 top_k2개의 문장 → 해당 chunk만 추출
        chunk_ids: List[int] = []
        seen = set()
        count = 0
        for e in edges:  # FAISS 랭킹 순서 유지
            cid = self.sent2cid.get(normalize(e["sentence"]))
            if cid is not None:
                e["chunk_id"] = cid
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
                    count += 1
            else:
                print(f"⚠️ unmatched sentence: {e['sentence'][:80]}…")
            if count == top_k2:
                break

        chunks_text = [self.chunks[c] for c in chunk_ids]
        print(f"🗂  returning {len(chunks_text)} chunks\n")

        simplified_edges = [
            {
                "source":   e.get("source"),
                "target":   e.get("target"),
                "label":    e.get("label"),
                "sentence": e.get("sentence"),
                "score":    e.get("score"),
                "rank":     e.get("rank"),
            }
            for e in edges
        ]

        return {
            "chunks": chunks_text,       # 상위 문장의 chunk만
            "edges": simplified_edges,              # top_k1 전체 문장
            "topics": topics,
            "subtopics": chosen_subtopics,
        }



if __name__ == "__main__":
    from openai import OpenAI
    
    retriever = Retriever(
        gexf_path=GEXF_PATH,
        chunks_path=CHUNKS_PATH,
        graph_json_path=GRAPH_JSON_PATH,
        index_path=INDEX_PATH,
        payload_path=PAYLOAD_PATH,
        embedding_model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        client=OpenAI(api_key=OPENAI_API_KEY),
    )

    res = retriever.retrieve(
        "In which election did Norm Coleman won to become the last republican elected as of 2017?",
        top_k1=50,
        top_k2=10,
    )
    print(res)