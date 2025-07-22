"""Thread‑powered version of `retriever_chunk.py`
-------------------------------------------------
Uses `concurrent.futures.ThreadPoolExecutor` to run *subtopic selection* in
parallel for each chosen topic.  All other logic is unchanged.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed  # 🆕

import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

from edge_embedding import EdgeEmbedderFAISS
from topic_choice import choose_topics_from_graph
from subtopic_choice import choose_subtopics_for_topic
from json_to_gexf import clean_text_for_xml

load_dotenv()


# ---------- 하드코딩/경로 ----------
GEXF_PATH        = "hotpotQA/graph_v1.gexf"
CHUNKS_PATH      = "hotpotQA/chunks.txt"      # 한 줄‑한 청크
GRAPH_JSON_PATH  = "hotpotQA/graph_v1.json"      # sentence + chunk_id
INDEX_PATH       = "hotpotQA/edge_index_v1.faiss"
PAYLOAD_PATH     = "hotpotQA/edge_payloads_v1.npy"
EMBEDDING_MODEL  = "text-embedding-3-small"
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
# -----------------------------------

# ──────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """대소문자·공백·구두점 최소 정규화."""
    s = re.sub(r"\s+", " ", s.strip())   # 줄바꿈·다중 공백 → 1칸
    return s.lower()


def build_sent2chunk(path: str) -> Dict[str, int]:
    """Build {sentence → chunk_id} dict from the graph‑JSON payload."""

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, int] = {}

    for block in data:
        if not isinstance(block, dict):
            continue
        cid = block.get("chunk_id")
        triples = block.get("triples")
        if cid is None or not isinstance(triples, list):
            continue
        for item in triples:
            sent = item.get("sentence")
            if isinstance(sent, str):
                mapping[normalize(clean_text_for_xml(sent))] = cid

    if not mapping:
        raise ValueError("❌  No sentence→chunk_id pairs found.")

    return mapping


# ──────────────────────────────────────────────────────────────────────
# Retriever class – now with thread‑level parallelism
# ──────────────────────────────────────────────────────────────────────

class Retriever:
    def __init__(
        self,
        gexf_path: str,
        chunks_path: str,
        graph_json_path: str,
        index_path: str,
        payload_path: str,
        embedding_model: str,
        openai_api_key: str,
        client: OpenAI | None = None,
        *,
        thread_workers: int = 8,  # 🆕 maximum threads for subtopic selection
    ) -> None:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")

        # 0) Load graph / chunks / sentence→chunk map
        print("📖  loading graph …", end=" ")
        self.graph = nx.read_gexf(gexf_path)
        print(f"done ({self.graph.number_of_nodes()} nodes)")

        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = [ln.rstrip("\n") for ln in f]
        print(f"📚  {len(self.chunks)} chunks loaded")

        self.sent2cid = build_sent2chunk(graph_json_path)
        print(f"🔗  sentence→chunk map: {len(self.sent2cid)} entries")

        self.client = client or OpenAI(api_key=openai_api_key)

        # FAISS embedder
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

        # Quick index for label→node‑id mapping
        self.topic_lbl2nid = {
            d["label"]: n for n, d in self.graph.nodes(data=True) if d.get("type") == "topic"
        }
        self.sub_lbl2nid = {
            d["label"]: n for n, d in self.graph.nodes(data=True) if d.get("type") == "subtopic"
        }

        self.thread_workers = thread_workers

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k1: int = 50, top_k2: int = 5) -> Dict[str, List[str]]:
        print("=== Retrieval ===")
        topics = choose_topics_from_graph(query, self.graph, self.client)
        print("topics:", topics)

        chosen_subtopics: dict[str, List[str]] = defaultdict(list)
        entities: Set[str] = set()

        # ───────── Threaded subtopic selection ──────────
        def _process_topic(t: str):
            """Select subtopics + collect connected entities for a single topic."""
            t_id = self.topic_lbl2nid.get(t)
            if t_id is None:
                return t, [], set()

            subs = choose_subtopics_for_topic(
                question=query,
                topic_nid=t_id,
                graph=self.graph,
                client=self.client,
            )

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

        max_workers = min(self.thread_workers, len(topics)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_process_topic, t) for t in topics]
            for fut in as_completed(futures):
                t, subs, ent_set = fut.result()
                chosen_subtopics[t] = subs
                entities |= ent_set

        # ───────── Rest of the original pipeline ──────────
        if not entities:
            print("🚫 no entities → abort")
            return {}

        # 1) FAISS search (edge‑level)
        edges = self.embedder.search(query, top_k=top_k1, filter_entities=entities)

        # 2) Keep only the top_k2 unique chunks
        chunk_ids: List[int] = []
        seen: Set[int] = set()
        count = 0

        def find_chunk_containing_sentence(sentence: str, chunks: List[str]) -> int | None:
            norm_sent = normalize(clean_text_for_xml(sentence))
            for idx, chunk in enumerate(chunks):
                if norm_sent in normalize(clean_text_for_xml(chunk)):
                    return idx
            return None

        for e in edges:  # preserve FAISS ranking order
            normed = normalize(clean_text_for_xml(e["sentence"]))
            cid = self.sent2cid.get(normed)

            if cid is not None:
                e["chunk_id"] = cid
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
                    count += 1
            else:
                # fallback: brute‑search within chunk text
                fallback_cid = find_chunk_containing_sentence(e["sentence"], self.chunks)
                if fallback_cid is not None:
                    e["chunk_id"] = fallback_cid
                    if fallback_cid not in seen:
                        seen.add(fallback_cid)
                        chunk_ids.append(fallback_cid)
                        count += 1
                    print(
                        f"♻️ fallback matched: chunk {fallback_cid} for sentence {e['sentence'][:80]}…"
                    )
                else:
                    print(f"⚠️ unmatched sentence: {e['sentence'][:80]}…")

            if count == top_k2:
                break

        chunks_text = [self.chunks[c] for c in chunk_ids]
        print(f"🗂  returning {len(chunks_text)} chunks\n")

        simplified_edges = [
            {
                "source": e.get("source"),
                "target": e.get("target"),
                "label": e.get("label"),
                "sentence": e.get("sentence"),
                "score": e.get("score"),
                "rank": e.get("rank"),
            }
            for e in edges
        ]

        return {
            "chunks": chunks_text,
            "edges": simplified_edges,
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
        "Of the retired tennis players Ross Case and Slobodan Živojinović, which player reached a higher career-high singles ranking?",
        top_k1=50,
        top_k2=10,
    )
    # print(res)