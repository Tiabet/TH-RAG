#!/usr/bin/env python3
"""Run the new Retriever pipeline and export a focused sub‑graph as GEXF.

Adapts the original *run_retrieve_and_visualize.py* to work with the revamped
`retriever.py` (LLM‑driven topic & subtopic selection).
"""

from __future__ import annotations

import os
import networkx as nx
import openai
from dotenv import load_dotenv
import time

from Retriever_v3 import Retriever  # ← 새 구현

# ---------------------------------------------------------------------------
# User‑configurable paths
# ---------------------------------------------------------------------------
GEXF_PATH = "InfiniteQA/graph_v1.gexf"
INDEX_PATH = "InfiniteQA/edge_index_v1.faiss"
PAYLOAD_PATH = "InfiniteQA/edge_payloads_v1.npy"
EMBEDDING_MODEL = "text-embedding-3-small"
OUTPUT_GEXF = "InfiniteQA/test/10years.gexf"
QUERY = (
    "How many years does Pete take to complete the painting of Mrs. Bronwyn?"  
)
TOP_K = 10000  # FAISS edges to retrieve

# ---------------------------------------------------------------------------
# Environment & OpenAI client
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("환경 변수 'OPENAI_API_KEY'가 설정되지 않았습니다.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Initialise Retriever
# ---------------------------------------------------------------------------
retriever = Retriever(
    gexf_path=GEXF_PATH,
    embedding_model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    index_path=INDEX_PATH,
    payload_path=PAYLOAD_PATH,
    client=client,
)
start_time = time.time()
# ---------------------------------------------------------------------------
# Run retrieval
# ---------------------------------------------------------------------------
results = retriever.retrieve(query=QUERY, top_k=TOP_K)

end_time = time.time()
print(f"Retrieval completed in {end_time - start_time:.2f} seconds.")
# --------------------  -------------------------------------------------------
# 1) Add nodes & FAISS edges from search hits
# ---------------------------------------------------------------------------
selected_nodes: set[str] = set()
selected_edges: list[tuple[str, str, dict]] = []

for hit in results["faiss_results"]:
    src, tgt = hit["source"], hit["target"]
    selected_nodes.update([src, tgt])
    selected_edges.append(
        (
            src,
            tgt,
            {
                "label": hit.get("label", ""),
                "sentence": hit.get("sentence", ""),
                "faiss_score": hit["score"],
                "rank": hit["rank"],
            },
        )
    )

# ---------------------------------------------------------------------------
# 2) Include chosen subtopics + (subtopic ↔ entity) edges
# ---------------------------------------------------------------------------
G = retriever.graph  # shorthand

# Build label→nid index for subtopics once
sub_lbl2nid = {
    data.get("label", ""): nid
    for nid, data in G.nodes(data=True)
    if data.get("type") == "subtopic"
}

for topic_lbl, sub_lbls in retriever.chosen_subtopics.items():
    for sub_lbl in sub_lbls:
        sub_nid = sub_lbl2nid.get(sub_lbl)
        if not sub_nid:
            continue
        selected_nodes.add(sub_nid)

        # gather edges subtopic‑entity
        for nbr in G.neighbors(sub_nid):
            if G.nodes[nbr].get("type") == "entity":
                edge_data = G.get_edge_data(sub_nid, nbr)
                selected_nodes.add(nbr)
                selected_edges.append(
                    (
                        sub_nid,
                        nbr,
                        {
                            "label": edge_data.get("label", "subtopic‑entity"),
                            "sentence": edge_data.get("sentence", ""),
                        },
                    )
                )

# ---------------------------------------------------------------------------
# 3) Build sub‑graph and export
# ---------------------------------------------------------------------------
subgraph = nx.Graph()
subgraph.add_nodes_from(selected_nodes)
subgraph.add_edges_from(selected_edges)

print(f"✅ 노드 수: {len(subgraph.nodes)} / 엣지 수: {len(subgraph.edges)}")

output_dir = os.path.dirname(OUTPUT_GEXF)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

nx.write_gexf(subgraph, OUTPUT_GEXF)
print(f"📁 저장 완료: {OUTPUT_GEXF}")
