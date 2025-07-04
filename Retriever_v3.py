from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Set

import networkx as nx
from openai import OpenAI

from edge_embedding import EdgeEmbedderFAISS
from topic_choice import choose_topics_from_graph
from subtopic_choice import choose_subtopics_for_topic

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def build_label_index(graph: nx.Graph, node_type: str) -> Dict[str, str]:
    """Return mapping *label → nid* for nodes of ``node_type`` (assumes unique)."""
    idx = {}
    for nid, data in graph.nodes(data=True):
        if data.get("type") == node_type:
            lbl = data.get("label", "")
            if lbl:
                idx[lbl] = nid
    return idx


def entities_of_subtopic(graph: nx.Graph, sub_nid: str) -> Set[str]:
    """Return entity neighbour node‑ids of a subtopic node."""
    return {
        nbr
        for nbr in graph.neighbors(sub_nid)
        if graph.nodes[nbr].get("type") == "entity"
    }


def collect_edge_sentences(graph: nx.Graph, entity_nids: Set[str]) -> List[str]:
    """Gather sentences on edges incident to *entity_nids*."""
    sentences = []
    for ent in entity_nids:
        for _u, _v, data in graph.edges(ent, data=True):
            txt = data.get("sentence", "")
            if txt:
                sentences += [t.strip() for t in txt.split("/") if t.strip()]
    return sentences

# ---------------------------------------------------------------------------
# Main Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(
        self,
        gexf_path: str,
        embedding_model: str,
        openai_api_key: str,
        index_path: str,
        payload_path: str,
        client: OpenAI | None = None,
    ) -> None:
        print(f"📖  Loading graph from {gexf_path} …", end=" ")
        self.graph = nx.read_gexf(gexf_path)
        print(f"done. ({len(self.graph.nodes())} nodes)")

        self.client = client or OpenAI(api_key=openai_api_key)

        # Pre‑compute label→nid look‑ups for speed
        self.topic_lbl2nid = build_label_index(self.graph, "topic")
        self.sub_lbl2nid   = build_label_index(self.graph, "subtopic")

        # Edge sentence cache (subtopic → entity)
        self.subtopic_entity_edges: list[tuple[str, str, dict]] = []

        # FAISS embedder (pre‑built index expected)
        self.embedder = EdgeEmbedderFAISS(
            gexf_path=gexf_path,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            index_path=index_path,
            payload_path=payload_path,
        )
        if os.path.exists(index_path):
            self.embedder.load_index()
            print("✅ FAISS index & payloads loaded.")
        else:
            print("ℹ️  FAISS index not found – building from scratch (one‑time cost)…")
            self.embedder.build_index()
            print("✅ FAISS index built.")

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 50) -> Dict[str, List[str]]:
        print("\n=== Retrieval Start ===")

        # 1) Topic selection (LLM)
        topics = choose_topics_from_graph(query, self.graph, self.client)
        print("Chosen topics:", topics)

        all_entities: Set[str] = set()
        chosen_subtopics: dict[str, List[str]] = defaultdict(list)

        # 2) For each topic → subtopic selection → entity expansion
        for t_lbl in topics:
            t_nid = self.topic_lbl2nid.get(t_lbl)
            if t_nid is None:
                print(f"⚠️  Topic label '{t_lbl}' missing in graph – skipped.")
                continue

            sub_lbls = choose_subtopics_for_topic(
                question=query,
                topic_nid=t_nid,
                graph=self.graph,
                client=self.client,
            )
            chosen_subtopics[t_lbl] = sub_lbls
            print(f"  • {t_lbl}: {sub_lbls}")

            # Convert subtopic labels → nids (only neighbours of t_nid for safety)
            neighbour_sub = {
                nbr
                for nbr in self.graph.neighbors(t_nid)
                if self.graph.nodes[nbr].get("type") == "subtopic"
                and self.graph.nodes[nbr].get("label") in sub_lbls
            }
            for sub_nid in neighbour_sub:
                all_entities |= entities_of_subtopic(self.graph, sub_nid)

        if not all_entities:
            print("🚫  No entities found – retrieval aborted.")
            return {"faiss_results": []}

        print(f"Total entities aggregated: {len(all_entities)}")

        # 3) (Optional) statistics – gather sentences count
        sentences = collect_edge_sentences(self.graph, all_entities)
        print(f"Edge sentences collected: {len(sentences)}")

        # 4) FAISS search filtered by entity set
        results = self.embedder.search(
            query=query,
            top_k=top_k,
            filter_entities=all_entities,
        )
        print(f"Returned {len(results)} edges from FAISS")
        print("=== Retrieval End ===")

        # Metadata caches (could be used downstream)
        self.chosen_topics = topics
        self.chosen_subtopics = chosen_subtopics
        self.collected_entities = all_entities

        return {"faiss_results": results}