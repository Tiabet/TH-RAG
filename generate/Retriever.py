"""Retriever implementation for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import networkx as nx
from openai import OpenAI

from config import get_config
from index.edge_embedding import EdgeEmbedderFAISS
from index.subtopic_choice import choose_subtopics_for_topic
from index.topic_choice import choose_topics_from_graph


class Retriever:
    """Topic-aware graph retriever that narrows edge search with graph structure."""

    def __init__(
        self,
        *,
        gexf_path: str,
        json_path: str,
        kv_json_path: str,
        index_path: str,
        payload_path: str,
        embedding_model: str,
        openai_api_key: str | None,
        client: OpenAI | None = None,
        thread_workers: int | None = None,
    ) -> None:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be configured before retrieval can run.")

        self.graph = nx.read_gexf(gexf_path)
        self.client = client or OpenAI(api_key=openai_api_key)
        self.embedder = EdgeEmbedderFAISS(
            gexf_path=gexf_path,
            json_path=json_path,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            index_path=index_path,
            payload_path=payload_path,
            client=self.client,
        )
        self.embedder.load_index()

        self.topic_label_to_id = {
            data.get("label"): node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "topic"
        }
        self.subtopic_label_to_id = {
            data.get("label"): node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "subtopic"
        }
        self.thread_workers = thread_workers or get_config().max_workers

    def _collect_entity_filter(self, query: str, topics: list[str]) -> tuple[dict[str, list[str]], set[str]]:
        chosen_subtopics: dict[str, list[str]] = defaultdict(list)
        entities: set[str] = set()

        def process_topic(topic_label: str) -> tuple[str, list[str], set[str]]:
            topic_id = self.topic_label_to_id.get(topic_label)
            if topic_id is None:
                return topic_label, [], set()

            subtopics = choose_subtopics_for_topic(
                question=query,
                topic_nid=topic_id,
                graph=self.graph,
                client=self.client,
            )
            entity_ids: set[str] = set()
            for subtopic_label in subtopics:
                subtopic_id = self.subtopic_label_to_id.get(subtopic_label)
                if subtopic_id is None:
                    continue
                entity_ids.update(
                    neighbor
                    for neighbor in self.graph.neighbors(subtopic_id)
                    if self.graph.nodes[neighbor].get("type") == "entity"
                )
            return topic_label, subtopics, entity_ids

        if not topics:
            return chosen_subtopics, entities

        worker_count = max(1, min(self.thread_workers, len(topics)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_topic, topic) for topic in topics]
            for future in as_completed(futures):
                topic_label, subtopics, entity_ids = future.result()
                chosen_subtopics[topic_label] = subtopics
                entities.update(entity_ids)

        return chosen_subtopics, entities

    def retrieve(
        self,
        query: str,
        top_k1: int | None = None,
        top_k2: int | None = None,
    ) -> dict[str, Any]:
        config = get_config()
        top_k1 = top_k1 or config.top_k1
        top_k2 = top_k2 or config.top_k2

        topics = choose_topics_from_graph(query, self.graph, self.client)
        chosen_subtopics, entity_filter = self._collect_entity_filter(query, topics)

        edges = self.embedder.search(query, top_k=top_k1, filter_entities=entity_filter or None)
        if not edges and entity_filter:
            edges = self.embedder.search(query, top_k=top_k1, filter_entities=None)

        chunk_ids: list[str] = []
        seen_chunk_ids: set[str] = set()
        for edge in edges:
            chunk_id = edge.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            chunk_ids.append(chunk_id)
            if len(chunk_ids) >= top_k2:
                break

        simplified_edges = [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "label": edge.get("label"),
                "sentence": edge.get("sentence"),
                "score": edge.get("score"),
                "rank": edge.get("rank"),
                "chunk_id": edge.get("chunk_id"),
            }
            for edge in edges
        ]

        return {
            "chunks": chunk_ids,
            "edges": simplified_edges,
            "topics": topics,
            "subtopics": dict(chosen_subtopics),
        }

