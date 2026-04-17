"""FAISS edge embedding utilities for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import faiss
import networkx as nx
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from config import THRAGConfig, get_config

if "SSL_CERT_FILE" in os.environ:
    os.environ.pop("SSL_CERT_FILE")



def build_sent2chunk(graph_json_path: str) -> dict[str, str]:
    """Map extracted evidence sentences to their originating chunk IDs."""

    with Path(graph_json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    mapping: dict[str, str] = {}
    if not isinstance(payload, list):
        return mapping

    for block in payload:
        if not isinstance(block, dict):
            continue
        chunk_id = str(block.get("chunk_id", ""))
        for item in block.get("triples", []):
            if not isinstance(item, dict):
                continue
            sentence_value = item.get("sentence")
            if isinstance(sentence_value, str) and sentence_value.strip():
                mapping[sentence_value.strip()] = chunk_id
            elif isinstance(sentence_value, list):
                for sentence in sentence_value:
                    if isinstance(sentence, str) and sentence.strip():
                        mapping[sentence.strip()] = chunk_id
    return mapping


class EdgeEmbedderFAISS:
    """Build and query a FAISS index over predicate-edge evidence sentences."""

    def __init__(
        self,
        gexf_path: str,
        json_path: str,
        embedding_model: str,
        openai_api_key: str | None,
        index_path: str,
        payload_path: str,
        client: OpenAI | None = None,
    ) -> None:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be configured before building embeddings.")

        self.graph = nx.read_gexf(gexf_path)
        self.embedding_model = embedding_model
        self.client = client or OpenAI(api_key=openai_api_key)
        self.index_path = str(index_path)
        self.payload_path = str(payload_path)
        self.sent2chunk = build_sent2chunk(json_path)

        self.index: faiss.IndexFlatIP | None = None
        self.payloads: list[dict[str, Any]] = []
        self.edge_records = self._collect_edge_records()

    def _collect_edge_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_sentences: set[str] = set()

        for source_id, target_id, data in self.graph.edges(data=True):
            if data.get("relation_type") != "predicate_relation":
                continue

            sentence_block = str(data.get("sentence", "")).strip()
            if not sentence_block:
                continue

            source_label = str(self.graph.nodes[source_id].get("label", source_id))
            target_label = str(self.graph.nodes[target_id].get("label", target_id))
            edge_label = str(data.get("label", "")).strip()

            for sentence in [part.strip() for part in sentence_block.split(" / ") if part.strip()]:
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                records.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "source": source_label,
                        "target": target_label,
                        "label": edge_label,
                        "sentence": sentence,
                        "chunk_id": self.sent2chunk.get(sentence),
                    }
                )

        return records

    def _embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(input=[text], model=self.embedding_model)
        embedding = np.array(response.data[0].embedding, dtype="float32")
        norm = np.linalg.norm(embedding)
        return embedding if norm == 0 else embedding / norm

    def build_index(self, max_workers: int = 4) -> None:
        if not self.edge_records:
            raise ValueError("No predicate-edge sentences were found in the graph.")

        first_vector = self._embed(self.edge_records[0]["sentence"])
        self.index = faiss.IndexFlatIP(len(first_vector))

        vectors: list[np.ndarray] = [first_vector]
        payloads: list[dict[str, Any]] = [self.edge_records[0]]

        remaining = self.edge_records[1:]

        def embed_record(record: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
            return self._embed(record["sentence"]), record

        if remaining:
            worker_count = max(1, max_workers)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for vector, payload in tqdm(
                    executor.map(embed_record, remaining),
                    total=len(remaining),
                    desc="Embedding predicate edges",
                ):
                    vectors.append(vector)
                    payloads.append(payload)

        self.index.add(np.vstack(vectors))
        self.payloads = payloads
        faiss.write_index(self.index, self.index_path)
        np.save(self.payload_path, np.array(self.payloads, dtype=object))

    def load_index(self) -> None:
        self.index = faiss.read_index(self.index_path)
        self.payloads = np.load(self.payload_path, allow_pickle=True).tolist()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_entities: set[str] | None = None,
        overretrieve: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            self.load_index()
        if self.index is None:
            raise ValueError("The FAISS index is not available.")

        config = get_config()
        top_k = top_k or config.embedding_top_k
        overretrieve = overretrieve or config.overretrieve_factor
        search_k = min(self.index.ntotal, top_k * overretrieve if filter_entities else top_k)
        if search_k == 0:
            return []

        query_vector = self._embed(query).reshape(1, -1)
        distances, indices = self.index.search(query_vector, search_k)

        results: list[dict[str, Any]] = []
        for distance, payload_index in zip(distances[0], indices[0], strict=False):
            if payload_index < 0:
                continue
            payload = self.payloads[payload_index]
            if filter_entities and (
                payload["source_id"] not in filter_entities and payload["target_id"] not in filter_entities
            ):
                continue

            results.append(
                {
                    "source_id": payload["source_id"],
                    "target_id": payload["target_id"],
                    "source": payload["source"],
                    "target": payload["target"],
                    "label": payload["label"],
                    "sentence": payload["sentence"],
                    "chunk_id": payload.get("chunk_id"),
                    "score": float(distance),
                    "rank": len(results) + 1,
                }
            )
            if len(results) >= top_k:
                break

        return results



def build_index_for_dataset(dataset_name: str, rebuild: bool = False) -> str:
    config = get_config(dataset_name)
    graph_path = config.get_graph_gexf_file()
    graph_json_path = config.get_graph_json_file()
    if not graph_path.exists():
        raise FileNotFoundError(f"GEXF graph not found: {graph_path}")
    if not graph_json_path.exists():
        raise FileNotFoundError(f"Graph JSON not found: {graph_json_path}")

    embedder = EdgeEmbedderFAISS(
        gexf_path=str(graph_path),
        json_path=str(graph_json_path),
        embedding_model=config.embed_model,
        openai_api_key=config.openai_api_key,
        index_path=str(config.get_edge_index_file()),
        payload_path=str(config.get_edge_payload_file()),
    )

    if rebuild or not config.get_edge_index_file().exists() or not config.get_edge_payload_file().exists():
        embedder.build_index(max_workers=config.max_workers)
        config.mark_step_completed(
            "edge_embedding",
            index_file=str(config.get_edge_index_file()),
            payload_file=str(config.get_edge_payload_file()),
        )
    return str(config.get_edge_index_file())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or rebuild the FAISS edge index for a dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset>/")
    parser.add_argument("--rebuild", action="store_true", help="Force a full FAISS rebuild.")
    args = parser.parse_args()
    output_path = build_index_for_dataset(dataset_name=args.dataset, rebuild=args.rebuild)
    print(f"Edge index ready at {output_path}")

