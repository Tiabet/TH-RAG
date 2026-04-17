"""Convert extracted graph JSON blocks into a GEXF knowledge graph."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx


def clean_id(text: str) -> str:
    """Create a deterministic node identifier fragment."""

    normalized = re.sub(r"\s+", "_", text.strip().lower())
    return re.sub(r"[^a-z0-9_.-]", "_", normalized)



def is_valid_triple(item: dict[str, Any]) -> bool:
    """Validate the minimal triple schema required for graph conversion."""

    return (
        isinstance(item, dict)
        and isinstance(item.get("triple"), list)
        and len(item["triple"]) == 3
        and isinstance(item.get("subject"), dict)
        and isinstance(item.get("object"), dict)
    )



def load_entries(input_file: Path) -> list[dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    blocks: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("triples"), list):
        blocks.append(payload)
    elif isinstance(payload, list):
        blocks.extend(block for block in payload if isinstance(block, dict))
    else:
        raise ValueError("Unsupported graph JSON structure.")

    entries: list[dict[str, Any]] = []
    for block in blocks:
        chunk_id = str(block.get("chunk_id", ""))
        for item in block.get("triples", []):
            if not is_valid_triple(item):
                continue
            entries.append({"chunk_id": chunk_id, **item})
    return entries



def add_labeled_node(graph: nx.Graph, node_id: str, label: str, node_type: str) -> None:
    graph.add_node(node_id, label=label.strip(), type=node_type)



def convert_json_to_gexf(input_file: str, output_file: str | None = None) -> str:
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path.with_suffix(".gexf")

    entries = load_entries(input_path)
    if not entries:
        raise ValueError(f"No valid triples found in {input_path}")

    graph = nx.Graph()
    for entry in entries:
        subject_label, predicate_label, object_label = [str(value).strip() for value in entry["triple"]]
        subject_subtopic = str(entry["subject"].get("subtopic", "")).strip() or "Unknown Subtopic"
        subject_topic = str(entry["subject"].get("main_topic", "")).strip() or "Unknown Topic"
        object_subtopic = str(entry["object"].get("subtopic", "")).strip() or "Unknown Subtopic"
        object_topic = str(entry["object"].get("main_topic", "")).strip() or "Unknown Topic"
        sentence_value = entry.get("sentence", "")
        if isinstance(sentence_value, list):
            sentence = " ".join(str(item).strip() for item in sentence_value if str(item).strip())
        else:
            sentence = str(sentence_value).strip()
        chunk_id = str(entry.get("chunk_id", ""))

        subject_node = f"entity_{clean_id(subject_label)}"
        subject_subtopic_node = f"subtopic_{clean_id(subject_subtopic)}"
        subject_topic_node = f"topic_{clean_id(subject_topic)}"
        object_node = f"entity_{clean_id(object_label)}"
        object_subtopic_node = f"subtopic_{clean_id(object_subtopic)}"
        object_topic_node = f"topic_{clean_id(object_topic)}"

        for node_id, label, node_type in [
            (subject_node, subject_label, "entity"),
            (subject_subtopic_node, subject_subtopic, "subtopic"),
            (subject_topic_node, subject_topic, "topic"),
            (object_node, object_label, "entity"),
            (object_subtopic_node, object_subtopic, "subtopic"),
            (object_topic_node, object_topic, "topic"),
        ]:
            add_labeled_node(graph, node_id, label, node_type)

        graph.add_edge(
            subject_node,
            subject_subtopic_node,
            label="has_subtopic",
            relation_type="subtopic_relation",
        )
        graph.add_edge(
            subject_subtopic_node,
            subject_topic_node,
            label="has_topic",
            relation_type="topic_relation",
        )
        graph.add_edge(
            object_node,
            object_subtopic_node,
            label="has_subtopic",
            relation_type="subtopic_relation",
        )
        graph.add_edge(
            object_subtopic_node,
            object_topic_node,
            label="has_topic",
            relation_type="topic_relation",
        )

        if graph.has_edge(subject_node, object_node):
            edge = graph[subject_node][object_node]
            labels = set(filter(None, str(edge.get("label", "")).split(" / ")))
            sentences = set(filter(None, str(edge.get("sentence", "")).split(" / ")))
            chunk_ids = set(filter(None, str(edge.get("chunk_ids", "")).split(" / ")))

            if predicate_label:
                labels.add(predicate_label)
            if sentence:
                sentences.add(sentence)
            if chunk_id:
                chunk_ids.add(chunk_id)

            edge["label"] = " / ".join(sorted(labels))
            edge["sentence"] = " / ".join(sorted(sentences))
            edge["chunk_ids"] = " / ".join(sorted(chunk_ids))
            edge["relation_type"] = "predicate_relation"
            edge["weight"] = int(edge.get("weight", 1)) + 1
        else:
            graph.add_edge(
                subject_node,
                object_node,
                label=predicate_label,
                relation_type="predicate_relation",
                sentence=sentence,
                chunk_ids=chunk_id,
                weight=1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, output_path)
    print(f"Wrote GEXF graph to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert graph JSON blocks into a GEXF graph.")
    parser.add_argument("input_file", help="Path to the extracted graph JSON file")
    parser.add_argument("output_file", nargs="?", help="Optional output GEXF path")
    args = parser.parse_args()
    convert_json_to_gexf(args.input_file, args.output_file)

