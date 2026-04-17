"""Subtopic selection helpers for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json
import time
from typing import List, Tuple

import networkx as nx
from openai import OpenAI

from config import get_config
from prompt.subtopic_choice import SUBTOPIC_CHOICE_PROMPT

config = get_config()

SUBTOPIC_CHOICE_MIN = config.subtopic_choice_min
SUBTOPIC_CHOICE_MAX = config.subtopic_choice_max
DEFAULT_MODEL = config.default_model
MAX_RETRIES = config.max_retries
RETRY_BACKOFF = config.retry_backoff


def extract_subtopics_for_topic(graph: nx.Graph, topic_node_id: str) -> List[Tuple[str, str]]:
    """Return the direct subtopic neighbors for a topic node."""

    return [
        (neighbor, str(graph.nodes[neighbor].get("label", "")).strip())
        for neighbor in graph.neighbors(topic_node_id)
        if graph.nodes[neighbor].get("type") == "subtopic"
    ]


def choose_subtopics_for_topic(
    *,
    question: str,
    topic_nid: str,
    graph: nx.Graph,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_subtopics: int = SUBTOPIC_CHOICE_MAX,
    min_subtopics: int = SUBTOPIC_CHOICE_MIN,
) -> list[str]:
    """Return the ordered list of subtopics chosen by the LLM."""

    if graph.nodes[topic_nid].get("type") != "topic":
        raise ValueError(f"Node {topic_nid} is not a topic node.")

    subtopics = extract_subtopics_for_topic(graph, topic_nid)
    if not subtopics:
        return []

    subtopic_labels = [label for _node_id, label in subtopics if label]
    if not subtopic_labels:
        return []

    min_subtopics = max(1, min(min_subtopics, len(subtopic_labels)))
    max_subtopics = max(min_subtopics, min(max_subtopics, len(subtopic_labels)))

    prompt = (
        SUBTOPIC_CHOICE_PROMPT
        .replace("{{TOPIC_LABEL}}", str(graph.nodes[topic_nid].get("label", "")))
        .replace("{{SUBTOPIC_LIST}}", json.dumps(subtopic_labels, ensure_ascii=False))
        .replace("{question}", question)
        .replace("{min_subtopics}", str(min_subtopics))
        .replace("{max_subtopics}", str(max_subtopics))
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You select relevant subtopics from a fixed list."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=config.answer_temperature,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            chosen = payload.get("subtopics", [])
            if not isinstance(chosen, list):
                raise ValueError("The model response did not contain a list under 'subtopics'.")

            valid = [label for label in subtopic_labels if label in set(chosen)]
            if valid:
                return valid[:max_subtopics]

            raise ValueError("The model did not return any valid subtopics.")
        except Exception as exc:
            print(f"Subtopic selection attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF)

    print("Subtopic selection fell back to the first available subtopics.")
    return subtopic_labels[:max_subtopics]

