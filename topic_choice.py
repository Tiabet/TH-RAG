"""
Utility helpers for picking relevant topic labels from a knowledge‑graph topic list
using the LLM prompt template defined in ``prompt/topic_choice.py``.
"""

from __future__ import annotations

import json
from typing import List

import networkx as nx
from openai import OpenAI

# Local prompt template
from prompt.topic_choice import TOPIC_CHOICE_PROMPT

from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"  # same as the rest of the code‑base
TOPIC_CHOICE_MIN = 5
TOPIC_CHOICE_MAX = 10
MAX_RETRIES = 10
# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def extract_graph_topic_labels(graph: nx.Graph) -> List[str]:
    """Return **unique** topic labels from the graph, preserving insertion order."""
    labels_seen = set()
    labels = []
    for _nid, data in graph.nodes(data=True):
        if data.get("type") == "topic":
            lbl = data.get("label", "")
            if lbl and lbl not in labels_seen:
                labels.append(lbl)
                labels_seen.add(lbl)
    return labels

def choose_topics_from_graph(
    question: str,
    graph: nx.Graph,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_topics: int = TOPIC_CHOICE_MAX,
    min_topics: int = TOPIC_CHOICE_MIN,
    max_retries: int = MAX_RETRIES,
) -> List[str]:
    """Ask the LLM to pick up to *max_topics* relevant topics from *graph*.

    If the LLM response is invalid, it will retry up to `max_retries` times
    by re-asking the same prompt.

    Returns
    -------
    List[str]
        List of chosen topic *labels* exactly as they appear in the graph label list.

    Raises
    ------
    ValueError
        If no valid result is obtained after max_retries.
    """

    topic_labels = extract_graph_topic_labels(graph)
    if not topic_labels:
        raise ValueError("Graph contains no topic nodes (type='topic').")

    prompt_str = (
        TOPIC_CHOICE_PROMPT
        .replace("{{TOPIC_LIST}}", json.dumps(topic_labels, ensure_ascii=False))
        .replace("{{question}}", question)
        .replace("{max_topics}", str(max_topics))
        .replace("{min_topics}", str(TOPIC_CHOICE_MIN))
        .replace("{min_topics}", str(min_topics))
    )
    # print(prompt_str)

    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_str},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        content = response.choices[0].message.content
        # print(content)

        try:
            data = json.loads(content)
            chosen = data.get("topics")

            if not isinstance(chosen, list) or not (1 <= len(chosen) <= max_topics):
                raise ValueError("Invalid topic list format or length.")

            invalid = [t for t in chosen if t not in topic_labels]
            if invalid:
                raise ValueError("Returned topics not in topic list.")

            ordered = [lbl for lbl in topic_labels if lbl in chosen]
            return ordered

        except Exception as e:
            print(f"[Attempt {attempt}] Failed to parse or validate response: {e}")
            if attempt == max_retries:
                raise ValueError("Failed to get valid topic list after multiple attempts.")

    raise RuntimeError("Unreachable fallback")  # Just in case

# ---------------------------------------------------------------------------
# Example standalone usage (for local testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import networkx as nx

    # Example: load graph and run a single query
    GEXF_PATH = 'MultihopRAG/graph_v1.gexf'
    if not os.path.exists(GEXF_PATH):
        raise SystemExit("Set GEXF_PATH environment variable or place graph.gexf in cwd.")

    G = nx.read_gexf(GEXF_PATH)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # question = "Which documentary was filmed first, Almost Sunrise or Hail! Hail! Rock 'n' Roll?"
    question = "Which is larger, Hunchun or Shijiazhuang?"
    print("Chosen topics:")
    print(choose_topics_from_graph(question, G, client))
