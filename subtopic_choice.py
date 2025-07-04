"""
Subtopic-selection helper using an LLM **with automatic retry logic**.

* Supplies the LLM with only the subtopic labels that are direct children of a
  given topic node (type=="subtopic" and 1‑hop neighbour of the topic).
* The LLM must choose 1‑10 relevant subtopics. If the response is malformed or
  the chosen list is empty/invalid, the call is retried up to ``MAX_RETRIES``
  times before giving up and returning an empty list.
"""

from __future__ import annotations

import json
import time
from typing import List, Tuple

import networkx as nx
from openai import OpenAI
from prompt.subtopic_choice import SUBTOPIC_CHOICE_PROMPT

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3          # ← configurable global
RETRY_BACKOFF = 2.0      # seconds between retries

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def extract_subtopics_for_topic(graph: nx.Graph, topic_nid: str) -> List[Tuple[str, str]]:
    """Return ``[(sub_nid, sub_label), ...]`` directly connected to *topic_nid*."""
    return [
        (nbr, graph.nodes[nbr].get("label", ""))
        for nbr in graph.neighbors(topic_nid)
        if graph.nodes[nbr].get("type") == "subtopic"
    ]

# ---------------------------------------------------------------------------
# Core LLM selector – with retries
# ---------------------------------------------------------------------------

def choose_subtopics_for_topic(
    *,
    question: str,
    topic_nid: str,
    graph: nx.Graph,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_subtopics: int = 10,
) -> List[str]:
    """Return up to ``max_subtopics`` relevant subtopic **labels** for *topic_nid*.

    The function will retry ``MAX_RETRIES`` times if the LLM response is not
    valid JSON, the chosen list is empty, or contains items outside the allowed
    list. On final failure it returns an empty list instead of raising.
    """

    if graph.nodes[topic_nid].get("type") != "topic":
        raise ValueError(f"Node {topic_nid} is not of type 'topic'.")

    # --------------------------------------------------
    # 1) Gather candidate subtopics
    # --------------------------------------------------
    sub_nodes = extract_subtopics_for_topic(graph, topic_nid)
    if not sub_nodes:
        return []
    sub_labels = [lbl for _nid, lbl in sub_nodes]

    # --------------------------------------------------
    # 2) Build prompt (static across retries)
    # --------------------------------------------------
    prompt = (
        SUBTOPIC_CHOICE_PROMPT
        .replace("{TOPIC_LABEL}", graph.nodes[topic_nid].get("label", ""))
        .replace("{SUBTOPIC_LIST}", json.dumps(sub_labels, ensure_ascii=False))
        .replace("{question}", question)
    )

    # --------------------------------------------------
    # 3) Retry loop
    # --------------------------------------------------
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content

            data = json.loads(content)
            chosen = data.get("subtopics")

            if (
                isinstance(chosen, list)
                and 1 <= len(chosen) <= max_subtopics
                and all(lbl in sub_labels for lbl in chosen)
            ):
                # Preserve original order of sub_labels
                return [lbl for lbl in sub_labels if lbl in chosen]

            print(
                f"⚠️  Attempt {attempt}: invalid chosen list → {chosen}. Retrying…"
            )
        except (json.JSONDecodeError, KeyError) as exc:
            print(
                f"⚠️  Attempt {attempt}: JSON parse/format error → {exc}. Retrying…"
            )
        except Exception as exc:  # catch‑all for OpenAI errors
            print(f"⚠️  Attempt {attempt}: OpenAI error → {exc}. Retrying…")

        # back‑off before next retry (unless last)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)

    # --------------------------------------------------
    # 4) Final fallback – return empty list
    # --------------------------------------------------
    print("🚫  All retries exhausted – returning empty subtopic list.")
    return []

# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    gexf_path = "hotpotQA/graph_v1.gexf"
    if not os.path.exists(gexf_path):
        raise SystemExit("Place a graph_v1.gexf or set GEXF_PATH env var.")

    G = nx.read_gexf(gexf_path)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # topic_id = next(n for n, d in G.nodes(data=True) if d.get("type") == "topic")
    topic_id = "topic_entertainment"
    print("Topic:", G.nodes[topic_id]["label"])

    subs = choose_subtopics_for_topic(
        question="Which American comedian born on March 21, 1962, appeared in the movie 'Sleepless in Seattle'?",
        topic_nid=topic_id,
        graph=G,
        client=client,
    )
    print("Chosen subtopics:", subs)
