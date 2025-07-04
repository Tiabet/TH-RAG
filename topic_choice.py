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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"  # same as the rest of the code‑base

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
    max_topics: int = 5,
) -> List[str]:
    """Ask the LLM to pick up to *max_topics* relevant topics from *graph*.

    Parameters
    ----------
    question : str
        The user's natural‑language query.
    graph : nx.Graph
        A NetworkX graph with node attribute ``type == 'topic'`` and a ``label``.
    client : OpenAI
        An instantiated ``openai.OpenAI`` client (same pattern as elsewhere).
    model : str, optional
        OpenAI model name (default ``gpt-4o-mini``).
    max_topics : int, optional
        Upper bound of topics to return. **Must** match the prompt spec (≤ 5).

    Returns
    -------
    List[str]
        List of chosen topic *labels* exactly as they appear in the graph label list.

    Raises
    ------
    ValueError
        If LLM output is not valid JSON, or chosen topics are not in the list.
    """

    topic_labels = extract_graph_topic_labels(graph)
    if not topic_labels:
        raise ValueError("Graph contains no topic nodes (type='topic').")

    # Fill the template placeholders
    prompt_str = (
        TOPIC_CHOICE_PROMPT
        .replace("{TOPIC_LIST}", json.dumps(topic_labels, ensure_ascii=False))
        .replace("{question}", question)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_str},
        ],
        # temperature 0 to minimise creative deviations
        temperature=0,
    )

    print(len(topic_labels))

    content = response.choices[0].message.content

    # ----------------- Parse & validate -----------------
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from LLM:\n{content}\nError: {exc}"  # noqa: E501
        )

    chosen = data.get("topics")
    if not isinstance(chosen, list) or not (1 <= len(chosen) <= max_topics):
        raise ValueError(
            f"LLM returned invalid 'topics' list (size): {chosen}"  # noqa: E501
        )

    # Ensure every chosen topic is in the allowed list
    invalid = [t for t in chosen if t not in topic_labels]
    if invalid:
        raise ValueError(
            f"LLM chose topics that are not in the supplied list: {invalid}"  # noqa: E501
        )

    # Preserve original order of topic_labels as required by prompt
    ordered = [lbl for lbl in topic_labels if lbl in chosen]
    return ordered

# ---------------------------------------------------------------------------
# Example standalone usage (for local testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import networkx as nx

    # Example: load graph and run a single query
    GEXF_PATH = 'hotpotQA/graph_v1.gexf'
    if not os.path.exists(GEXF_PATH):
        raise SystemExit("Set GEXF_PATH environment variable or place graph.gexf in cwd.")

    G = nx.read_gexf(GEXF_PATH)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    question = "Which American comedian born on March 21, 1962, appeared in the movie \"Sleepless in Seattle?\""
    print("Chosen topics:")
    print(choose_topics_from_graph(question, G, client))
