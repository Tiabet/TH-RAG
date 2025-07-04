from __future__ import annotations

import json
from typing import List, Tuple

import networkx as nx
from openai import OpenAI
from prompt.subtopic_choice import SUBTOPIC_CHOICE_PROMPT

DEFAULT_MODEL = "gpt-4o-mini"
SUBTOPIC_CHOICE_PROMPT = SUBTOPIC_CHOICE_PROMPT

def extract_subtopics_for_topic(graph: nx.Graph, topic_nid: str) -> List[Tuple[str, str]]:
    """Return ``[(sub_nid, sub_label), ...]`` directly connected to *topic_nid*."""
    subtopics = []
    for nbr in graph.neighbors(topic_nid):
        data = graph.nodes[nbr]
        if data.get("type") == "subtopic":
            subtopics.append((nbr, data.get("label", "")))
    return subtopics


def choose_subtopics_for_topic(
    *,
    question: str,
    topic_nid: str,
    graph: nx.Graph,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_subtopics: int = 10,
) -> List[str]:
    """Ask the LLM to pick up to *max_subtopics* relevant subtopics for a given topic.

    Parameters
    ----------
    question : str
        The user's question.
    topic_nid : str
        Node‑id of the topic in the graph.
    graph : nx.Graph
    client : OpenAI
    model : str
    max_subtopics : int, default 10

    Returns
    -------
    List[str]
        List of chosen subtopic *labels* preserving the order of the supplied list.
    """
    if graph.nodes[topic_nid].get("type") != "topic":
        raise ValueError(f"Node {topic_nid} is not of type 'topic'.")

    sub_nodes = extract_subtopics_for_topic(graph, topic_nid)
    if not sub_nodes:
        return []

    sub_labels = [lbl for _nid, lbl in sub_nodes]

    prompt = (
        SUBTOPIC_CHOICE_PROMPT
        .replace("{TOPIC_LABEL}", graph.nodes[topic_nid].get("label", ""))
        .replace("{SUBTOPIC_LIST}", json.dumps(sub_labels, ensure_ascii=False))
        .replace("{question}", question)
    )

    print(len(sub_labels))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )


    content = response.choices[0].message.content

    # ----------------- Parse -----------------
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from LLM:\n{content}\nError: {exc}")

    chosen = data.get("subtopics")
    if not isinstance(chosen, list) or not (1 <= len(chosen) <= max_subtopics):
        raise ValueError(f"Invalid 'subtopics' list size: {chosen}")

    invalid = [s for s in chosen if s not in sub_labels]
    if invalid:
        raise ValueError(f"LLM chose subtopics not in allowed list: {invalid}")

    ordered = [lbl for lbl in sub_labels if lbl in chosen]
    return ordered

# ---------------------------------------------------------------------------
# Stand‑alone quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    gexf_path = "hotpotQA/graph_v1.gexf"
    if not os.path.exists(gexf_path):
        raise SystemExit("Place a graph.gexf or set GEXF_PATH env var.")

    G = nx.read_gexf(gexf_path)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Example pick the first topic node
    # topic_id = next(n for n, d in G.nodes(data=True) if d.get("type") == "topic")
    topic_id = "topic_entertainment"

    print("Topic:", G.nodes[topic_id]["label"])
    subs = choose_subtopics_for_topic(
        question="Which American comedian born on March 21, 1962, appeared in the movie \"Sleepless in Seattle?\"",
        topic_nid=topic_id,
        graph=G,
        client=client,
    )
    print("Chosen subtopics:", subs)
