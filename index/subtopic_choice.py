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
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompt.subtopic_choice import SUBTOPIC_CHOICE_PROMPT

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configurable selection range
# ---------------------------------------------------------------------------
SUBTOPIC_CHOICE_MIN = 10
SUBTOPIC_CHOICE_MAX = 25



DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 5        # ← configurable global
RETRY_BACKOFF = 0.2      # seconds between retries

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
    max_subtopics: int = SUBTOPIC_CHOICE_MAX,
    min_subtopics: int = SUBTOPIC_CHOICE_MIN,
) -> dict[str, List[str]]:
    """Return up to ``max_subtopics`` relevant subtopic **labels** for *topic_nid*.

    LLM의 응답 중 유효한 서브토픽만 필터링해서 반환함. 응답이 JSON 파싱 실패 시에만 재시도.
    """

    if graph.nodes[topic_nid].get("type") != "topic":
        raise ValueError(f"Node {topic_nid} is not of type 'topic'.")

    # 1) 후보 서브토픽 모으기
    sub_nodes = extract_subtopics_for_topic(graph, topic_nid)
    if not sub_nodes:
        return []
    sub_labels = [lbl for _nid, lbl in sub_nodes]
    min_subtopics = min(min_subtopics, len(sub_labels))
    # 2) 프롬프트 구성
    prompt = (
        SUBTOPIC_CHOICE_PROMPT
        .replace("{{TOPIC_LABEL}}", graph.nodes[topic_nid].get("label", ""))
        .replace("{{SUBTOPIC_LIST}}", json.dumps(sub_labels, ensure_ascii=False))
        .replace("{question}", question)
        .replace("{max_subtopics}", str(max_subtopics))
        .replace("{min_subtopics}", str(min_subtopics))
    )

    # 3) 최대 MAX_RETRIES까지 JSON 파싱 실패 시 재시도
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                frequency_penalty=1.2,
            )
            # print(prompt)
            content = response.choices[0].message.content

            data = json.loads(content)
            chosen = data.get("subtopics", [])

            if not isinstance(chosen, list):
                print(f"⚠️ Attempt {attempt}: 'subtopics' is not a list. Returning [].")
                continue

            # 허용된 서브토픽만 필터링
            valid_chosen = [lbl for lbl in sub_labels if lbl in chosen]

            if not valid_chosen:
                print("⚠️ raw LLM response:", content) 
                print(f"⚠️ Attempt {attempt}: No valid subtopics in LLM response.")
            return valid_chosen[:max_subtopics]

        except (json.JSONDecodeError, KeyError) as exc:
            print("⚠️ raw LLM response:", content) 
            print(f"⚠️ Attempt {attempt}: JSON parse/format error → {exc}. Retrying…")
        except Exception as exc:
            print(f"⚠️ Attempt {attempt}: OpenAI error → {exc}. Retrying…")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)

    print("🚫 All retries exhausted – returning empty subtopic list.")
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
    topic_id = ['topic_culture', 'topic_art', 'topic_music', 'topic_performance', 'topic_theater']

    for topic in topic_id :
        subs = choose_subtopics_for_topic(
            question="How many operas are among the artist who composed The Prelude for Clarinet in B-flat major best known works?",
            topic_nid=topic,
            graph=G,
            client=client,
        )
        print("Chosen subtopics:", subs)

