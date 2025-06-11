import json
import os
from typing import Dict, List, Set

import networkx as nx

from edge_embedding import EdgeEmbedder, Edge
from prompt.topic_prompt import TOPIC_PROMPT

try:
    import openai
except ImportError:  # pragma: no cover - optional
    openai = None  # type: ignore


class TopicEdgeRAG:
    def __init__(
        self,
        gexf_path: str = "graph_v6.gexf",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.graph = nx.read_gexf(gexf_path)
        self.node_labels: Dict[str, str] = {
            n: data.get("label", n) for n, data in self.graph.nodes(data=True)
        }
        self.embedder = EdgeEmbedder(gexf_path, embedding_model)
        self.embedder.embed_edges()
        self.openai_client = None
        if openai and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self.openai_client = openai.OpenAI()
        else:
            print(
                "OPENAI_API_KEY not set or openai missing; topic extraction will fail."
            )

    def _extract_topics(self, question: str) -> Dict[str, List[str]]:
        if not self.openai_client:
            return {}
        prompt = TOPIC_PROMPT.replace("{{question}}", question)
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        try:
            data = json.loads(response.choices[0].message.content.strip())
            result = {item["topic"]: item["subtopics"] for item in data["topics"]}
            return result
        except Exception:
            return {}

    def _match_nodes(self, topics: Dict[str, List[str]]) -> Set[str]:
        matched: Set[str] = set()
        for label in self.node_labels:
            text = self.node_labels[label]
            for t, subs in topics.items():
                if text == t or text in subs:
                    matched.add(label)
        return matched

    def retrieve_edges(self, question: str, top_k: int = 50) -> List[Edge]:
        topics = self._extract_topics(question)
        candidates = self._match_nodes(topics)
        allowed_ids: List[str] = []
        for src, dst, key, data in self.graph.edges(keys=True, data=True):
            if src in candidates or dst in candidates:
                relation = data.get("label") or data.get("relation_type", "")
                if relation in {"has_subtopic", "has_topic"}:
                    continue
                sentence = data.get("sentence")
                if sentence:
                    eid = f"{src}-{dst}-{relation}"
                    allowed_ids.append(eid)
        edges = self.embedder.search(question, top_k=top_k, allowed_ids=allowed_ids)
        return edges

    def build_context(self, edges: List[Edge]) -> str:
        lines = [
            f"{src} --{label}--> {dst} | {sentence}"
            for _, src, dst, label, sentence in edges
        ]
        return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Edge-based retrieval with topics")
    parser.add_argument("question", help="input question")
    parser.add_argument(
        "--top-k", type=int, default=50, help="number of edges to return"
    )
    args = parser.parse_args()

    rag = TopicEdgeRAG()
    edges = rag.retrieve_edges(args.question, top_k=args.top_k)
    print(rag.build_context(edges))


if __name__ == "__main__":
    main()
