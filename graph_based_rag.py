import os
from typing import Dict, List

import networkx as nx

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

try:
    import openai
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    openai = None  # type: ignore


class GraphRAG:
    def __init__(
        self,
        gexf_path: str,
        embedding_model: str = "text-embedding-3-small",
        use_qdrant: bool = True,
    ) -> None:
        self.gexf_path = gexf_path
        self.graph = nx.read_gexf(gexf_path)
        self.node_labels: Dict[str, str] = {
            n: data.get("label", n) for n, data in self.graph.nodes(data=True)
        }
        self.embedding_model = embedding_model
        self.embeddings: Dict[str, List[float]] = {}
        self.openai_client = None
        self.use_qdrant = use_qdrant and QdrantClient is not None
        self.qdrant_client = None
        self.collection_name = "graph_nodes"

        if openai and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self.openai_client = openai.OpenAI()
        else:
            print(
                "OPENAI_API_KEY not found or openai package missing. Embeddings will use simple TF-IDF similarity."
            )

    def _embed_text(self, text: str) -> List[float]:
        if self.openai_client:
            response = self.openai_client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            return response.data[0].embedding
        else:
            # fallback: simple bag-of-words vector using words as set
            words = text.lower().split()
            return [hash(w) % 1000 / 1000 for w in words]

    def embed_nodes(self):
        for node, label in self.node_labels.items():
            if node not in self.embeddings:
                self.embeddings[node] = self._embed_text(str(label))

        if self.use_qdrant:
            if self.qdrant_client is None:
                first_emb = next(iter(self.embeddings.values()))
                self.qdrant_client = QdrantClient(":memory:")
                self.qdrant_client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=len(first_emb), distance=qmodels.Distance.COSINE
                    ),
                )

            points = [
                qmodels.PointStruct(
                    id=i,
                    vector=emb,
                    payload={"node": node},
                )
                for i, (node, emb) in enumerate(self.embeddings.items())
            ]
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        if not self.embeddings:
            self.embed_nodes()
        query_emb = self._embed_text(query)

        if self.use_qdrant and self.qdrant_client is not None:
            hits = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_emb,
                limit=top_k,
            )
            return [h.payload["node"] for h in hits]

        scores = []
        for node, emb in self.embeddings.items():
            if self.openai_client:
                score = cosine_similarity([query_emb], [emb])[0][0]
            else:
                # simple Jaccard similarity between word sets
                q_set = set(query.lower().split())
                n_set = set(str(self.node_labels[node]).lower().split())
                intersection = q_set & n_set
                union = q_set | n_set
                score = len(intersection) / len(union) if union else 0.0
            scores.append((score, node))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [node for _, node in scores[:top_k]]

    def build_context(self, nodes: List[str]) -> str:
        lines = []
        for node in nodes:
            label = self.node_labels.get(node, node)
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                if edge_data is None:
                    continue

                if self.graph.is_multigraph():
                    edge_values = edge_data.values()
                else:
                    edge_values = [edge_data]

                for edge in edge_values:
                    relation = edge.get("label") or edge.get("relation_type", "")
                    sentence = edge.get("sentence")
                    n_label = self.node_labels.get(neighbor, neighbor)
                    line = f"{label} --{relation}--> {n_label}"
                    if sentence:
                        line += f" | {sentence}"
                    lines.append(line)
        return "\n".join(lines)

    def answer(self, question: str, top_k: int = 5) -> str:
        nodes = self.retrieve(question, top_k=top_k)
        context = self.build_context(nodes)
        if self.openai_client:
            prompt = (
                "Use the following knowledge graph triples to answer the question. "
                "If the information is insufficient, say so.\n"
                f"Context:\n{context}\nQuestion: {question}"
            )
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        else:
            return f"Context:\n{context}\n---\nAnswer generation requires OpenAI API."


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Graph-based RAG using graph_v6.gexf")
    parser.add_argument("question", help="question to answer")
    parser.add_argument(
        "--top-k", type=int, default=5, help="number of nodes to retrieve"
    )
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="disable qdrant vector search",
    )
    args = parser.parse_args()

    rag = GraphRAG("graph_v6.gexf", use_qdrant=not args.no_qdrant)
    rag.embed_nodes()
    answer = rag.answer(args.question, top_k=args.top_k)
    print(answer)


if __name__ == "__main__":
    main()
