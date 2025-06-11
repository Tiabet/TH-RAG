import os
from typing import Dict, List, Tuple, Optional

import networkx as nx

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - optional
    cosine_similarity = None  # type: ignore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:  # pragma: no cover - optional
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

try:
    import openai
except ImportError:  # pragma: no cover - optional
    openai = None  # type: ignore


Edge = Tuple[str, str, str, str, str]  # id, source, target, label, sentence


class EdgeEmbedder:
    """Embed edge sentences and provide similarity search."""

    def __init__(
        self,
        gexf_path: str,
        embedding_model: str = "text-embedding-3-small",
        use_qdrant: bool = True,
    ) -> None:
        self.graph = nx.read_gexf(gexf_path)
        self.embedding_model = embedding_model
        self.edges: List[Edge] = []
        self.embeddings: Dict[str, List[float]] = {}
        self.openai_client = None
        self.use_qdrant = use_qdrant and QdrantClient is not None
        self.qdrant_client = None
        self.collection_name = "edge_sentences"

        if openai and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
            self.openai_client = openai.OpenAI()
        else:
            print(
                "OPENAI_API_KEY not found or openai package missing. Falling back to simple embeddings."
            )

        for src, dst, data in self.graph.edges(data=True):
            sentence = data.get("sentence")
            label = data.get("label") or data.get("relation_type", "")
            if not sentence:
                continue
            eid = f"{src}-{dst}-{label}"
            self.edges.append((eid, src, dst, label, sentence))

    def _embed_text(self, text: str) -> List[float]:
        if self.openai_client:
            response = self.openai_client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            return response.data[0].embedding
        # simple fallback embedding using word hashes
        words = text.lower().split()
        return [hash(w) % 1000 / 1000 for w in words]

    def embed_edges(self) -> None:
        for eid, _, _, _, sentence in self.edges:
            if eid not in self.embeddings:
                self.embeddings[eid] = self._embed_text(sentence)

        if self.use_qdrant:
            if not self.embeddings:
                return
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
                qmodels.PointStruct(id=i, vector=emb, payload={"edge_id": eid})
                for i, (eid, emb) in enumerate(self.embeddings.items())
            ]
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )

    def search(
        self, query: str, top_k: int = 5, allowed_ids: Optional[List[str]] = None
    ) -> List[Edge]:
        if not self.embeddings:
            self.embed_edges()
        query_emb = self._embed_text(query)

        if allowed_ids is not None:
            edges = [e for e in self.edges if e[0] in allowed_ids]
        else:
            edges = self.edges

        scores: List[Tuple[float, Edge]] = []
        for eid, src, dst, label, sentence in edges:
            emb = self.embeddings[eid]
            if self.openai_client and cosine_similarity is not None:
                score = cosine_similarity([query_emb], [emb])[0][0]
            else:
                q_set = set(query.lower().split())
                s_set = set(sentence.lower().split())
                inter = q_set & s_set
                union = q_set | s_set
                score = len(inter) / len(union) if union else 0.0
            scores.append((score, (eid, src, dst, label, sentence)))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [edge for _, edge in scores[:top_k]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed edge sentences")
    parser.add_argument("gexf", help="path to graph file")
    parser.add_argument("--top-k", type=int, default=5, help="demo search results")
    parser.add_argument("--query", default="example", help="query for demo search")
    args = parser.parse_args()

    embedder = EdgeEmbedder(args.gexf)
    embedder.embed_edges()
    results = embedder.search(args.query, top_k=args.top_k)
    for eid, src, dst, label, sentence in results:
        print(f"{src} --{label}--> {dst} | {sentence}")
