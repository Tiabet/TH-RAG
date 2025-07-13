import os
from typing import List, Dict

import networkx as nx  # noqa – kept for possible future use
import openai
from dotenv import load_dotenv

# Local imports – be sure these modules exist in your project
from Retriever_v3 import Retriever  # retrieval layer
from prompt.answer_infinitechoice import ANSWER_PROMPT  # <-- make sure this now contains the long template shown by the user

# ────────────────────────────────────────────────────────────────────────────────
# Environment / model configuration
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY environment variable required.")

EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL    = os.getenv("CHAT_MODEL", "gpt-4o-mini")
GRAPH_PATH    = os.getenv("GRAPH_PATH", "InfiniteQA/graph_v1.gexf")
INDEX_PATH    = os.getenv("INDEX_PATH", "InfiniteQA/edge_index_v1.faiss")
PAYLOAD_PATH  = os.getenv("PAYLOAD_PATH", "InfiniteQA/edge_payloads_v1.npy")


# ────────────────────────────────────────────────────────────────────────────────
# Helper class
# ────────────────────────────────────────────────────────────────────────────────
class GraphRAG:
    """Small wrapper that retrieves graph context then calls the chat model."""

    def __init__(
        self,
        gexf_path: str = GRAPH_PATH,
        index_path: str = INDEX_PATH,
        payload_path: str = PAYLOAD_PATH,
        embed_model: str = EMBED_MODEL,
        chat_model: str = CHAT_MODEL,
    ) -> None:
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Initialise retriever
        self.retriever = Retriever(
            gexf_path=gexf_path,
            embedding_model=embed_model,
            openai_api_key=OPENAI_API_KEY,
            index_path=index_path,
            payload_path=payload_path,
            client=self.client,
        )
        self.chat_model = chat_model

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _compose_context(entity_sentences: Dict[str, List[str]], faiss_results: List[Dict]) -> str:
        """Turns retriever output into a single context string."""
        parts: List[str] = []
        for ent, sents in entity_sentences.items():
            for s in sents:
                parts.append(f"[Entity: {ent}] {s}")
        for hit in faiss_results:
            parts.append(f"(Edge {hit['edge_id']} score={hit['score']:.3f}): {hit['sentence']}")
        return "\n".join(parts)

    # ---------------------------------------------------------------- main call
    def answer(self, question: str, options: List[str]) -> str:
        """Retrieve context and get a *single‑option* answer from the LLM.

        Parameters
        ----------
        question : str
            The user query.
        options : List[str]
            List of candidate answers; *exactly one* will be returned.
        """
        # ── 1. Retrieve graph context ────────────────────────────────────
        retrieved = self.retriever.retrieve(question, top_k=50)
        context_str = self._compose_context(
            retrieved.get("entity_sentences", {}),
            retrieved.get("faiss_results", []),
        )

        # ── 2. Build final prompt ────────────────────────────────────────
        opts_formatted = "\n".join(options)
        prompt = (
            ANSWER_PROMPT
            .replace("{{context}}", context_str)
            .replace("{{question}}", question)
            .replace("{{options}}", opts_formatted)
        )

        # ── 3. Call the model ────────────────────────────────────────────
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a graph‑aware assistant, capable of understanding complex relationships.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
