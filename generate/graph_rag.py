"""Shared GraphRAG implementation for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json
import time
from pathlib import Path
from typing import Any

import tiktoken
from openai import OpenAI

from config import get_config
from generate.Retriever import Retriever


class GraphRAG:
    """Graph-backed answer generator shared by the short and long answer modes."""

    def __init__(
        self,
        *,
        dataset_name: str,
        answer_prompt: str,
        system_prompt: str,
        default_top_k1: int,
        default_top_k2: int,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        self.config = get_config(dataset_name)
        self.dataset_name = dataset_name
        self.answer_prompt = answer_prompt
        self.system_prompt = system_prompt
        self.default_top_k1 = default_top_k1
        self.default_top_k2 = default_top_k2
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be configured before answer generation can run.")

        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.chunk_map = self._load_chunk_map(self.config.get_kv_store_file())
        self.retriever = Retriever(
            gexf_path=str(self.config.get_graph_gexf_file()),
            json_path=str(self.config.get_graph_json_file()),
            kv_json_path=str(self.config.get_kv_store_file()),
            index_path=str(self.config.get_edge_index_file()),
            payload_path=str(self.config.get_edge_payload_file()),
            embedding_model=self.config.embed_model,
            openai_api_key=self.config.openai_api_key,
            client=self.client,
            thread_workers=self.config.max_workers,
        )
        self.last_chunk_ids: list[str] = []
        self.all_sentence_chunk_ids: list[str] = []

    @staticmethod
    def _load_chunk_map(path: Path) -> dict[str, str]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {chunk_id: block["content"] for chunk_id, block in payload.items() if "content" in block}

    def compose_context(self, chunk_ids: list[str], edges_meta: list[dict[str, Any]]) -> str:
        sections: list[str] = []

        for index, chunk_id in enumerate(chunk_ids, start=1):
            chunk_text = self.chunk_map.get(chunk_id, "")
            if chunk_text:
                sections.append(f"[Chunk {index}]\n{chunk_text}")

        for index, edge in enumerate(edges_meta, start=1):
            source = edge.get("source", "unknown")
            label = edge.get("label", "related_to")
            target = edge.get("target", "unknown")
            sentence = edge.get("sentence", "")
            sections.append(f"[Evidence {index}]\n{source} --{label}--> {target}\n{sentence}")

        return "\n\n".join(sections)

    def _count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.config.chat_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))

    def answer(
        self,
        query: str,
        top_k1: int | None = None,
        top_k2: int | None = None,
    ) -> tuple[str, float, int]:
        top_k1 = top_k1 or self.default_top_k1
        top_k2 = top_k2 or self.default_top_k2

        started_at = time.time()
        retrieval = self.retriever.retrieve(query, top_k1=top_k1, top_k2=top_k2)
        elapsed = time.time() - started_at

        chunk_ids = retrieval.get("chunks", [])
        edges_meta = retrieval.get("edges", [])
        self.last_chunk_ids = chunk_ids

        sentence_chunk_ids: list[str] = []
        seen_chunk_ids: set[str] = set()
        for edge in edges_meta:
            chunk_id = edge.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                sentence_chunk_ids.append(chunk_id)
        self.all_sentence_chunk_ids = sentence_chunk_ids

        if not chunk_ids:
            return "I do not have enough retrieved evidence to answer this question.", elapsed, 0

        context = self.compose_context(chunk_ids, edges_meta)
        prompt = self.answer_prompt.replace("{{question}}", query).replace("{{context}}", context)
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        answer_text = (response.choices[0].message.content or "").strip()
        return answer_text, elapsed, self._count_tokens(context)

