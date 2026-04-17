"""Long-answer GraphRAG wrapper for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import get_config
from generate.graph_rag import GraphRAG as BaseGraphRAG
from prompt.answer import ANSWER_PROMPT


LONG_SYSTEM_PROMPT = (
    "You answer multi-hop questions using only the supplied graph evidence. "
    "Provide a detailed and well-structured explanation, but do not invent facts."
)


class GraphRAG(BaseGraphRAG):
    def __init__(self, dataset_name: str) -> None:
        config = get_config(dataset_name)
        super().__init__(
            dataset_name=dataset_name,
            answer_prompt=ANSWER_PROMPT,
            system_prompt=LONG_SYSTEM_PROMPT,
            default_top_k1=config.top_k1_long,
            default_top_k2=config.top_k2_long,
            temperature=config.answer_temperature,
            max_output_tokens=config.answer_max_tokens,
        )

