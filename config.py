"""Configuration helpers for the TH-RAG research codebase."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class THRAGConfig:
    """Central configuration and path management for the repository."""

    def __init__(self, dataset_name: str | None = None) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.dataset_name = dataset_name

        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.temp_dir = self.project_root / "temp"

        self.index_results_dir = self.results_dir / "index"
        self.generated_results_dir = self.results_dir / "generated"
        self.evaluated_results_dir = self.results_dir / "evaluated"
        self.chunks_dir = self.results_dir / "chunks"

        self._load_environment()
        self._ensure_directories()

    def _load_environment(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.eval_model = os.getenv("EVAL_MODEL", "gpt-4o-mini")

        self.temperature = float(os.getenv("TEMPERATURE", "0.5"))
        self.max_tokens_response = int(os.getenv("MAX_TOKENS_RESPONSE", "2000"))
        self.answer_temperature = float(os.getenv("ANSWER_TEMPERATURE", "0.3"))
        self.answer_max_tokens = int(os.getenv("ANSWER_MAX_TOKENS", "1000"))
        self.eval_temperature = float(os.getenv("EVAL_TEMPERATURE", "0.1"))

        self.max_tokens = int(os.getenv("MAX_TOKENS", "3000"))
        self.overlap = int(os.getenv("OVERLAP", "300"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "10"))
        self.alt_max_tokens = int(os.getenv("ALT_MAX_TOKENS", "1200"))
        self.alt_overlap = int(os.getenv("ALT_OVERLAP", "100"))

        self.topic_choice_min = int(os.getenv("TOPIC_CHOICE_MIN", "5"))
        self.topic_choice_max = int(os.getenv("TOPIC_CHOICE_MAX", "10"))
        self.subtopic_choice_min = int(os.getenv("SUBTOPIC_CHOICE_MIN", "10"))
        self.subtopic_choice_max = int(os.getenv("SUBTOPIC_CHOICE_MAX", "25"))

        self.max_retries = int(os.getenv("MAX_RETRIES", "10"))
        self.retry_backoff = float(os.getenv("RETRY_BACKOFF", "0.2"))

        self.top_k1 = int(os.getenv("TOP_K1", "50"))
        self.top_k2 = int(os.getenv("TOP_K2", "10"))
        self.top_k1_long = int(os.getenv("TOP_K1_LONG", "25"))
        self.top_k2_long = int(os.getenv("TOP_K2_LONG", "5"))
        self.embedding_top_k = int(os.getenv("EMBEDDING_TOP_K", "5"))
        self.overretrieve_factor = int(os.getenv("OVERRETRIEVE_FACTOR", "5"))

        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))

        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "thrag.log")
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))

    def _ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.results_dir,
            self.temp_dir,
            self.index_results_dir,
            self.generated_results_dir,
            self.evaluated_results_dir,
            self.chunks_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _require_dataset_name(self, dataset_name: str | None = None) -> str:
        name = dataset_name or self.dataset_name
        if not name:
            raise ValueError("A dataset name is required for this operation.")
        return name

    def get_dataset_dir(self, dataset_name: str | None = None) -> Path:
        return self.data_dir / self._require_dataset_name(dataset_name)

    def get_input_file(self, dataset_name: str | None = None) -> Path:
        return self.get_dataset_dir(dataset_name) / "contexts.txt"

    def get_contexts_file(self, dataset_name: str | None = None) -> Path:
        return self.get_input_file(dataset_name)

    def get_qa_file(self, dataset_name: str | None = None) -> Path:
        return self.get_dataset_dir(dataset_name) / "qa.json"

    def get_questions_file(self, dataset_name: str | None = None) -> Path:
        return self.get_qa_file(dataset_name)

    def get_graph_json_file(self, dataset_name: str | None = None) -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.index_results_dir / f"{name}_graph.json"

    def get_graph_gexf_file(self, dataset_name: str | None = None) -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.index_results_dir / f"{name}_graph.gexf"

    def get_kv_store_file(self, dataset_name: str | None = None) -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.index_results_dir / f"{name}_kv_store.json"

    def get_edge_index_file(self, dataset_name: str | None = None) -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.index_results_dir / f"{name}_edge_index.faiss"

    def get_edge_payload_file(self, dataset_name: str | None = None) -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.index_results_dir / f"{name}_edge_payloads.npy"

    def get_answer_file(self, dataset_name: str | None = None, answer_type: str = "short") -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.generated_results_dir / f"{name}_answers_{answer_type}.json"

    def get_chunk_log_file(self, dataset_name: str | None = None, answer_type: str = "short") -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.chunks_dir / f"{name}_chunks_{answer_type}.jsonl"

    def get_evaluation_file(self, dataset_name: str | None = None, eval_method: str = "f1") -> Path:
        name = self._require_dataset_name(dataset_name)
        return self.evaluated_results_dir / f"{name}_eval_{eval_method}.json"

    def get_pipeline_state_file(self) -> Path:
        return self.temp_dir / "pipeline_state.json"

    def save_pipeline_state(self, state: dict[str, Any]) -> None:
        with self.get_pipeline_state_file().open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)

    def load_pipeline_state(self) -> dict[str, Any] | None:
        state_file = self.get_pipeline_state_file()
        if not state_file.exists():
            return None
        with state_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_dataset_state(self, dataset_name: str | None = None) -> dict[str, Any]:
        state = self.load_pipeline_state() or {}
        return state.get(self._require_dataset_name(dataset_name), {})

    def mark_step_completed(
        self,
        step_name: str,
        dataset_name: str | None = None,
        **metadata: Any,
    ) -> None:
        dataset = self._require_dataset_name(dataset_name)
        state = self.load_pipeline_state() or {}
        state.setdefault(dataset, {})
        state[dataset][step_name] = {"completed": True, **metadata}
        self.save_pipeline_state(state)

    def clear_pipeline_state(self) -> None:
        state_file = self.get_pipeline_state_file()
        if state_file.exists():
            state_file.unlink()

    def list_available_datasets(self) -> list[str]:
        if not self.data_dir.exists():
            return []
        datasets = [
            item.name
            for item in self.data_dir.iterdir()
            if item.is_dir() and (item / "contexts.txt").exists()
        ]
        return sorted(datasets)

    def list_indexed_datasets(self) -> list[str]:
        datasets: set[str] = set()
        if self.index_results_dir.exists():
            for item in self.index_results_dir.glob("*_graph.gexf"):
                datasets.add(item.stem.removesuffix("_graph"))
        return sorted(datasets)

    def list_generated_datasets(self) -> list[str]:
        datasets: set[str] = set()
        if self.generated_results_dir.exists():
            for item in self.generated_results_dir.glob("*_answers_*.json"):
                datasets.add(item.name.split("_answers_")[0])
        return sorted(datasets)

    def list_evaluated_datasets(self) -> list[str]:
        datasets: set[str] = set()
        if self.evaluated_results_dir.exists():
            for item in self.evaluated_results_dir.glob("*_eval_*.json"):
                datasets.add(item.name.split("_eval_")[0])
        return sorted(datasets)



_config = THRAGConfig()


def get_config(dataset_name: str | None = None) -> THRAGConfig:
    """Return a shared config instance or a dataset-specific config."""

    if dataset_name is None:
        return _config
    return THRAGConfig(dataset_name)
