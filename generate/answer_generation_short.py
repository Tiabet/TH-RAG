"""Batch short-answer generation for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import get_config
from generate.graph_based_rag_short import GraphRAG

_THREAD_STATE = threading.local()



def get_rag(dataset_name: str) -> GraphRAG:
    rag = getattr(_THREAD_STATE, "rag", None)
    if rag is None or getattr(rag, "dataset_name", None) != dataset_name:
        rag = GraphRAG(dataset_name=dataset_name)
        _THREAD_STATE.rag = rag
    return rag



def load_questions(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Question file must contain a JSON list: {input_path}")
    return payload



def main(dataset_name: str, force_rebuild: bool = False) -> str:
    config = get_config(dataset_name)
    input_path = config.get_questions_file()
    output_path = config.get_answer_file(answer_type="short")
    chunk_log_path = config.get_chunk_log_file(answer_type="short")
    temp_output_path = output_path.with_name(output_path.stem + "_temp.json")

    questions = load_questions(input_path)
    results: list[dict[str, Any] | None] = [None] * len(questions)

    def process(index: int, item: dict[str, Any]) -> tuple[int, dict[str, Any], list[dict[str, str]]]:
        query = str(item.get("query", "")).strip()
        rag = get_rag(dataset_name)
        try:
            answer_text, elapsed, context_tokens = rag.answer(query=query)
            chunk_log_entries = [
                {"query": query, "chunk_id": chunk_id}
                for chunk_id in rag.last_chunk_ids
            ]
            chunk_log_entries.extend(
                {"query": query, "sentence_chunk_id": chunk_id}
                for chunk_id in rag.all_sentence_chunk_ids
            )
            result = {
                "query": query,
                "result": answer_text,
                "meta": {
                    "total_spent": elapsed,
                    "context_tokens": context_tokens,
                },
            }
        except Exception as exc:
            result = {
                "query": query,
                "result": f"[Error] {exc}",
                "meta": {
                    "total_spent": 0.0,
                    "context_tokens": 0,
                },
            }
            chunk_log_entries = []
        return index, result, chunk_log_entries

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_log_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [executor.submit(process, index, item) for index, item in enumerate(questions)]
        for completed_count, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Generating short answers"),
            start=1,
        ):
            index, result, chunk_log_entries = future.result()
            results[index] = result
            log_lines.extend(json.dumps(entry, ensure_ascii=False) for entry in chunk_log_entries)

            if completed_count % 10 == 0 or completed_count == len(futures):
                with temp_output_path.open("w", encoding="utf-8") as handle:
                    json.dump(results, handle, indent=2, ensure_ascii=False)

    finalized_results = [item for item in results if item is not None]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(finalized_results, handle, indent=2, ensure_ascii=False)

    with chunk_log_path.open("w", encoding="utf-8") as handle:
        if log_lines:
            handle.write("\n".join(log_lines) + "\n")

    valid_answers = sum(
        1 for item in finalized_results if isinstance(item, dict) and not str(item.get("result", "")).startswith("[Error]")
    )
    config.mark_step_completed(
        "answer_generation_short",
        input_file=str(input_path),
        output_file=str(output_path),
        chunk_log_file=str(chunk_log_path),
        total_questions=len(finalized_results),
        valid_answers=valid_answers,
        force_rebuild=force_rebuild,
    )
    print(f"Short-answer results written to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate short answers for a TH-RAG dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset>/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()
    main(dataset_name=args.dataset, force_rebuild=args.force)

