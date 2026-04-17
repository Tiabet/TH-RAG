"""Graph construction for TH-RAG.

This step chunks a dataset's contexts.txt file, stores the chunk text in a KV store,
and extracts topic-aware triples for each chunk with an OpenAI model.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import tiktoken
from openai import OpenAI
from tqdm import tqdm

from config import THRAGConfig, get_config
from prompt.extract_graph import EXTRACTION_PROMPT


def chunk_text(text: str, max_tokens: int, overlap: int, model_name: str) -> list[str]:
    """Split text into overlapping token windows."""

    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(0, end - overlap)

    return chunks


def parse_triples_response(response_text: str) -> list[dict[str, Any]]:
    """Parse a model response into a list of triple dictionaries."""

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    payload = json.loads(cleaned)
    if isinstance(payload, dict):
        triples = payload.get("triples", [])
    elif isinstance(payload, list):
        triples = payload
    else:
        raise ValueError("The extraction response must be a JSON list or an object with a 'triples' field.")

    validated: list[dict[str, Any]] = []
    for item in triples:
        if not isinstance(item, dict):
            continue
        triple = item.get("triple")
        subject = item.get("subject")
        object_ = item.get("object")
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        if not isinstance(subject, dict) or not isinstance(object_, dict):
            continue
        validated.append(item)
    return validated



def build_kv_store(chunks: list[str]) -> dict[str, dict[str, str]]:
    """Create the chunk lookup structure used during answer generation."""

    return {
        f"chunk-{index:05d}": {"content": chunk}
        for index, chunk in enumerate(chunks)
    }



def load_existing_blocks(output_path: Path) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}

    with output_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            return {}

    existing: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        for block in payload:
            if isinstance(block, dict) and "chunk_id" in block:
                existing[str(block["chunk_id"])] = block
    return existing



def call_model(client: OpenAI, model_name: str, chunk_text_value: str, chunk_id: str) -> dict[str, Any]:
    prompt = EXTRACTION_PROMPT.replace("{{document}}", chunk_text_value.strip())
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You extract factual triples from text and return valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=get_config().max_tokens_response,
        response_format={"type": "text"},
    )
    content = response.choices[0].message.content or "[]"
    triples = parse_triples_response(content)
    return {
        "chunk_id": chunk_id,
        "content": chunk_text_value,
        "triples": triples,
    }



def save_blocks(output_path: Path, blocks: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(blocks, handle, indent=2, ensure_ascii=False)



def save_kv_store(output_path: Path, kv_store: dict[str, dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(kv_store, handle, indent=2, ensure_ascii=False)



def run_graph_construction(config: THRAGConfig, force_rebuild: bool = False) -> str:
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be configured before running graph construction.")

    input_path = config.get_contexts_file()
    output_path = config.get_graph_json_file()
    kv_store_path = config.get_kv_store_file()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    full_text = input_path.read_text(encoding="utf-8")
    chunks = chunk_text(full_text, config.max_tokens, config.overlap, config.default_model)
    kv_store = build_kv_store(chunks)
    save_kv_store(kv_store_path, kv_store)

    chunk_ids = list(kv_store.keys())
    existing_blocks = {} if force_rebuild else load_existing_blocks(output_path)

    ordered_blocks: list[dict[str, Any] | None] = [None] * len(chunks)
    pending_indices: list[int] = []

    for index, chunk_id in enumerate(chunk_ids):
        existing_block = existing_blocks.get(chunk_id)
        if existing_block and isinstance(existing_block.get("triples"), list):
            ordered_blocks[index] = existing_block
        else:
            pending_indices.append(index)

    if pending_indices:
        client = OpenAI(api_key=config.openai_api_key)
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(call_model, client, config.default_model, chunks[index], chunk_ids[index]): index
                for index in pending_indices
            }
            for completed_count, future in enumerate(
                tqdm(as_completed(futures), total=len(futures), desc="Extracting triples"),
                start=1,
            ):
                index = futures[future]
                try:
                    ordered_blocks[index] = future.result()
                except Exception as exc:
                    ordered_blocks[index] = {
                        "chunk_id": chunk_ids[index],
                        "content": chunks[index],
                        "triples": [],
                        "error": str(exc),
                    }

                if completed_count % 10 == 0 or completed_count == len(futures):
                    save_blocks(output_path, [block for block in ordered_blocks if block is not None])
    else:
        save_blocks(output_path, [block for block in ordered_blocks if block is not None])

    final_blocks = [block for block in ordered_blocks if block is not None]
    save_blocks(output_path, final_blocks)
    config.mark_step_completed(
        "graph_construction",
        input_file=str(input_path),
        output_file=str(output_path),
        kv_store_file=str(kv_store_path),
        chunks=len(chunks),
    )
    return str(output_path)



def main(dataset_name: str, force_rebuild: bool = False) -> str:
    """CLI-compatible entry point for graph construction."""

    config = get_config(dataset_name)
    print(f"Building graph inputs for dataset: {dataset_name}")
    print(f"Input: {config.get_contexts_file()}")
    print(f"Graph JSON: {config.get_graph_json_file()}")
    print(f"KV store: {config.get_kv_store_file()}")
    return run_graph_construction(config, force_rebuild=force_rebuild)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graph-construction artifacts for a dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset>/")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute chunk extraction even if output artifacts already exist.",
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset, force_rebuild=args.force)

