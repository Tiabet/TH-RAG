"""Unified pipeline runner for the TH-RAG research codebase."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from config import THRAGConfig, get_config

PROJECT_ROOT = Path(__file__).resolve().parent

STEP_SEQUENCE = [
    "graph_construction",
    "json_to_gexf",
    "edge_embedding",
    "answer_generation_short",
    "answer_generation_long",
    "evaluation_f1",
]

STEP_GROUPS = {
    "graph_build": ["graph_construction", "json_to_gexf", "edge_embedding"],
    "answer_generation": ["answer_generation_short", "answer_generation_long"],
    "evaluation": ["evaluation_f1"],
}


def resolve_steps(requested_steps: list[str] | None) -> list[str]:
    """Expand group aliases and preserve the canonical execution order."""

    if not requested_steps:
        return STEP_SEQUENCE.copy()

    selected: list[str] = []
    for step in requested_steps:
        for candidate in STEP_GROUPS.get(step, [step]):
            if candidate not in STEP_SEQUENCE:
                raise ValueError(f"Unknown pipeline step: {step}")
            if candidate not in selected:
                selected.append(candidate)

    return [step for step in STEP_SEQUENCE if step in selected]


def expected_outputs(config: THRAGConfig) -> dict[str, list[Path]]:
    return {
        "graph_construction": [config.get_graph_json_file(), config.get_kv_store_file()],
        "json_to_gexf": [config.get_graph_gexf_file()],
        "edge_embedding": [config.get_edge_index_file(), config.get_edge_payload_file()],
        "answer_generation_short": [
            config.get_answer_file(answer_type="short"),
            config.get_chunk_log_file(answer_type="short"),
        ],
        "answer_generation_long": [
            config.get_answer_file(answer_type="long"),
            config.get_chunk_log_file(answer_type="long"),
        ],
        "evaluation_f1": [config.get_evaluation_file(eval_method="f1")],
    }


def step_is_complete(config: THRAGConfig, step_name: str) -> bool:
    outputs = expected_outputs(config)[step_name]
    if all(path.exists() for path in outputs):
        return True

    state = config.get_dataset_state()
    return bool(state.get(step_name, {}).get("completed")) and all(path.exists() for path in outputs)


def validate_inputs(config: THRAGConfig, steps: list[str]) -> None:
    graph_steps = {"graph_construction", "json_to_gexf", "edge_embedding"}
    answer_steps = {"answer_generation_short", "answer_generation_long", "evaluation_f1"}

    if graph_steps.intersection(steps) and not config.get_contexts_file().exists():
        raise FileNotFoundError(
            f"Dataset '{config.dataset_name}' is missing contexts.txt at {config.get_contexts_file()}"
        )

    if answer_steps.intersection(steps) and not config.get_questions_file().exists():
        raise FileNotFoundError(
            f"Dataset '{config.dataset_name}' is missing qa.json at {config.get_questions_file()}"
        )


def run_graph_construction(dataset_name: str, force_rebuild: bool) -> str:
    from index.graph_construction import main as graph_construction_main

    return graph_construction_main(dataset_name=dataset_name, force_rebuild=force_rebuild)


def run_json_to_gexf(dataset_name: str, force_rebuild: bool) -> str:
    from index.json_to_gexf import convert_json_to_gexf

    config = get_config(dataset_name)
    graph_json_path = config.get_graph_json_file()
    if not graph_json_path.exists():
        raise FileNotFoundError(
            f"Graph JSON file not found. Run graph_construction first: {graph_json_path}"
        )

    output_path = convert_json_to_gexf(str(graph_json_path), str(config.get_graph_gexf_file()))
    config.mark_step_completed(
        "json_to_gexf",
        output_file=output_path,
        force_rebuild=force_rebuild,
    )
    return output_path


def run_edge_embedding(dataset_name: str, force_rebuild: bool) -> str:
    from index.edge_embedding import build_index_for_dataset

    return build_index_for_dataset(dataset_name=dataset_name, rebuild=force_rebuild)


def run_answer_generation_short(dataset_name: str, force_rebuild: bool) -> str:
    from generate.answer_generation_short import main as generate_short_main

    return generate_short_main(dataset_name=dataset_name, force_rebuild=force_rebuild)


def run_answer_generation_long(dataset_name: str, force_rebuild: bool) -> str:
    from generate.answer_generation_long import main as generate_long_main

    return generate_long_main(dataset_name=dataset_name, force_rebuild=force_rebuild)


def run_evaluation_f1(dataset_name: str, force_rebuild: bool) -> dict:
    from evaluate.judge_F1 import main as evaluate_f1_main

    return evaluate_f1_main(dataset_name=dataset_name, force_rebuild=force_rebuild)


STEP_HANDLERS: dict[str, Callable[[str, bool], object]] = {
    "graph_construction": run_graph_construction,
    "json_to_gexf": run_json_to_gexf,
    "edge_embedding": run_edge_embedding,
    "answer_generation_short": run_answer_generation_short,
    "answer_generation_long": run_answer_generation_long,
    "evaluation_f1": run_evaluation_f1,
}


def run_pipeline(
    dataset_name: str,
    requested_steps: list[str] | None = None,
    force_rebuild: bool = False,
) -> dict[str, object]:
    config = get_config(dataset_name)
    steps = resolve_steps(requested_steps)
    validate_inputs(config, steps)

    print(f"Running TH-RAG pipeline for dataset: {dataset_name}")
    print(f"Steps: {', '.join(steps)}")
    print("=" * 72)

    results: dict[str, object] = {}

    for index, step_name in enumerate(steps, start=1):
        print(f"\nStep {index}: {step_name}")
        print("-" * 72)

        if not force_rebuild and step_is_complete(config, step_name):
            print("Skipped because the expected outputs already exist. Use --force to rebuild.")
            continue

        results[step_name] = STEP_HANDLERS[step_name](dataset_name, force_rebuild)
        print("Completed.")

    print("\n" + "=" * 72)
    print("Pipeline finished.")
    print_summary(config, steps)
    return results


def print_summary(config: THRAGConfig, steps: list[str]) -> None:
    artifacts = expected_outputs(config)
    print(f"Summary for dataset: {config.dataset_name}")
    for step_name in steps:
        outputs = [str(path) for path in artifacts[step_name] if path.exists()]
        status = "complete" if outputs else "missing"
        print(f"- {step_name}: {status}")
        for output in outputs:
            print(f"  {output}")



def list_datasets() -> None:
    config = get_config()
    available = set(config.list_available_datasets())
    indexed = set(config.list_indexed_datasets())
    generated = set(config.list_generated_datasets())
    evaluated = set(config.list_evaluated_datasets())

    if not available:
        print("No datasets found under data/. Add data/<dataset>/contexts.txt to begin.")
        return

    print("Available datasets")
    print("-" * 72)
    for dataset in sorted(available):
        status_parts = []
        if dataset in indexed:
            status_parts.append("indexed")
        if dataset in generated:
            status_parts.append("generated")
        if dataset in evaluated:
            status_parts.append("evaluated")
        suffix = f" ({', '.join(status_parts)})" if status_parts else ""
        print(f"- {dataset}{suffix}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TH-RAG pipeline.")
    parser.add_argument("--dataset", help="Dataset name under data/<dataset>/")
    parser.add_argument(
        "--steps",
        nargs="+",
        help=(
            "Optional subset of steps to run. Available steps: "
            f"{', '.join(STEP_SEQUENCE)}. "
            f"Group aliases: {', '.join(STEP_GROUPS)}."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if the target artifacts already exist.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets that contain contexts.txt under data/.",
    )
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_datasets:
        list_datasets()
        return

    if not args.dataset:
        parser.error("--dataset is required unless --list-datasets is used.")

    try:
        run_pipeline(
            dataset_name=args.dataset,
            requested_steps=args.steps,
            force_rebuild=args.force,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
