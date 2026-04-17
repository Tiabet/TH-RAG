"""Convenience wrapper for the graph-building portion of the TH-RAG pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse

from config import get_config
from index.edge_embedding import build_index_for_dataset
from index.graph_construction import main as run_graph_construction
from index.json_to_gexf import convert_json_to_gexf


def main(
    dataset_name: str,
    skip_extraction: bool = False,
    skip_gexf: bool = False,
    skip_index: bool = False,
    force_rebuild: bool = False,
) -> None:
    config = get_config(dataset_name)

    if not skip_extraction:
        run_graph_construction(dataset_name=dataset_name, force_rebuild=force_rebuild)

    if not skip_gexf:
        graph_json_path = config.get_graph_json_file()
        if not graph_json_path.exists():
            raise FileNotFoundError(
                f"Graph JSON file not found. Run extraction first: {graph_json_path}"
            )
        convert_json_to_gexf(str(graph_json_path), str(config.get_graph_gexf_file()))
        config.mark_step_completed("json_to_gexf", output_file=str(config.get_graph_gexf_file()))

    if not skip_index:
        build_index_for_dataset(dataset_name=dataset_name, rebuild=force_rebuild)

    print(f"Graph-building pipeline completed for dataset: {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph extraction, GEXF conversion, and FAISS indexing.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset>/")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip triple extraction.")
    parser.add_argument("--skip-gexf", action="store_true", help="Skip GEXF conversion.")
    parser.add_argument("--skip-index", action="store_true", help="Skip FAISS index creation.")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if they exist.")
    args = parser.parse_args()
    main(
        dataset_name=args.dataset,
        skip_extraction=args.skip_extraction,
        skip_gexf=args.skip_gexf,
        skip_index=args.skip_index,
        force_rebuild=args.force,
    )

