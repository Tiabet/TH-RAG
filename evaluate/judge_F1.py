"""F1-style evaluation for TH-RAG answer files."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from config import get_config


def normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = "".join(character for character in lowered if character not in string.punctuation)
    return " ".join(lowered.split())



def compute_metrics(prediction: str, gold: str) -> tuple[int, float, float, float]:
    prediction_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()

    if not prediction_tokens or not gold_tokens:
        exact_match = int(prediction_tokens == gold_tokens)
        return exact_match, 0.0, 0.0, 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0, 0.0, 0.0, 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1_score = 2 * precision * recall / (precision + recall)
    exact_match = int(prediction_tokens == gold_tokens)
    return exact_match, f1_score, precision, recall



def load_pairs(path: Path, answer_key: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(item["query"]): str(item[answer_key])
        for item in payload
        if isinstance(item, dict) and "query" in item and answer_key in item
    }



def main(
    dataset_name: str,
    pred_path_param: str | None = None,
    gold_path_param: str | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    config = get_config(dataset_name)
    pred_path = Path(pred_path_param) if pred_path_param else config.get_answer_file(answer_type="short")
    gold_path = Path(gold_path_param) if gold_path_param else config.get_questions_file()

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    predictions = load_pairs(pred_path, "result")
    gold_answers = load_pairs(gold_path, "answer")

    exact_match_sum = 0.0
    f1_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    contains_gold_count = 0
    missing = 0

    for query, gold_answer in gold_answers.items():
        prediction = predictions.get(query)
        if prediction is None or prediction.startswith("[Error]"):
            missing += 1
            continue

        exact_match, f1_score, precision, recall = compute_metrics(prediction, gold_answer)
        exact_match_sum += exact_match
        f1_sum += f1_score
        precision_sum += precision
        recall_sum += recall
        if normalize(gold_answer) in normalize(prediction):
            contains_gold_count += 1

    compared = len(gold_answers) - missing
    results = {
        "dataset": dataset_name,
        "compared": compared,
        "total": len(gold_answers),
        "missing": missing,
        "exact_match": exact_match_sum / compared if compared else 0.0,
        "f1_score": f1_sum / compared if compared else 0.0,
        "precision": precision_sum / compared if compared else 0.0,
        "recall": recall_sum / compared if compared else 0.0,
        "accuracy": contains_gold_count / compared if compared else 0.0,
    }

    output_path = config.get_evaluation_file(eval_method="f1")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    config.mark_step_completed(
        "evaluation_f1",
        pred_file=str(pred_path),
        gold_file=str(gold_path),
        eval_file=str(output_path),
        f1_score=results["f1_score"],
        accuracy=results["accuracy"],
        force_rebuild=force_rebuild,
    )

    print(f"F1 evaluation written to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate short-answer predictions with F1-style metrics.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset>/")
    parser.add_argument("--pred", help="Optional prediction file override")
    parser.add_argument("--gold", help="Optional gold-answer file override")
    parser.add_argument("--force", action="store_true", help="Overwrite existing evaluation outputs.")
    args = parser.parse_args()
    main(args.dataset, args.pred, args.gold, force_rebuild=args.force)

