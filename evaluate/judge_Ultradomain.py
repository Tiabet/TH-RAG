"""Pairwise answer comparison utility following the UltraDomain-style prompt."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from config import get_config
from prompt.evaluation import EVALUATION_PROMPT



def load_results(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        str(item["query"]): item
        for item in payload
        if isinstance(item, dict) and "query" in item and "result" in item
    }



def extract_json_from_response(response_text: str) -> str:
    if response_text.strip().startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response_text.strip()



def render_summary(results: list[dict[str, Any]], label_a: str, label_b: str) -> dict[str, dict[str, int]]:
    categories = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
    tallies = {label_a: Counter(), label_b: Counter()}

    for item in results:
        if "error" in item:
            continue
        for category in categories:
            winner = item.get(category, {}).get("Winner")
            if winner == "Answer 1":
                tallies[item["answer1_model"]][category] += 1
            elif winner == "Answer 2":
                tallies[item["answer2_model"]][category] += 1

    return {
        model_label: {category: tallies[model_label][category] for category in categories}
        for model_label in [label_a, label_b]
    }



def save_plot(summary: dict[str, dict[str, int]], label_a: str, label_b: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    categories = list(next(iter(summary.values())).keys())
    values_a = [summary[label_a][category] for category in categories]
    values_b = [summary[label_b][category] for category in categories]
    totals = [a + b for a, b in zip(values_a, values_b, strict=False)]

    pct_a = [a / total * 100 if total else 0 for a, total in zip(values_a, totals, strict=False)]
    pct_b = [b / total * 100 if total else 0 for b, total in zip(values_b, totals, strict=False)]

    x_positions = range(len(categories))
    bar_width = 0.35

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x_positions, pct_a, width=bar_width, label=label_a)
    axis.bar([x + bar_width for x in x_positions], pct_b, width=bar_width, label=label_b)

    axis.set_xlabel("Evaluation Criterion")
    axis.set_ylabel("Win Rate (%)")
    axis.set_title("Pairwise Evaluation Summary")
    axis.set_xticks([x + bar_width / 2 for x in x_positions])
    axis.set_xticklabels(categories)
    axis.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def main(
    answer_a_path: str,
    answer_b_path: str,
    output_path: str,
    label_a: str | None = None,
    label_b: str | None = None,
    plot_path: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    config = get_config()
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be configured before pairwise evaluation can run.")

    path_a = Path(answer_a_path)
    path_b = Path(answer_b_path)
    output = Path(output_path)
    label_a = label_a or path_a.stem
    label_b = label_b or path_b.stem

    predictions_a = load_results(path_a)
    predictions_b = load_results(path_b)
    shared_queries = sorted(set(predictions_a) & set(predictions_b))
    if not shared_queries:
        raise ValueError("The two answer files do not share any queries.")

    random.seed(seed)
    shuffled_indices = list(range(len(shared_queries)))
    random.shuffle(shuffled_indices)
    label_a_first = set(shuffled_indices[: len(shared_queries) // 2])

    client = OpenAI(api_key=config.openai_api_key)

    def judge_one(index: int, query: str) -> tuple[int, dict[str, Any]]:
        answer_a = predictions_a[query]["result"]
        answer_b = predictions_b[query]["result"]

        if index in label_a_first:
            answer1, answer2 = answer_a, answer_b
            answer1_label, answer2_label = label_a, label_b
        else:
            answer1, answer2 = answer_b, answer_a
            answer1_label, answer2_label = label_b, label_a

        prompt = EVALUATION_PROMPT.format(query=query, answer1=answer1, answer2=answer2)
        response = client.chat.completions.create(
            model=config.eval_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.eval_temperature,
        )
        raw_content = (response.choices[0].message.content or "").strip()

        try:
            parsed = json.loads(extract_json_from_response(raw_content))
            result = {
                "query": query,
                "answer1_model": answer1_label,
                "answer2_model": answer2_label,
                **parsed,
            }
        except Exception as exc:
            result = {
                "query": query,
                "answer1_model": answer1_label,
                "answer2_model": answer2_label,
                "error": str(exc),
                "raw_response": raw_content,
            }
        return index, result

    judged: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [executor.submit(judge_one, index, query) for index, query in enumerate(shared_queries)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running pairwise evaluation"):
            index, result = future.result()
            judged[index] = result

    ordered_results = [judged[index] for index in range(len(shared_queries))]
    summary = render_summary(ordered_results, label_a, label_b)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"results": ordered_results, "summary": summary}, handle, indent=2, ensure_ascii=False)

    if plot_path:
        save_plot(summary, label_a, label_b, Path(plot_path))

    print(f"Pairwise evaluation written to {output}")
    return {"results": ordered_results, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two answer files with the UltraDomain-style evaluation prompt.")
    parser.add_argument("--answer-a", required=True, help="Path to the first answer file")
    parser.add_argument("--answer-b", required=True, help="Path to the second answer file")
    parser.add_argument("--output", required=True, help="Path for the pairwise evaluation JSON output")
    parser.add_argument("--label-a", help="Optional display label for answer file A")
    parser.add_argument("--label-b", help="Optional display label for answer file B")
    parser.add_argument("--plot", help="Optional path for a PNG summary plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for answer-order balancing")
    args = parser.parse_args()
    main(
        answer_a_path=args.answer_a,
        answer_b_path=args.answer_b,
        output_path=args.output,
        label_a=args.label_a,
        label_b=args.label_b,
        plot_path=args.plot,
        seed=args.seed,
    )

