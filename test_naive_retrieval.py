from pathlib import Path
from collections import defaultdict
import json
import math

# Updated file paths
RETRIEVED_PATH = Path("Result/NaiveRAG/hotpot_result_with_uuid_chunks.json")
GOLD_PATH = Path("hotpotQA/hotpot_gold_chunk_ids_v2.json")

# Load retrieved results
def load_retrieved(path: Path) -> dict[str, list[str]]:
    retrieved = {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            query = item["query"]
            retrieved[query] = item.get("used_chunks", [])
    return retrieved

# Load gold standards
def load_gold(path: Path) -> dict[str, set[str]]:
    with path.open(encoding="utf-8") as f:
        gold_items = json.load(f)
    return {item["query"]: set(item["gold_chunk_ids"]) for item in gold_items}

# Metric calculations
def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def precision_at_k(pred_list: list[str], gold_set: set[str], k: int) -> float:
    top_k = pred_list[:k]
    hits = sum(1 for pid in top_k if pid in gold_set)
    return hits / k

def recall_at_k(pred_list: list[str], gold_set: set[str], k: int) -> float:
    top_k = pred_list[:k]
    hits = sum(1 for pid in top_k if pid in gold_set)
    return hits / len(gold_set) if gold_set else 0.0

def ndcg_at_k(pred_list: list[str], gold_set: set[str], k: int) -> float:
    def dcg(scores: list[int]) -> float:
        return sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))
    rel = [1 if pid in gold_set else 0 for pid in pred_list[:k]]
    ideal_rel = sorted(rel, reverse=True)
    dcg_val = dcg(rel)
    idcg_val = dcg(ideal_rel)
    return dcg_val / idcg_val if idcg_val != 0 else 0.0

# Evaluate results
def evaluate(retrieved: dict[str, list[str]], gold: dict[str, set[str]]) -> dict:
    results = []
    sum_tp = sum_fp = sum_fn = 0
    sum_p_at_3 = sum_r_at_3 = sum_ndcg_3 = 0
    sum_p_at_5 = sum_r_at_5 = sum_ndcg_5 = 0
    sum_p_at_10 = sum_r_at_10 = sum_ndcg_10 = 0

    for q, gold_set in gold.items():
        ret_list = retrieved.get(q, [])
        ret_set = set(ret_list)

        tp = len(ret_set & gold_set)
        fp = len(ret_set - gold_set)
        fn = len(gold_set - ret_set)

        prec, rec, f1 = precision_recall_f1(tp, fp, fn)

        p_at_3 = precision_at_k(ret_list, gold_set, k=3)
        r_at_3 = recall_at_k(ret_list, gold_set, k=3)
        ndcg_3 = ndcg_at_k(ret_list, gold_set, k=3)

        p_at_5 = precision_at_k(ret_list, gold_set, k=5)
        r_at_5 = recall_at_k(ret_list, gold_set, k=5)
        ndcg_5 = ndcg_at_k(ret_list, gold_set, k=5)

        p_at_10 = precision_at_k(ret_list, gold_set, k=10)
        r_at_10 = recall_at_k(ret_list, gold_set, k=10)
        ndcg_10 = ndcg_at_k(ret_list, gold_set, k=10)

        results.append({
            "query": q,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "precision@3": p_at_3,
            "recall@3": r_at_3,
            "ndcg@3": ndcg_3,
            "precision@5": p_at_5,
            "recall@5": r_at_5,
            "ndcg@5": ndcg_5,
            "precision@10": p_at_10,
            "recall@10": r_at_10,
            "ndcg@10": ndcg_10,
            "retrieved": ret_list,
            "gold": list(gold_set),
        })

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sum_p_at_3 += p_at_3
        sum_r_at_3 += r_at_3
        sum_ndcg_3 += ndcg_3
        sum_p_at_5 += p_at_5
        sum_r_at_5 += r_at_5
        sum_ndcg_5 += ndcg_5
        sum_p_at_10 += p_at_10
        sum_r_at_10 += r_at_10
        sum_ndcg_10 += ndcg_10

    n = len(results)
    micro_p, micro_r, micro_f1 = precision_recall_f1(sum_tp, sum_fp, sum_fn)
    macro_p = sum(r["precision"] for r in results) / n
    macro_r = sum(r["recall"] for r in results) / n
    macro_f1 = sum(r["f1"] for r in results) / n

    summary = {
        "n_queries": n,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "@3": {
            "precision@3": sum_p_at_3 / n,
            "recall@3": sum_r_at_3 / n,
            "ndcg@3": sum_ndcg_3 / n
        },
        "@5": {
            "precision@5": sum_p_at_5 / n,
            "recall@5": sum_r_at_5 / n,
            "ndcg@5": sum_ndcg_5 / n
        },
        "@10": {
            "precision@10": sum_p_at_10 / n,
            "recall@10": sum_r_at_10 / n,
            "ndcg@10": sum_ndcg_10 / n
        }
    }

    return {"per_query": results, "summary": summary}

# Main execution
retrieved = load_retrieved(RETRIEVED_PATH)
gold = load_gold(GOLD_PATH)
eval_result = evaluate(retrieved, gold)
summary = eval_result["summary"]

print(f"Evaluation Summary:\n"
      f"Total Queries: {summary['n_queries']}\n"
      f"Micro Precision: {summary['micro']['precision']:.3f}\n"
      f"Micro Recall: {summary['micro']['recall']:.3f}\n"
      f"Micro F1: {summary['micro']['f1']:.3f}\n"
      f"Macro Precision: {summary['macro']['precision']:.3f}\n"
      f"Macro Recall: {summary['macro']['recall']:.3f}\n"
      f"Macro F1: {summary['macro']['f1']:.3f}\n"
      f"@3 Precision: {summary['@3']['precision@3']:.3f}\n"
      f"@3 Recall: {summary['@3']['recall@3']:.3f}\n"
      f"@3 nDCG: {summary['@3']['ndcg@3']:.3f}\n"
      f"@5 Precision: {summary['@5']['precision@5']:.3f}\n"
      f"@5 Recall: {summary['@5']['recall@5']:.3f}\n"
      f"@5 nDCG: {summary['@5']['ndcg@5']:.3f}\n"
      f"@10 Precision: {summary['@10']['precision@10']:.3f}\n"
      f"@10 Recall: {summary['@10']['recall@10']:.3f}\n"
      f"@10 nDCG: {summary['@10']['ndcg@10']:.3f}\n")