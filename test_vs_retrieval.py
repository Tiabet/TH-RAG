import json
import math
from pathlib import Path
from collections import defaultdict

# ── 파일 경로 ─────────────────────────────────────────────────────────────
RETRIEVED_PATH = Path("Result/LightRAG/chunk_query_chunks_multihoprag.jsonl")
# RETRIEVED_PATH = Path("Result/HypergraphRAG/chunk_query_chunks_multihop.jsonl")
# GOLD_PATH      = Path("MultihopRAG/multihop_gold_chunk_ids.json")
GOLD_PATH = Path("MultihopRAG/multihop_gold_chunk_ids.json")

# ── 로더 ──────────────────────────────────────────────────────────────────
def load_retrieved(path: Path) -> dict[str, list[str]]:
    retrieved = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["query"]
            if "chunk_id" in obj:
                if obj["chunk_id"] not in retrieved[q]:
                    retrieved[q].append(obj["chunk_id"])
            else:
                retrieved[q].extend(obj.get("retrieved_chunk_ids", []))
    return retrieved

def load_gold(path: Path) -> dict[str, set[str]]:
    with path.open(encoding="utf-8") as f:
        gold_items = json.load(f)
    return {item["query"]: set(item["gold_chunk_ids"]) for item in gold_items}

# ── 메트릭 계산 ──────────────────────────────────────────────────────────
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

# ── 평가 ────────────────────────────────────────────────────────────────
def evaluate(retrieved: dict[str, list[str]], gold: dict[str, set[str]]) -> dict:
    results = []
    sum_tp = sum_fp = sum_fn = 0

    # 누적값
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

        # @3
        p_at_3 = precision_at_k(ret_list, gold_set, k=3)
        r_at_3 = recall_at_k(ret_list, gold_set, k=3)
        ndcg_3 = ndcg_at_k(ret_list, gold_set, k=3)

        # @5
        p_at_5 = precision_at_k(ret_list, gold_set, k=5)
        r_at_5 = recall_at_k(ret_list, gold_set, k=5)
        ndcg_5 = ndcg_at_k(ret_list, gold_set, k=5)

        # @10
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

# ── 메인 ─────────────────────────────────────────────────────────────────
def main() -> None:
    retrieved = load_retrieved(RETRIEVED_PATH)
    gold = load_gold(GOLD_PATH)
    eval_result = evaluate(retrieved, gold)
    s = eval_result["summary"]

    print(
        f"✅ 평가 완료 ({s['n_queries']}개 쿼리)\n"
        f"• Micro  P/R/F1: {s['micro']['precision']:.3f} / "
        f"{s['micro']['recall']:.3f} / {s['micro']['f1']:.3f}\n"
        f"• Macro  P/R/F1: {s['macro']['precision']:.3f} / "
        f"{s['macro']['recall']:.3f} / {s['macro']['f1']:.3f}\n"
        f"• @3     P@3 / R@3 / nDCG@3: {s['@3']['precision@3']:.3f} / "
        f"{s['@3']['recall@3']:.3f} / {s['@3']['ndcg@3']:.3f}\n"
        f"• @5     P@5 / R@5 / nDCG@5: {s['@5']['precision@5']:.3f} / "
        f"{s['@5']['recall@5']:.3f} / {s['@5']['ndcg@5']:.3f}\n"
        f"• @10    P@10 / R@10 / nDCG@10: {s['@10']['precision@10']:.3f} / "
        f"{s['@10']['recall@10']:.3f} / {s['@10']['ndcg@10']:.3f}"
    )

if __name__ == "__main__":
    main()
