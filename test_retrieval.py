import json
from pathlib import Path
from collections import defaultdict

# ── 파일 경로 ─────────────────────────────────────────────────────────────
# ① 리트리버가 반환한 청크‑ID 목록(JSONL)
RETRIEVED_PATH = Path("Result/GraphRAGL/local_chunk_query_chunks_hotpot.jsonl")

# ② gold 청크‑ID 매핑(JSON, 위 단계에서 생성)
GOLD_PATH       = Path("Result/hotpot_gold_chunk_ids.json")

# ③ 결과 저장 위치(CSV나 JSON 등 필요 시 수정)
OUT_PATH        = Path("Result/GraphRAGL/retriever_eval.json")

# ── 로더 ──────────────────────────────────────────────────────────────────
def load_retrieved(path: Path) -> dict[str, set[str]]:
    """
    { query : {retrieved_chunk_ids} } 딕셔너리 반환.
    라인 구조가 ① {"query": "...", "chunk_id": "..."}
               또는 ② {"query": "...", "retrieved_chunk_ids": [...]} 둘 다 대응.
    """
    retrieved = defaultdict(set)

    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["query"]

            if "chunk_id" in obj:                           # 형태 ①
                retrieved[q].add(obj["chunk_id"])
            else:                                           # 형태 ②
                retrieved[q].update(obj.get("retrieved_chunk_ids", []))
    return retrieved


def load_gold(path: Path) -> dict[str, set[str]]:
    """{ query : {gold_chunk_ids} } 딕셔너리 반환"""
    with path.open(encoding="utf-8") as f:
        gold_items = json.load(f)
    return {item["query"]: set(item["gold_chunk_ids"]) for item in gold_items}


# ── 메트릭 계산 ──────────────────────────────────────────────────────────
def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate(retrieved: dict[str, set[str]], gold: dict[str, set[str]]) -> dict:
    results = []

    # 마이크로 합계용
    sum_tp = sum_fp = sum_fn = 0

    for q, gold_set in gold.items():
        ret_set = retrieved.get(q, set())

        tp = len(ret_set & gold_set)
        fp = len(ret_set - gold_set)
        fn = len(gold_set - ret_set)

        prec, rec, f1 = precision_recall_f1(tp, fp, fn)

        results.append({
            "query": q,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "retrieved": list(ret_set), "gold": list(gold_set)
        })

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    micro_p, micro_r, micro_f1 = precision_recall_f1(sum_tp, sum_fp, sum_fn)
    macro_p = sum(r["precision"] for r in results) / len(results)
    macro_r = sum(r["recall"]    for r in results) / len(results)
    macro_f1 = sum(r["f1"]       for r in results) / len(results)

    summary = {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "n_queries": len(results),
    }

    return {"per_query": results, "summary": summary}


# ── 메인 ─────────────────────────────────────────────────────────────────
def main() -> None:
    retrieved = load_retrieved(RETRIEVED_PATH)
    gold      = load_gold(GOLD_PATH)

    eval_result = evaluate(retrieved, gold)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    s = eval_result["summary"]
    print(
        f"✅ 평가 완료 ({s['n_queries']}개 쿼리)\n"
        f"• Micro  P/R/F1: {s['micro']['precision']:.3f} / "
        f"{s['micro']['recall']:.3f} / {s['micro']['f1']:.3f}\n"
        f"• Macro  P/R/F1: {s['macro']['precision']:.3f} / "
        f"{s['macro']['recall']:.3f} / {s['macro']['f1']:.3f}\n"
        f"→ 상세 결과는 {OUT_PATH} 에 저장했습니다."
    )


if __name__ == "__main__":
    main()
