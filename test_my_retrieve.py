"""
evaluate_ctx_fuzzy.py (v1.1-hc)
────────────────────────────────────────────
• 하드코딩된 경로 기반 평가 실행
• context_token 이 None 이거나 빈 문자열이어도 안전하게 동작
• context_token 에 [Chunk N] 태그가 없으면 빈 리스트 반환
• JSON‑list 형식의 retrieved + kv‑store fuzzy 매칭 + P/R/F1 계산
"""

import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import math

# ── 하드코딩된 경로 설정 ──────────────────────────
RETRIEVED_PATH = Path("Result/Ours/hotpot_result_1000.json")
GOLD_PATH = Path("hotpotQA/hotpot_gold_chunk_ids.json")
KV_PATH = Path("hotpotQA/kv_store_text_chunks.json")
OUT_PATH = Path("Result/retriever_eval_ctx.json")
PREFIX_LEN = 160

# ── 정규식 & 전처리 ───────────────────────────────
TAG = re.compile(r"\[Chunk\s+(\d+)]")

def normalize(text: str) -> str:
    """소문자 & 영숫자만 남김"""
    return re.sub(r"[^a-z0-9]", "", text.lower())

def slice_by_tag(ctx: str) -> List[Tuple[str, str]]:
    """[Chunk N] ... [Chunk M] ...  ⇒ [(N, 텍스트N), (M, 텍스트M)]"""
    if not ctx:
        return []
    parts = TAG.split(ctx)
    out: List[Tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        num = parts[i].strip()
        text = parts[i + 1].strip()
        out.append((num, text))
    return out

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


# ── 데이터 로드 ───────────────────────────────────
retrieved_items = json.loads(RETRIEVED_PATH.read_text("utf-8"))

gold_map: Dict[str, set] = {
    item["query"]: set(item["gold_chunk_ids"])
    for item in json.loads(GOLD_PATH.read_text("utf-8"))
}

kv_raw = json.loads(KV_PATH.read_text("utf-8"))
kv_prefix: Dict[str, str] = {
    normalize(obj["content"][:PREFIX_LEN]): cid for cid, obj in kv_raw.items()
}

# ── Precision@k 계산 ─────────────────────────────
def precision_at_k(pred_list: List[str], gold_set: set, k: int = 5) -> float:
    pred_top_k = pred_list[:k]
    hits = sum(1 for pid in pred_top_k if pid in gold_set)
    return hits / k

# ── nDCG@k 계산 ──────────────────────────────────
def ndcg_at_k(pred_list: List[str], gold_set: set, k: int = 5) -> float:
    def dcg(rel_scores: List[int]) -> float:
        return sum(score / math.log2(idx + 2) for idx, score in enumerate(rel_scores))
    
    rel = [1 if pid in gold_set else 0 for pid in pred_list[:k]]
    ideal_rel = sorted(rel, reverse=True)
    
    dcg_val = dcg(rel)
    idcg_val = dcg(ideal_rel)
    return dcg_val / idcg_val if idcg_val != 0 else 0.0

def recall_at_k(pred_list: list[str], gold_set: set[str], k: int = 5) -> float:
    top_k = pred_list[:k]
    hits = sum(1 for pid in top_k if pid in gold_set)
    return hits / len(gold_set) if gold_set else 0.0

# ── 헬퍼: 텍스트 → chunk‑id ──────────────────────
def match_chunk(text: str, cutoff: float = 0.85) -> str:
    key = normalize(text[:PREFIX_LEN])
    if key in kv_prefix:
        return kv_prefix[key]
    cands = difflib.get_close_matches(key, kv_prefix.keys(), n=1, cutoff=cutoff)
    return kv_prefix[cands[0]] if cands else f"UNMATCHED::{hash(key) & 0xFFFFFFFF:08x}"

# ── 평가 루프 ─────────────────────────────────────
results = []
sum_tp = sum_fp = sum_fn = 0

for item in retrieved_items:
    query = item["query"]
    ctx_raw = item.get("context_token", "")
    chunks = slice_by_tag(ctx_raw)

    ctx_ids = {match_chunk(text) for _, text in chunks}
    sup_set = {cid for cid in ctx_ids if not cid.startswith("UNMATCHED::")}
    gold_set = gold_map.get(query, set())

    tp = len(sup_set & gold_set)
    fp = len(sup_set - gold_set)
    fn = len(gold_set - sup_set)

    prec, rec, f1 = prf(tp, fp, fn)

    # 평가용 랭킹 목록: 순서대로 유지
    ranked_ids = [match_chunk(text) for _, text in chunks]
    ranked_ids = [cid for cid in ranked_ids if not cid.startswith("UNMATCHED::")]

    p_at_3 = precision_at_k(ranked_ids, gold_set, k=3)
    ndcg_3 = ndcg_at_k(ranked_ids, gold_set, k=3)
    recall_at_3 = recall_at_k(ranked_ids, gold_set, k=3)
    p_at_5 = precision_at_k(ranked_ids, gold_set, k=5)
    ndcg_5 = ndcg_at_k(ranked_ids, gold_set, k=5)
    recall_at_5 = recall_at_k(ranked_ids, gold_set, k=5)
    p_at_10 = precision_at_k(ranked_ids, gold_set, k=10)
    ndcg_10 = ndcg_at_k(ranked_ids, gold_set, k=10)
    recall_at_10 = recall_at_k(ranked_ids, gold_set, k=10)

    results.append({
        "query": query,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "gold": list(gold_set),
        "eval_set": list(sup_set),
        "unmatched": [c for c in ctx_ids if c.startswith("UNMATCHED::")],
        "precision@3": p_at_3,
        "ndcg@3": ndcg_3,
        "recall@3": recall_at_3,
        "precision@5": p_at_5,
        "ndcg@5": ndcg_5,
        "recall@5": recall_at_5,
        "precision@10": p_at_10,
        "ndcg@10": ndcg_10,
        "recall@10": recall_at_10,
    })

    sum_tp += tp
    sum_fp += fp
    sum_fn += fn

micro_p, micro_r, micro_f1 = prf(sum_tp, sum_fp, sum_fn)
macro_p = sum(r["precision"] for r in results) / len(results)
macro_r = sum(r["recall"] for r in results) / len(results)
macro_f1 = sum(r["f1"] for r in results) / len(results)
avg_p_at_3 = sum(r["precision@3"] for r in results) / len(results)
avg_ndcg_3 = sum(r["ndcg@3"] for r in results) / len(results)
avg_recall_at_3 = sum(r["recall@3"] for r in results) / len(results)
avg_p_at_5 = sum(r["precision@5"] for r in results) / len(results)
avg_ndcg_5 = sum(r["ndcg@5"] for r in results) / len(results)
avg_recall_at_5 = sum(r["recall@5"] for r in results) / len(results)
avg_p_at_10 = sum(r["precision@10"] for r in results) / len(results)
avg_ndcg_10 = sum(r["ndcg@10"] for r in results) / len(results)
avg_recall_at_10 = sum(r["recall@10"] for r in results) / len(results)

summary = {
    "n": len(results),
    "micro": {"p": micro_p, "r": micro_r, "f1": micro_f1},
    "macro": {"p": macro_p, "r": macro_r, "f1": macro_f1},
    "@3": {
        "precision@3": avg_p_at_3,
        "ndcg@3": avg_ndcg_3,
        "recall@3": avg_recall_at_3
    },
    "@5": {
        "precision@5": avg_p_at_5,
        "ndcg@5": avg_ndcg_5,
        "recall@5": avg_recall_at_5
    },
    "@10": {
        "precision@10": avg_p_at_10,
        "ndcg@10": avg_ndcg_10,
        "recall@10": avg_recall_at_10
}
}

OUT_PATH.write_text(
    json.dumps({"summary": summary, "per_query": results}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(
    f"Micro  P/R/F1 = {micro_p:.3f}/{micro_r:.3f}/{micro_f1:.3f}\n"
    f"Macro  P/R/F1 = {macro_p:.3f}/{macro_r:.3f}/{macro_f1:.3f}\n"
    f"@3     P@3/R@3/nDCG@3 = {avg_p_at_3:.3f}/{avg_recall_at_3:.3f}/{avg_ndcg_3:.3f}\n"
    f"@5     P@5/R@5/nDCG@5 = {avg_p_at_5:.3f}/{avg_recall_at_5:.3f}/{avg_ndcg_5:.3f}\n"
    f"@10    P@10/R@10/nDCG@10 = {avg_p_at_10:.3f}/{avg_recall_at_10:.3f}/{avg_ndcg_10:.3f}\n"
    f"→ 세부 결과는 {OUT_PATH}"
)