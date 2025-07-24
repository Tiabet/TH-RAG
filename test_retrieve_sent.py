"""
debug_retrieval.py
------------------
hotpot_gold_chunk_ids.json 과 used_chunks_1.jsonl 를 기반으로
쿼리‑단위로 청크 매칭 문제를 진단한다.
"""

import json, statistics, csv
from pathlib import Path
from collections import defaultdict

# ───────── 경로 설정 ──────────
DATA_DIR   = Path("Result")          # 필요하면 수정
GOLD_PATH  = DATA_DIR / "hotpot_gold_chunk_ids.json"
RETR_PATH  = DATA_DIR / "Ours/used_chunks_1.jsonl"

# ── ① chunk‑id 정규화 ------------------------------------
def norm_chunk(cid: str) -> str:
    """
    • 대소문자 통일, 공백 제거
    • 'chunk-' ↔ 'chunk_' ↔ 'chunk' 접두사 차이를 막아 준다.
    • 필요하면 해시 앞 16자만 쓰도록 줄여도 OK
    """
    cid = str(cid).strip().lower()
    if cid.startswith(("chunk-", "chunk_")):
        cid = cid.split("-", 1)[-1] if "-" in cid else cid.split("_", 1)[-1]
    return cid

# ── ② gold 로드 ------------------------------------------
gold_raw = json.loads(GOLD_PATH.read_text("utf-8"))
gold_map = {
    item["query"].strip(): {norm_chunk(c) for c in item["gold_chunk_ids"]}
    for item in gold_raw
}

# ── ③ retrieved 로드 & 집합화 -----------------------------
retr_map = defaultdict(set)  # {query → {chunk_id, ...}}
with RETR_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        q   = rec["query"].strip()                # 공백 트림
        cid = norm_chunk(rec["sentence_chunk_id"])
        retr_map[q].add(cid)

# ── ④ 평가 대상 = 교집합 ----------------------------------
eval_queries = gold_map.keys() & retr_map.keys()
print(f"평가 대상 쿼리 (교집합) : {len(eval_queries)} 개")   # 기대값 200

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2*p*r / (p + r) if p + r else 0.0
    return p, r, f

micro_tp = micro_fp = micro_fn = 0
macro_p = macro_r = macro_f = 0.0

for q in eval_queries:
    gset = gold_map[q]
    rset = retr_map[q]

    tp = len(gset & rset)
    fp = len(rset - gset)
    fn = len(gset - rset)

    p, r, f = prf(tp, fp, fn)

    micro_tp += tp
    micro_fp += fp
    micro_fn += fn
    macro_p  += p
    macro_r  += r
    macro_f  += f

N = len(eval_queries)
macro_p /= N
macro_r /= N
macro_f /= N
micro_p, micro_r, micro_f = prf(micro_tp, micro_fp, micro_fn)

print("\n──────── CHUNK‑LEVEL METRICS (200 개) ────────")
print(f"Macro Precision : {macro_p:.4f}")
print(f"Macro Recall    : {macro_r:.4f}")
print(f"Macro F1        : {macro_f:.4f}\n")
print(f"Micro Precision : {micro_p:.4f}")
print(f"Micro Recall    : {micro_r:.4f}")
print(f"Micro F1        : {micro_f:.4f}")