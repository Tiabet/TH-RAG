import json, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# ── 경로 설정 ────────────────────────────────────────────────────────────
RESULTS_PATH = Path("Result/Ours/hotpot_result_sampled_v5.json")   # 리트리버 결과 (200개)
GOLD_PATH    = Path("Result/hotpot_gold_chunk_ids.json")           # gold 1 000개
STORE_PATH   = Path("hotpotQA/hotpot_kv_store_text_chunks.json")   # 전체 chunk store
OUT_PATH     = Path("Result/Ours/retriever_eval.json")             # 평가 지표 저장

# ── 1) chunk store 로드 ────────────────────────────────────────────────
print("📂 loading chunk store ...")
with STORE_PATH.open(encoding="utf-8") as f:
    STORE: Dict[str, str] = json.load(f)        # {chunk_id: {"content": "...", ...}, ...}
CONTENT_MAP = {cid: obj["content"] for cid, obj in STORE.items()}

# (성능 up) 선두 60자 → id 인덱스
HEAD2ID = defaultdict(list)
for cid, text in CONTENT_MAP.items():
    head = text.strip()[:60]          # 앞 60자
    HEAD2ID[head].append(cid)

# ── 2) 리트리버 결과 로드 & chunk‑id 매핑 ────────────────────────────────
CHUNK_SPLIT = re.compile(r"\[Chunk\s+\d+\]")

def text2id(chunk_text: str) -> str | None:
    """
    chunk 본문을 받아 store에서 같은 내용을 가진 id 반환.
    - 정확히 같지 않아도 충분히 긴 substring 매칭이면 OK
    """
    snippet = chunk_text.strip()[:60]
    # 1) 빠른 경로: 선두 60자 완전 일치
    for cid in HEAD2ID.get(snippet, []):
        return cid
    # 2) fallback: 전체 store scanning (느리지만 200×topk 정도면 감당 가능)
    for cid, text in CONTENT_MAP.items():
        if snippet and snippet in text:
            return cid
    return None

def parse_retrieved_chunks(context_token: str) -> Set[str]:
    """
    context_token 문자열 -> {chunk_hash_id ...}
    """
    if not isinstance(context_token, str):
        return set()

    parts = CHUNK_SPLIT.split(context_token)    # ['', '본문1', '본문2', ...]
    # parts[0]는 [Chunk] 이전 공백이므로 버리고 나머지 본문들만
    ids = set()
    for chunk_body in parts[1:]:
        cid = text2id(chunk_body)
        if cid:
            ids.add(cid)
    return ids

print("📂 loading retriever output ...")
with RESULTS_PATH.open(encoding="utf-8") as f:
    data = json.load(f)    # 200개 항목 배열

retrieved: Dict[str, Set[str]] = defaultdict(set)
for obj in data:
    q = obj["query"]
    ctx = obj.get("context_token", "")
    retrieved[q].update(parse_retrieved_chunks(ctx))

# ── 3) gold 로드 ────────────────────────────────────────────────────────
with GOLD_PATH.open(encoding="utf-8") as f:
    gold_items = json.load(f)
gold = {item["query"]: set(item["gold_chunk_ids"]) for item in gold_items}

# ── 4) 평가 ────────────────────────────────────────────────────────────
def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

eval_queries = [q for q in retrieved if q in gold]   # 200개 교집합
per_query, S_tp, S_fp, S_fn = [], 0, 0, 0

for q in eval_queries:
    g = gold[q]
    r = retrieved[q]
    tp = len(r & g)
    fp = len(r - g)
    fn = len(g - r)
    p, rcl, f1 = prf(tp, fp, fn)
    per_query.append({"query": q, "tp": tp, "fp": fp, "fn": fn,
                      "precision": p, "recall": rcl, "f1": f1})
    S_tp += tp; S_fp += fp; S_fn += fn

micro_p, micro_r, micro_f1 = prf(S_tp, S_fp, S_fn)
macro_p = sum(d["precision"] for d in per_query) / len(per_query)
macro_r = sum(d["recall"]    for d in per_query) / len(per_query)
macro_f1 = sum(d["f1"]       for d in per_query) / len(per_query)

summary = {
    "n_queries": len(per_query),
    "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
    "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump({"summary": summary, "per_query": per_query},
              f, ensure_ascii=False, indent=2)

print("✅ 평가 완료 —", summary["n_queries"], "개 쿼리")
print("• Micro  P/R/F1 : "
      f"{micro_p:.3f} / {micro_r:.3f} / {micro_f1:.3f}")
print("• Macro  P/R/F1 : "
      f"{macro_p:.3f} / {macro_r:.3f} / {macro_f1:.3f}")
print("→ 세부 결과:", OUT_PATH)
