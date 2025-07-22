import json, re, html
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

WRONG_SF_PATH  = Path("Result/Ours/wrong_supporting_facts.json")
RETRIEVED_PATH = Path("Result/Ours/hotpot_result_sampled_v5.json")
OUT_PATH       = Path("Result/Ours/supporting_recall.json")

# ── normalize (supporting·context·query 공통) ──────────────────────────
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).lower().strip()
    return text

TOK_RE = re.compile(r"[a-z0-9]+")
def tokset(t: str) -> set[str]: return set(TOK_RE.findall(t))
def matched(s: str, ctx: str) -> bool:
    if s in ctx: return True
    st, ct = tokset(s), tokset(ctx)
    if st and len(st & ct) / len(st) >= .7:   # 70 % 이상 커버
        return True
    if SequenceMatcher(None, s, ctx).quick_ratio() >= .7:
        return True
    return False

# ── 1) gold sentences ────────────────────────────────────────────────
with WRONG_SF_PATH.open(encoding="utf-8") as f:
    gold_items = json.load(f)

gold_sentences = {
    normalize(item["query"]): [normalize(sf["sentence"])
                               for sf in item["supporting_facts"]]
    for item in gold_items
}

# ── 2) retrieved contexts ────────────────────────────────────────────
def ctx_text(obj):
    if isinstance(obj.get("context_token"), str):
        return obj["context_token"]
    if isinstance(obj.get("context_tokens"), list):
        return " ".join(obj["context_tokens"])
    if isinstance(obj.get("context"), str):
        return obj["context"]
    return ""

with RETRIEVED_PATH.open(encoding="utf-8") as f:
    retrieved = json.load(f)

retrieved_context = defaultdict(str)
for obj in retrieved:
    q_norm = normalize(obj["query"])
    retrieved_context[q_norm] += " " + ctx_text(obj)

# ── 3) recall 계산 ───────────────────────────────────────────────────
total_gold = total_found = 0
per_query  = []

for q, g_sents in gold_sentences.items():
    ctx = normalize(retrieved_context.get(q, ""))
    found = sum(1 for s in g_sents if matched(s, ctx))
    per_query.append({
        "query": q,
        "n_gold": len(g_sents),
        "n_found": found,
        "recall": found / len(g_sents) if g_sents else 0.0
    })
    total_gold  += len(g_sents)
    total_found += found

summary = {
    "total_supporting_facts": total_gold,
    "total_found": total_found,
    "overall_recall": total_found / total_gold if total_gold else 0.0,
    "overall_missing_ratio": 1 - total_found / total_gold if total_gold else 0.0
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump({"summary": summary, "per_query": per_query},
              f, ensure_ascii=False, indent=2)

print("✅ 체크 완료")
print(f"• 전체 supporting facts : {total_gold}")
print(f"• 가져온 문장            : {total_found}")
print(f"• Recall                : {summary['overall_recall']:.3%}")
print(f"• 누락 비율              : {summary['overall_missing_ratio']:.3%}")
print(f"→ 세부 결과는 {OUT_PATH} 에 저장했습니다")
