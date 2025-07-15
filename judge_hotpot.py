#!/usr/bin/env python
# evaluate_kgrag_hardcoded.py
import json, re, string
from collections import Counter
from pathlib import Path

# 하드코딩된 파일 경로
GOLD_PATH = Path("hotpotQA/sampled_qa_200_v2.json")
PRED_PATH = Path("hotpotQA/result2/kgrag_v2_200.json")

# ---------- text normalization ----------
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

# ---------- metrics ----------
def compute_metrics(pred: str, gold: str):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        em = int(pred_tokens == gold_tokens)
        return em, 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    em = int(pred_tokens == gold_tokens)

    return em, f1, precision, recall

# ---------- driver ----------
def load_pairs(path: Path, answer_key: str):
    with path.open(encoding="utf-8") as f:
        return {d["query"]: d[answer_key] for d in json.load(f)}

def main():
    gold = load_pairs(GOLD_PATH, "answer")
    pred = load_pairs(PRED_PATH, "result")

    em_sum = f1_sum = precision_sum = recall_sum = 0
    missing = 0

    for q, gold_ans in gold.items():
        if q not in pred:
            missing += 1
            continue
        em, f1_val, prec, rec = compute_metrics(pred[q], gold_ans)
        em_sum += em
        f1_sum += f1_val
        precision_sum += prec
        recall_sum += rec

    compared = len(gold) - missing
    em = em_sum / compared if compared else 0
    f1 = f1_sum / compared if compared else 0
    precision = precision_sum / compared if compared else 0
    recall = recall_sum / compared if compared else 0

    print(f"#items compared : {compared}/{len(gold)} (missing={missing})")
    print(f"Exact‑Match     : {em:.3f}")
    print(f"F1              : {f1:.3f}")
    print(f"Precision       : {precision:.3f}")
    print(f"Recall          : {recall:.3f}")

if __name__ == "__main__":
    main()
