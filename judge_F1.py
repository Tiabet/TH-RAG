#!/usr/bin/env python
# evaluate_kgrag_hardcoded.py
import json, re, string
from collections import Counter
from pathlib import Path

# ---------- 하드코딩된 파일 경로 ----------
PRED_PATH = Path("Result/Ours/hotpot_result_temp.json")
GOLD_PATH = Path("hotpotQA/qa.json")

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
    ### --- NEW --- ###
    contain_correct = 0          # Accuracy용 카운터
    ### ------------- ###
    missing = 0

    for q, gold_ans in gold.items():
        if q not in pred:
            missing += 1
            continue
        pred_ans = pred[q]
        em, f1_val, prec, rec = compute_metrics(pred_ans, gold_ans)
        em_sum += em
        f1_sum += f1_val
        precision_sum += prec
        recall_sum += rec
        ### --- NEW --- ###
        # 정답 문자열이 예측 안에 '포함'되어 있으면 correct
        if normalize(gold_ans) in normalize(pred_ans):
            contain_correct += 1
        ### ------------- ###

    compared = len(gold) - missing
    em         = em_sum         / compared if compared else 0
    f1         = f1_sum         / compared if compared else 0
    precision  = precision_sum  / compared if compared else 0
    recall     = recall_sum     / compared if compared else 0
    accuracy   = contain_correct / compared if compared else 0
    ### ------------- ###

    print(f"#items compared : {compared}/{len(gold)} (missing={missing})")
    print(f"Exact‑Match     : {em:.3f}")
    print(f"F1              : {f1:.3f}")
    print(f"Precision       : {precision:.3f}")
    print(f"Recall          : {recall:.3f}")
    print(f"Accuracy        : {accuracy:.3f}  (gold answer ⊆ prediction)")
    ### ------------- ###

if __name__ == "__main__":
    main()
