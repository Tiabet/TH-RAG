#!/usr/bin/env python
# evaluate_kgrag_hardcoded.py
import json, re, string
from collections import Counter
from pathlib import Path

# ---------- 하드코딩된 파일 경로 ----------
PRED_PATH = Path("Result/Ours/hotpot_result_v3_50_5.json")
GOLD_PATH = Path("hotpotQA/qa.json")
WRONG_PATH = Path("Result/Ours/wrong_cases.json")  # 🔹 추가된 저장 경로

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
    contain_correct = 0
    missing = 0
    wrong_cases = []

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

        # accuracy 기준으로 카운트 및 틀린 문제 저장
        if normalize(gold_ans) in normalize(pred_ans):
            contain_correct += 1
        else:
            wrong_cases.append({"query": q, "gold": gold_ans, "pred": pred_ans})  # ✅ 이 조건만 사용

    compared = len(gold) - missing
    em         = em_sum         / compared if compared else 0
    f1         = f1_sum         / compared if compared else 0
    precision  = precision_sum  / compared if compared else 0
    recall     = recall_sum     / compared if compared else 0
    accuracy   = contain_correct / compared if compared else 0

    print(f"#items compared : {compared}/{len(gold)} (missing={missing})")
    print(f"Exact‑Match     : {em:.3f}")
    print(f"F1              : {f1:.3f}")
    print(f"Precision       : {precision:.3f}")
    print(f"Recall          : {recall:.3f}")
    print(f"Accuracy        : {accuracy:.3f}  (gold answer ⊆ prediction)")

    # 틀린 문제 저장 (accuracy 기준)
    with WRONG_PATH.open("w", encoding="utf-8") as f:
        json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
    print(f"\n❌ 틀린 문제 {len(wrong_cases)}개를 {WRONG_PATH}에 저장했습니다.")

if __name__ == "__main__":
    main()