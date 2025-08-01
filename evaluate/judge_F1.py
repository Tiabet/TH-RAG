#!/usr/bin/env python
# evaluate_kgrag_hardcoded.py
import json, re, string, sys, argparse
from collections import Counter
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import get_config

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

def main(dataset_name: str, pred_path_param: str = None, gold_path_param: str = None):
    """
    Main function for F1 score evaluation
    
    Args:
        dataset_name: 데이터셋 이름
        pred_path_param: 예측 결과 파일 경로 (선택사항)
        gold_path_param: 정답 파일 경로 (선택사항)
    """
    config = get_config(dataset_name)
    
    # 경로 설정
    pred_path = Path(pred_path_param) if pred_path_param else config.get_answer_file(answer_type="short")
    gold_path = Path(gold_path_param) if gold_path_param else config.get_qa_file()
    
    if not pred_path.exists():
        print(f"❌ Prediction file not found: {pred_path}")
        return None
    
    if not gold_path.exists():
        print(f"❌ Gold file not found: {gold_path}")
        return None
    
    print(f"📂 Evaluating dataset: {dataset_name}")
    print(f"📂 Predictions: {pred_path}")
    print(f"📂 Gold answers: {gold_path}")
    
    gold = load_pairs(gold_path, "answer")
    pred = load_pairs(pred_path, "result")

    em_sum = f1_sum = precision_sum = recall_sum = 0
    ### --- NEW --- ###
    contain_correct = 0          # Accuracy용 카운터
    ### ------------- ###
    missing = 0

    for q, gold_ans in gold.items():
        if q not in pred or "[Error]" in pred[q]:
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
    
    # 결과 저장
    results = {
        'dataset': dataset_name,
        'compared': compared,
        'total': len(gold),
        'missing': missing,
        'exact_match': em,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }
    
    # 평가 결과 파일 저장
    eval_output_path = config.get_evaluation_file(eval_method="f1")
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Evaluation results saved to: {eval_output_path}")
    
    # 파이프라인 상태 업데이트
    state = config.load_pipeline_state() or {}
    state[dataset_name] = state.get(dataset_name, {})
    state[dataset_name]['evaluation_f1'] = {
        'completed': True,
        'pred_file': str(pred_path),
        'gold_file': str(gold_path),
        'eval_file': str(eval_output_path),
        'f1_score': f1,
        'accuracy': accuracy
    }
    config.save_pipeline_state(state)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 evaluation for KGRAG")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--pred", help="Prediction file path")
    parser.add_argument("--gold", help="Gold answer file path")
    
    args = parser.parse_args()
    main(args.dataset, args.pred, args.gold)
