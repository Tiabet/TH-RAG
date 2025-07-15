import json
from pathlib import Path
from statistics import mean

# -------------------- 정확도(Accuracy) -------------------- #
def normalize(text: str) -> str:
    """간단한 전처리: 앞뒤 공백 제거 및 소문자화."""
    return text.strip().lower()

def accuracy(ref: str, hyp: str) -> int:
    """정확도: 완전 일치면 1, 아니면 0"""
    return int(normalize(ref) == normalize(hyp))
# ---------------------------------------------------------- #
def load_q2a(path: str) -> dict:
    """query → answer/result 딕셔너리 로드 (JSON·JSONL 모두 지원)"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["query"]: item["result" if "result" in item else "answer"]
            for item in data}

def main(
    ref_file="hotpotQA/sampled_qa_100.json",
    hyp_file="hotpotQA/result2/kgrag_v2_100.json",
):

    ref = load_q2a(ref_file)
    hyp = load_q2a(hyp_file)

    # ⬇️ 두 파일 모두에 존재하는 쿼리만 사용
    common_queries = sorted(set(ref) & set(hyp))
    if not common_queries:
        print("두 파일에 공통으로 존재하는 query가 없습니다.")
        return

    acc_list = []
    for q in common_queries:
        acc_list.append(accuracy(ref[q], hyp[q]))

    avg_acc = mean(acc_list)
    print(f"Common queries        : {len(common_queries)}")
    print(f"Average Accuracy (∩)  : {avg_acc:.4f}")

if __name__ == "__main__":
    main()