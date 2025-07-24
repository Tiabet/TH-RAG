import json
from pathlib import Path

WRONG_CASES_PATH = Path("Result/Ours/wrong_cases.json")
HOTPOT_PATH = Path("hotpotQA/sampled_1000.jsonl")
OUT_PATH = Path("Result/Ours/wrong_supporting_facts.json")

def load_data():
    with WRONG_CASES_PATH.open(encoding="utf-8") as f:
        wrong_cases = json.load(f)

    # 🔽 jsonl 파일은 한 줄씩 json.loads 로 읽어야 함
    with HOTPOT_PATH.open(encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    return wrong_cases, all_data



def build_query_index(data):
    """query 기준으로 HotpotQA 항목을 빠르게 검색하기 위한 인덱스 생성"""
    return {item["question"]: item for item in data}

def extract_supporting_sentences(item):
    context = {title: sents for title, sents in item["context"]}
    extracted = []
    for title, idx in item["supporting_facts"]:
        sentence = context.get(title, [])[idx] if idx < len(context.get(title, [])) else None
        if sentence:
            extracted.append({
                "title": title,
                "sentence_index": idx,
                "sentence": sentence
            })
    return extracted

def main():
    wrong_cases, hotpot_data = load_data()
    # print("🔥 데이터 예시:", json.dumps(hotpot_data[0], indent=2, ensure_ascii=False))
    question_lookup = build_query_index(hotpot_data)
    result = []

    for case in wrong_cases:
        query = case["query"]
        if query not in question_lookup:
            print(f"[SKIP] query not found in original data: {query}")
            continue
        item = question_lookup[query]  # 이건 hotpot 원본 항목
        support_sents = extract_supporting_sentences(item)
        result.append({
            "query": query,
            "supporting_facts": support_sents
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 지원 문장 추출 완료: {len(result)}개 → {OUT_PATH}")


if __name__ == "__main__":
    main()
