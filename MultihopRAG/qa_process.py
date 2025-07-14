import json

# JSON 파일 불러오기
with open('MultihopRAG/MultiHopRAG.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# query와 answer만 추출
filtered_data = [{"query": item["query"], "answer": item["answer"]} for item in data]

# 새로운 JSON 파일로 저장
with open('MultihopRAG/qa.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)
