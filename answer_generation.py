import json
from graph_based_rag import GraphRAG
from tqdm import tqdm  # tqdm 임포트

# 입력 파일과 출력 파일 경로 설정
input_path = "result/agriculture_lightrag_result.json"
output_path = "result/agriculture_graphrag_result.json"

# GraphRAG 인스턴스 생성
rag = GraphRAG()

# 입력 파일 읽기
with open(input_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)

# 결과 저장할 리스트
output_data = []

# tqdm으로 루프 감싸기
for item in tqdm(questions, desc="Generating answers"):
    query = item["query"]
    try:
        answer = rag.answer(query)
    except Exception as e:
        answer = f"[Error] {e}"
    output_data.append({"query": query, "result": answer})

# 결과 JSON으로 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved output to {output_path}")
