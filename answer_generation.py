import json
from graph_based_rag import GraphRAG
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 입력/출력 경로
input_path = "result/agriculture_lightrag_result.json"
output_path = "result/agriculture_kgrag_result_v2.json"

# GraphRAG 인스턴스
rag = GraphRAG()

# 입력 로딩
with open(input_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)

# 결과 저장 리스트 (index 순서 보존)
output_data = [None] * len(questions)

# 작업 함수
def process(index_query):
    idx, item = index_query
    query = item["query"]
    try:
        answer = rag.answer(query)
    except Exception as e:
        answer = f"[Error] {e}"
    
    return idx, {"query": query, "result": answer}

# 병렬 처리
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process, (i, item)) for i, item in enumerate(questions)]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
        idx, result = future.result()
        output_data[idx] = result  # 순서 유지

# 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved output to {output_path}")
