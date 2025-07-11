import json
from graph_based_rag import GraphRAG
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 입력/출력 경로
input_path = "UltraDomain/result/agriculture_lightrag_result.json"
output_path = "UltraDomain/result/kgrag_new_v1.json"
temp_output_path = output_path.replace(".json", "_temp.json")

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
    query = item.get("query", "")
    try:
        answer = rag.answer(query)
        print(answer)
    except Exception as e:
        answer = f"[Error] {e}"
    
    result = {"query": query, "result": answer}
    return idx, result

# 병렬 처리
completed = 0
save_every = 100

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process, (i, item)) for i, item in enumerate(questions)]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
        idx, result = future.result()
        output_data[idx] = result  # 순서 유지
        completed += 1

        # 10개마다 임시 저장
        if completed % save_every == 0:
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"[Temp Save] {completed} items saved to {temp_output_path}")

# 최종 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Saved final output to {output_path}")
