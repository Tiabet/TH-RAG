import json, sys, os
from pathlib import Path
from graph_based_rag_long import GraphRAG
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

# 프로젝트 루트로 작업 디렉터리 변경
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# 인코더 초기화
enc = tiktoken.encoding_for_model("gpt-4o")

# 설정
input_path       = "UltraDomain/Mix/qa.json"
output_path      = "Result/Ours/mix_result.json"
chunk_log_path   = "Result/Ours/Chunks/used_chunks_mix.jsonl"
temp_output_path = output_path.replace(".json", "_temp.json")

MAX_WORKERS = 30
TOP_K1 = 50
TOP_K2 = 5

# 결과 디렉터리 생성
os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(chunk_log_path), exist_ok=True)

# GraphRAG 인스턴스 생성
rag = GraphRAG()
chunk_log_file = open(chunk_log_path, "w", encoding="utf-8")

# 질문 로딩
with open(input_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)

# 결과 리스트 초기화
output_data = [None] * len(questions)

# 처리 함수
def process(index_query):
    idx, item = index_query
    query = item.get("query", "")
    try:
        answer, spent, context = rag.answer(query=query, top_k1=TOP_K1, top_k2=TOP_K2)
        chunk_ids = getattr(rag, "last_chunk_ids", [])
    except Exception as e:
        answer = f"[Error] {e}"
        spent = 0.0
        context = ""
        chunk_ids = []

    # chunk-id 로그
    for cid in chunk_ids:
        chunk_log_file.write(json.dumps({"query": query, "chunk_id": cid}, ensure_ascii=False) + "\n")

    # 문장 기반 chunk-id 로그
    sentence_chunk_ids = set(getattr(rag, "all_sentence_chunk_ids", []))
    for cid in sentence_chunk_ids:
        chunk_log_file.write(json.dumps({"query": query, "sentence_chunk_id": cid}, ensure_ascii=False) + "\n")

    # 결과 저장
    result = {
        "query": query,
        "result": answer,
        "time": spent,
        "context_token": context,
    }
    return idx, result

# 병렬 실행
completed = 0
save_every = 10

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process, (i, item)) for i, item in enumerate(questions)]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
        idx, result = future.result()
        output_data[idx] = result
        completed += 1

        if completed % save_every == 0:
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

chunk_log_file.close()

# 최종 결과 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ 최종 결과 저장 완료 → {output_path}")

# # 평균 시간 및 토큰 수 계산
# valid_items = [it for it in output_data if it and not it["result"].startswith("[Error]")]

# if valid_items:
#     avg_time = sum(it["time"] for it in valid_items) / len(valid_items)
#     avg_tokens = sum(len(enc.encode(it["context_token"])) for it in valid_items) / len(valid_items)

#     print(f"\n📊 평균 소요 시간: {avg_time:.2f}초")
#     print(f"📊 평균 컨텍스트 토큰 수: {avg_tokens:.1f}개")
# else:
#     print("⚠️ 유효한 결과가 없어 평균을 계산할 수 없습니다.")
