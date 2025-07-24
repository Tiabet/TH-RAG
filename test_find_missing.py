import json

# 파일 경로 설정
result_file_path = "result.json"
qa_file_path = "qa.json"

# JSON 파일 불러오기
with open(result_file_path, 'r', encoding='utf-8') as f:
    result_data = json.load(f)

with open(qa_file_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# result 데이터에서 모든 query 추출
result_queries = set(item['query'] for item in result_data)

# qa 데이터에서 누락된 query 찾기
missing_queries = [item for item in qa_data if item['query'] not in result_queries]

# 결과 출력
print(f"누락된 query 개수: {len(missing_queries)}")
for item in missing_queries:
    print(item['query'])