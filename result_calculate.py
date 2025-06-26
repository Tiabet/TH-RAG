import json

# 파일 경로 설정
file_path = 'hotpotQA/result/hotpotQA_kgrag_result.json'  # 여기에 파일 경로를 입력하세요

# JSON 파일 불러오기
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 결과 계산
total = len(data)
correct = sum(1 for item in data if item.get('result') == 'yes')

accuracy = correct / total if total > 0 else 0

print(f"Accuracy: {accuracy:.2%}")
