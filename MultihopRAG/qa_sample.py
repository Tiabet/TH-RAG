import json
import random

INPUT_PATH  = "MultihopRAG/qa_original.json"   # 원본 파일
OUTPUT_PATH = "MultihopRAG/qa_1000.json"         # 샘플 저장 파일
SAMPLE_SIZE = 1000                        # 원하는 샘플 개수

# 1) JSON 읽기
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2) 샘플링
sample_size = min(SAMPLE_SIZE, len(data))        # 부족할 때 대비
sampled = random.sample(data, sample_size)       # 중복 없이 무작위 추출

# 3) 결과 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sampled, f, ensure_ascii=False, indent=2)

print(f"{sample_size}개 항목을 '{OUTPUT_PATH}'에 저장했습니다.")
