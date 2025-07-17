import xml.etree.ElementTree as ET
import tiktoken
file_path = "MultihopRAG/contexts.txt"  # 여기에 너의 실제 .txt 파일 경로 넣어

# 파일에서 텍스트 읽기
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 토크나이저 선택 (GPT-4 기준)
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# 토큰 개수 계산
token_count = len(encoding.encode(text))
print(f"총 토큰 길이: {token_count}")

