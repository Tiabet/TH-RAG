import json
import re

# JSON 불러오기
with open("Result/Ours/hotpot_result_v3_50_5.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 텍스트 파일로 저장할 리스트
formatted_lines = []

# 각 항목 처리
for idx, item in enumerate(data):
    text = item.get("context_token", "")
    if isinstance(text, str) and text.strip():
        text = re.sub(r'(\[Chunk \d+\])', r'\n\1', text)
        text = re.sub(r'(\(Edge \d+ \| rank=\d+ score=[0-9.]+\))', r'\n\1', text)
        data[idx]["context_token"] = text.strip()  # JSON에도 업데이트
        formatted_lines.append(f"### Sample {idx}\n{text.strip()}\n")

# JSON 그대로 저장 (원본 유지용)
with open("formatted_context_token.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# 사람이 보기 좋은 텍스트 파일도 저장
with open("formatted_context_token.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(formatted_lines))

print("✅ 저장 완료: formatted_context_token.json + formatted_context_token.txt")
