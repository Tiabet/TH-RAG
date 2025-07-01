import xml.etree.ElementTree as ET
import tiktoken

# # GEXF 파일 파싱
# tree = ET.parse("hotpotQA/graph_v3.gexf")
# root = tree.getroot()

# # 네임스페이스 추출
# ns = {'g': root.tag.split('}')[0].strip('{')}  # 예: {'g': 'http://www.gexf.net/1.2draft'}

# # TikToken 토크나이저 선택
# encoding = tiktoken.encoding_for_model("gpt-4")

# # 모든 <attvalue for="2" value="..."> 추출
# values = [
#     att.get("value")
#     for att in root.findall(".//g:attvalue", ns)
#     if att.get("for") == "2" and att.get("value") is not None
# ]

# # 문자열 합치기
# full_text = " ".join(values)

# # 토큰 개수 세기
# token_count = len(encoding.encode(full_text))
# print(f"총 토큰 길이: {token_count}")

# 읽을 파일 경로
file_path = "hotpotQA/contexts.txt"  # 여기에 너의 실제 .txt 파일 경로 넣어

# 파일에서 텍스트 읽기
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# # 토크나이저 선택 (GPT-4 기준)
# encoding = tiktoken.encoding_for_model("gpt-4")

# # 토큰 개수 계산
# token_count = len(encoding.encode(text))
# print(f"총 토큰 길이: {token_count}")

# 문자 수 세기
char_count = len(text)
print(f"총 문자 수: {char_count}")
