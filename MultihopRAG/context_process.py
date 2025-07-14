import json

# JSON 파일 불러오기
with open('MultihopRAG/corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# body만 추출
bodies = [item['body'] for item in data if 'body' in item]

# txt 파일로 저장
with open('contexts.txt', 'w', encoding='utf-8') as f:
    for body in bodies:
        f.write(body.strip() + "\n\n")  # 각 body 사이에 빈 줄 추가