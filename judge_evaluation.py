import json
import os
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
from prompt.evaluation import EVALUATION_PROPMPT  # 여기에 your prompt 템플릿이 문자열로 정의되어 있어야 합니다
import re
import random

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) 

# Load the two answer files
with open("result/agriculture_kgrag_result.json", encoding="utf-8") as f1, \
     open("result/agriculture_lightrag_result.json", encoding="utf-8") as f2:
    graph_results = json.load(f1)
    light_results = json.load(f2)

# Evaluation 결과 저장 리스트
judged_results = []


def extract_json_from_response(response_text: str) -> str:
    """
    응답에서 ```json ... ``` 또는 그냥 JSON 본문만 추출
    """
    if response_text.strip().startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response_text.strip()  # fallback

# 평가
for g, l in tqdm(zip(graph_results, light_results), total=len(graph_results), desc="Evaluating answers"):
    query = g["query"]
    
    # 랜덤하게 answer1, answer2를 결정
    if random.random() < 0.5:
        answer1, answer2 = g["result"], l["result"]
        answer1_model, answer2_model = "KG-RAG", "LightRAG"
    else:
        answer1, answer2 = l["result"], g["result"]
        answer1_model, answer2_model = "LightRAG", "KG-RAG"

    prompt = EVALUATION_PROPMPT.format(query=query, answer1=answer1, answer2=answer2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    raw_content = response.choices[0].message.content.strip()
    print(raw_content)

    try:
        json_string = extract_json_from_response(raw_content)
        eval_json = json.loads(json_string)
        judged_results.append({
            "query": query,
            "answer1_model": answer1_model,
            "answer2_model": answer2_model,
            **eval_json
        })
    except Exception as e:
        judged_results.append({
            "query": query,
            "answer1_model": answer1_model,
            "answer2_model": answer2_model,
            "error": str(e),
            "raw_response": raw_content
        })

# 저장
with open("result/agriculture_judged_results.json", "w", encoding="utf-8") as f:
    json.dump(judged_results, f, indent=2, ensure_ascii=False)
