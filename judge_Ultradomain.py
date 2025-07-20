import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from prompt.evaluation import EVALUATION_PROMPT 
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import Counter

load_dotenv()
# ────────────────────── 설정 ──────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MAX_WORKERS = 15         # 스레드 개수(네트워크·API 한도에 맞춰 조절)
RANDOM_SEED = 42           # 재현성 필요 시 None 대신 정수
model_name = "gpt-4o-mini"  # 사용할 모델 이름
# ──────────────────────────────────────────────────

with open("Result/Ours/agriculture_result.json", encoding="utf-8") as f1, \
     open("Result/LightRAG/hybrid_agriculture_result.json", encoding="utf-8") as f2:
    graph_results = json.load(f1)
    light_results = json.load(f2)

out_path = "Result/Ours/agriculture_LightRAG.json"

my_rag = "KGRAG"
other_rag = "LightRAG"

# answer1/answer2 위치 균등 분배
N = len(graph_results)
indices = list(range(N))
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
random.shuffle(indices)
kg_first_set = set(indices[: N // 2])


def extract_json_from_response(response_text: str) -> str:
    """```json ...``` 또는 JSON 본문만 추출"""
    if response_text.strip().startswith("```"):
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return response_text.strip()

def judge_one(idx: int, g_answer: dict, l_answer: dict) -> tuple[int, dict]:
    """한 쌍(query)의 평가를 수행하고 (idx, result_dict) 반환"""
    query = g_answer["query"]

    if idx in kg_first_set:
        answer1, answer2 = g_answer["result"], l_answer["result"]
        answer1_model, answer2_model = my_rag, other_rag
    else:
        answer1, answer2 = l_answer["result"], g_answer["result"]
        answer1_model, answer2_model = other_rag, my_rag

    prompt = EVALUATION_PROMPT.format(query=query, answer1=answer1, answer2=answer2)
    response = client.chat.completions.create(
        model=  model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw_content = response.choices[0].message.content.strip()

    try:
        json_string = extract_json_from_response(raw_content)
        eval_json = json.loads(json_string)
        result = {
            "query": query,
            "answer1_model": answer1_model,
            "answer2_model": answer2_model,
            **eval_json
        }
    except Exception as e:
        result = {
            "query": query,
            "answer1_model": answer1_model,
            "answer2_model": answer2_model,
            "error": str(e),
            "raw_response": raw_content
        }
    return idx, result

# ─────────── 병렬 실행 ───────────
judged_results_tmp = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [
        executor.submit(judge_one, idx, g, l)
        for idx, (g, l) in enumerate(zip(graph_results, light_results))
    ]
    for f in tqdm(as_completed(futures), total=N, desc="Evaluating answers"):
        idx, res = f.result()
        judged_results_tmp[idx] = res   # 딕셔너리로 모으면 스레드 안전

# 인덱스 기준으로 정렬해 리스트로 변환
judged_results = [judged_results_tmp[i] for i in range(N)]

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(judged_results, f, indent=2, ensure_ascii=False)

print(f"완료! 결과가 {out_path} 에 저장되었습니다.")

# 평가 결과 불러오기
with open(out_path, encoding="utf-8") as f:
    data = json.load(f)

categories = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
model_scores = {
    my_rag : Counter(),
    other_rag : Counter()
}

# 결과 집계
for item in data:
    if "error" in item:
        continue
    for cat in categories:
        winner = item.get(cat, {}).get("Winner")
        if winner == "Answer 1":
            model_scores[item["answer1_model"]][cat] += 1
        elif winner == "Answer 2":
            model_scores[item["answer2_model"]][cat] += 1

# 시각화 데이터 정리
labels = categories
kgrag_values = [model_scores[my_rag][cat] for cat in labels]
lightrag_values = [model_scores[other_rag][cat] for cat in labels]
totals = [k + l for k, l in zip(kgrag_values, lightrag_values)]

kgrag_pct = [k / t * 100 if t > 0 else 0 for k, t in zip(kgrag_values, totals)]
lightrag_pct = [l / t * 100 if t > 0 else 0 for l, t in zip(lightrag_values, totals)]

# 막대 그래프
x = range(len(labels))
bar_width = 0.35
fig, ax = plt.subplots()
ax.bar(x, kgrag_pct, width=bar_width, label=my_rag, color='blue')
ax.bar([i + bar_width for i in x], lightrag_pct, width=bar_width, label=other_rag, color='orange')

# 퍼센트 텍스트
for i in x:
    ax.text(i, kgrag_pct[i] + 1, f"{kgrag_pct[i]:.1f}%", ha='center')
    ax.text(i + bar_width, lightrag_pct[i] + 1, f"{lightrag_pct[i]:.1f}%", ha='center')

# 설정
ax.set_xlabel("Evaluation Criteria")
ax.set_ylabel("Winning Percentage (%)")
ax.set_title("KG-RAG vs hyde Evaluation Results (agriculture)")
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()
