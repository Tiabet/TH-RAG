import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
from prompt.evaluation import EVALUATION_PROPMPT  # 여기에 your prompt 템플릿이 문자열로 정의되어 있어야 합니다

# ────────────────────── 설정 ──────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MAX_WORKERS = 6            # 스레드 개수(네트워크·API 한도에 맞춰 조절)
RANDOM_SEED = 42           # 재현성 필요 시 None 대신 정수
# ──────────────────────────────────────────────────

with open("UltraDomain/result/kgrag_new_v1.json", encoding="utf-8") as f1, \
     open("UltraDomain/result/agriculture_graphragglobal_general_result.json", encoding="utf-8") as f2:
    graph_results = json.load(f1)
    light_results = json.load(f2)

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
        answer1_model, answer2_model = "KG-RAG", "LightRAG"
    else:
        answer1, answer2 = l_answer["result"], g_answer["result"]
        answer1_model, answer2_model = "LightRAG", "KG-RAG"

    prompt = EVALUATION_PROPMPT.format(query=query, answer1=answer1, answer2=answer2)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

# 저장
out_path = "UltraDomain/result/agriculture_judged_results_graphrag.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(judged_results, f, indent=2, ensure_ascii=False)

print(f"완료! 결과가 {out_path} 에 저장되었습니다.")
