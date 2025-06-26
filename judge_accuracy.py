# evaluate_accuracy.py

import json
import openai
import os
from tqdm import tqdm
from prompt.accuracy_evaluation import ACCURACY_EVALUATION_PROMPT  # 여기에 your prompt 템플릿이 문자열로 정의되어 있어야 합니다

# 모델 이름
MODEL_NAME = "gpt-4o-mini"  # 필요시 gpt-3.5-turbo 등으로 변경
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_alignment(answer, response):
    prompt = ACCURACY_EVALUATION_PROMPT.format(answer=answer, response=response)
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return completion.choices[0].message["content"]

def main():
    answer_data = load_json("hotpotQA/hotpot_questions.json")
    response_data = load_json("hotpotQA/result/hotpot_kgrag_results.json")
    
    results = []
    for i in tqdm(range(len(answer_data))):
        answer = answer_data[i]["result"]
        response = response_data[i]["result"]

        evaluation = evaluate_alignment(answer, response)

        results.append({
            "id": answer_data[i].get("id", i),
            "answer": answer,
            "response": response,
            "evaluation": evaluation
        })

    with open("hotpotQA/hotpot_accuracy_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Accuracy evaluation completed and saved to hotpotQA/hotpot_accuracy_evaluation.json")

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
