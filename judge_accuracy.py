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
    return completion.choices[0].message.content.strip()

def main():
    answer_data = load_json("hotpotQA/hotpot_questions.json")
    response_data = load_json("hotpotQA/result/hotpot_kgrag_results.json")
    
    results = []
    for i in tqdm(range(len(answer_data))):
        answer = answer_data[i]["result"]
        response = response_data[i]["result"]

        evaluation = evaluate_alignment(answer, response)
        print(evaluation)

        results.append({
            "id": answer_data[i].get("id", i),
            "answer": answer,
            "response": response,
            "evaluation": evaluation
        })

    with open("hotpotQA/result/hotpot_accuracy_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Accuracy evaluation completed and saved to hotpotQA/result/hotpot_accuracy_evaluation.json")

    # ----- 추가: accuracy 계산 -----
    yes_count = sum(1 for r in results
                    if str(r.get("evaluation", "")).lower().startswith("yes"))
    accuracy = yes_count / len(results) if results else 0

    print(f"\nAccuracy (evaluation == 'yes') : {accuracy:.4f}  "
          f"({accuracy*100:.2f}%)  [{yes_count}/{len(results)}]")

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
