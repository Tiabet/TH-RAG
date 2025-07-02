import json
import openai
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt.accuracy_evaluation import ACCURACY_EVALUATION_PROMPT

# 모델 이름
MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

MAX_WORKERS = 10  # 병렬로 실행할 스레드 수 (rate limit을 고려해 적절히 설정)

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_alignment_single(item):
    i, answer, response = item
    try:
        prompt = ACCURACY_EVALUATION_PROMPT.format(answer=answer, response=response)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        evaluation = completion.choices[0].message.content.strip()
    except Exception as e:
        evaluation = f"Error: {str(e)}"

    return {
        "id": i,
        "answer": answer,
        "response": response,
        "evaluation": evaluation
    }

def main():
    answer_data = load_json("hotpotQA/questions.json")
    response_data = load_json("hotpotQA/naive_answers.json")

    items = [(answer_data[i].get("id", i),
              answer_data[i]["answer"],
              response_data[i]["answer"]) for i in range(len(answer_data))]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_alignment_single, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    with open("hotpotQA/result/hotpot_accuracy_evaluation_naive.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Accuracy evaluation completed and saved to hotpotQA/result/hotpot_accuracy_evaluation_naive.json")

    yes_count = sum(1 for r in results
                    if str(r.get("evaluation", "")).lower().startswith("yes"))
    accuracy = yes_count / len(results) if results else 0

    print(f"\nAccuracy (evaluation == 'yes') : {accuracy:.4f}  "
          f"({accuracy*100:.2f}%)  [{yes_count}/{len(results)}]")

if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    main()
