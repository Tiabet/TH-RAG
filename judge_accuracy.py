import json
import openai
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt.accuracy_evaluation import ACCURACY_EVALUATION_PROMPT
from dotenv import load_dotenv

# 모델 이름
MODEL_NAME = "gemini-2.5-flash"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = openai.OpenAI(api_key=OPENAI_API_KEY,
#                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(f"Using Gemini API Key: {GEMINI_API_KEY}")
client = openai.OpenAI(api_key=GEMINI_API_KEY,
                       base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

MAX_WORKERS = 20  # 병렬로 실행할 스레드 수 (rate limit을 고려해 적절히 설정)

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_alignment_single(item):
    i, query, answer, response = item
    try:
        prompt = ACCURACY_EVALUATION_PROMPT.format(query=query, answer=answer, response=response)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        evaluation = completion.choices[0].message.content.strip()
    except Exception as e:
        evaluation = f"Error: {str(e)}"

    return {
        "id": i,
        "query" : query,
        "answer": answer,
        "response": response,
        "evaluation": evaluation
    }

def main():
    answer_data = load_json("MultihopRAG/qa_1000.json")
    response_data = load_json("MultihopRAG/result/kgrag_1000.json")

    items = [(answer_data[i].get("id", i),
              answer_data[i]["query"],
              answer_data[i]["answer"],
              response_data[i]["result"]) for i in range(len(answer_data))]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_alignment_single, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    with open("MultihopRAG/result/kgrag_accuracy_v1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to MultihopRAG/result/kgrag_accuracy_v1.json")

    yes_count = sum(1 for r in results
                    if str(r.get("evaluation", "")).lower().startswith("yes"))
    accuracy = yes_count / len(results) if results else 0

    print(f"\nAccuracy (evaluation == 'yes') : {accuracy:.4f}  "
          f"({accuracy*100:.2f}%)  [{yes_count}/{len(results)}]")

if __name__ == "__main__":
    main()
