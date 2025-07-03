import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import openai
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt.extract_graph import EXTRACTION_PROMPT
# from prompt.normal_extract_graph import EXTRACTION_PROMPT

if "SSL_CERT_FILE" in os.environ:
    print("⚠️ Removing problematic SSL_CERT_FILE:", os.environ["SSL_CERT_FILE"])
    os.environ.pop("SSL_CERT_FILE")

# ==== 설정 ====
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILES = ["hotpotQA/contexts_distractor_1000.txt"]
OUTPUT_FILE = "hotpotQA/graph_v1.json"
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1200
OVERLAP = 100
MAX_WORKERS = 50

# ==== 텍스트 청크 함수 ====
def chunk_text(text: str, max_tokens: int, overlap: int, model: str) -> List[str]:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap
    return chunks

# ==== 모델 호출 함수 ====
def call_model(client: openai.OpenAI, model: str, chunk: str, index: int) -> dict:
    prompt_filled = EXTRACTION_PROMPT.replace("{{document}}", chunk.strip())
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in information extraction."},
                {"role": "user", "content": prompt_filled},
            ],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        return {"error": str(e), "chunk_index": index}

# ==== 전체 텍스트 로딩 및 청크 생성 ====
texts = [Path(p).read_text(encoding="utf-8") for p in INPUT_FILES]
full_text = "\n".join(texts)
chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)

# ==== 이전 결과 불러오기 ====
if Path(OUTPUT_FILE).exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
            print(f"🔄 Loaded {len(results)} existing results.")
        except json.JSONDecodeError:
            print("⚠️ JSON decode error. Starting fresh.")
            results = [None] * len(chunks)
else:
    results = [None] * len(chunks)

# ==== 처리되지 않은 청크만 선택 ====
pending_indices = [i for i, r in enumerate(results) if r is None or "error" in r]
print(f"🕐 Remaining chunks to process: {len(pending_indices)}")

# ==== 모델 호출 ====
client = openai.OpenAI()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(call_model, client, MODEL_NAME, chunks[idx], idx): idx
        for idx in pending_indices
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
        idx = futures[future]
        results[idx] = future.result()

        # 중간 저장
        if (pending_indices.index(idx) + 1) % 10 == 0 or idx == pending_indices[-1]:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

print(f"✅ Extraction complete. Output saved to {OUTPUT_FILE}")
