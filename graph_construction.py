import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import openai
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# from prompt.extract_graph import EXTRACTION_PROMPT
from prompt.normal_extract_graph import EXTRACTION_PROMPT

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILES = ["contexts.txt"]
OUTPUT_FILE = "graph.json"
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1200
OVERLAP = 100
MAX_WORKERS = 10  # 동시에 실행할 스레드 개수

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

# ==== 실행 준비 ====
texts = [Path(p).read_text(encoding="utf-8") for p in INPUT_FILES]
full_text = "\n".join(texts)
chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)
client = openai.OpenAI()

results = [None] * len(chunks)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # future → (index, future)
    futures = {
        executor.submit(call_model, client, MODEL_NAME, chunk, idx): idx
        for idx, chunk in enumerate(chunks)
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
        idx = futures[future]
        results[idx] = future.result()

        # 10개마다 중간 저장
        if (idx + 1) % 10 == 0 or idx == len(chunks) - 1:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

print(f"✅ Extraction complete. Output saved to {OUTPUT_FILE}")
