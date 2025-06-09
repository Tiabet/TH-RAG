import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import openai
import tiktoken
from tqdm import tqdm  # tqdm 추가

from prompt.extract_graph import EXTRACTION_PROMPT

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== 설정값 수동 지정 ====
INPUT_FILES = ["contexts.txt"]
OUTPUT_FILE = "graph.json"
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1200
OVERLAP = 100


def chunk_text(text: str, max_tokens: int, overlap: int, model: str) -> List[str]:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk = enc.decode(chunk_tokens)
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def call_model(client: openai.OpenAI, model: str, chunk: str, index: int) -> dict:
    prompt_filled = EXTRACTION_PROMPT.replace("{{document}}", chunk.strip())

    try:
        print(f"⏳ Processing chunk {index} with {len(tiktoken.encoding_for_model(model).encode(chunk))} tokens...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in information extraction."},
                {"role": "user", "content": prompt_filled},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        print(f"🧾 Raw response (chunk {index}):\n{text[:300]}...\n")  # 일부 미리보기

        return json.loads(text)
    except Exception as e:
        print(f"❌ Error in chunk {index}: {e}")
        return {"error": str(e), "chunk_index": index}



# ==== 실행 ====
texts = [Path(p).read_text(encoding="utf-8") for p in INPUT_FILES]
full_text = "\n".join(texts)

chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)

client = openai.OpenAI()
results = []

for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
    result = call_model(client, MODEL_NAME, chunk, i)
    results.append(result)

    # 10개마다 중간 저장
    if (i + 1) % 10 == 0 or i == len(chunks) - 1:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

print(f"✅ Extraction complete. Output saved to {OUTPUT_FILE}")
