import argparse
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

if "SSL_CERT_FILE" in os.environ:
    print("⚠️ Removing problematic SSL_CERT_FILE:", os.environ["SSL_CERT_FILE"])
    os.environ.pop("SSL_CERT_FILE")

# ==== Argument Parsing ====
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input .txt file")
parser.add_argument("--output", required=True, help="Output .json file")
args = parser.parse_args()

# ==== 고정 설정 ====
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1200
OVERLAP = 100
MAX_WORKERS = 50

# ==== Load env ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==== Chunk Function ====
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

# ==== Call Model ====
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
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content.strip())
        if isinstance(data, list):
            for item in data:
                item["chunk_id"] = index
        elif isinstance(data, dict):
            data["chunk_id"] = index
        return data
    except Exception as e:
        return {"error": str(e), "chunk_index": index}

# ==== Load Text & Chunk ====
full_text = Path(args.input).read_text(encoding="utf-8")
chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)

# ==== Load Existing ====
output_path = Path(args.output)
if output_path.exists():
    with open(output_path, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
            print(f"🔄 Loaded {len(results)} existing results.")
        except json.JSONDecodeError:
            print("⚠️ JSON decode error. Starting fresh.")
            results = [None] * len(chunks)
else:
    results = [None] * len(chunks)

pending_indices = [i for i, r in enumerate(results) if r is None or "error" in r]
print(f"🕐 Remaining chunks to process: {len(pending_indices)}")

# ==== Run ====
client = openai.OpenAI(api_key=OPENAI_API_KEY)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(call_model, client, MODEL_NAME, chunks[idx], idx): idx
        for idx in pending_indices
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
        idx = futures[future]
        results[idx] = future.result()

        if (pending_indices.index(idx) + 1) % 10 == 0 or idx == pending_indices[-1]:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

print(f"✅ Extraction complete. Output saved to {args.output}")
