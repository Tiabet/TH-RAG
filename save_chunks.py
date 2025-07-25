from pathlib import Path
import tiktoken
import json
import hashlib

# 파일 경로
TXT_PATH = Path("UltraDomain/Mix/contexts.txt")
OUT_JSON_PATH = Path("UltraDomain/Mix/kv_store_text_chunks.json")

# 청크 설정
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1200
OVERLAP = 100

# 텍스트 → 청크 리스트
def chunk_text(text: str, max_tokens: int, overlap: int, model: str):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        chunk_tokens = tokens[start : start + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks

# 1) 전체 텍스트 불러오기
full_text = TXT_PATH.read_text(encoding="utf-8")

# 2) 청킹
chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)
print(f"✅ 총 {len(chunks)}개의 청크 생성 완료.")

# 3) kv-store JSON 저장
kv_data = {}
for i, chunk in enumerate(chunks):
    cleaned = chunk.replace("\n", " ").strip()
    chunk_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
    chunk_id = f"chunk-{chunk_hash}"
    kv_data[chunk_id] = {
        "tokens": len(cleaned.split()),  # 또는 실제 토큰 수 사용 가능
        "content": cleaned,
        "chunk_order_index": i,
        "full_doc_id": "doc-from-contexts"
    }

# 4) 저장
with OUT_JSON_PATH.open("w", encoding="utf-8") as f:
    json.dump(kv_data, f, indent=2, ensure_ascii=False)

print(f"💾 저장 완료 → {OUT_JSON_PATH}")
