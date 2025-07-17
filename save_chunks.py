# save_chunks.py
# --- 하드코딩 설정 ---
TXT_PATH   = "MultihopRAG/contexts.txt"   # 원본 텍스트
OUT_PATH   = "MultihopRAG/chunks_v1.txt"                 # 청크 저장 파일
MODEL_NAME = "gpt-4o-mini"                            # tiktoken 모델
MAX_TOKENS = 1200                                     # 청크 토큰 수
OVERLAP    = 100                                      # 청크 간 겹침 토큰 수
# ----------------------

from pathlib import Path
import tiktoken

# 청크 함수
def chunk_text(text: str, max_tokens: int, overlap: int, model: str):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        chunk_tokens = tokens[start : start + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks

# 1) 원본 읽기
full_text = Path(TXT_PATH).read_text(encoding="utf-8")

# 2) 청킹
chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)
print(f"✅ 청크 {len(chunks)}개 생성")

# 3) 저장 (한 줄 = 한 청크, 줄바꿈 제거)
with Path(OUT_PATH).open("w", encoding="utf-8") as f:
    for ch in chunks:
        f.write(ch.replace("\n", " ").strip() + "\n")

print(f"💾 저장 완료 → {OUT_PATH}")
