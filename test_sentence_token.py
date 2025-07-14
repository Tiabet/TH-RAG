import json
import tiktoken
from pathlib import Path

JSON_FILE = Path("hotpotQA/graph_v4.json")

# ── JSON 로드 ──────────────────────────────────────────
data = json.loads(JSON_FILE.read_text(encoding="utf-8"))

def extract_sentences(obj):
    """재귀적으로 'sentence' 값을 모두 수집."""
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, str) and "sentence" in obj and v == obj.get("sentence"):
                yield obj["sentence"]
            else:
                yield from extract_sentences(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_sentences(item)

# ── sentence 수집 & 중복 제거 ──────────────────────────
all_sentences   = list(extract_sentences(data))
unique_sentences = list(dict.fromkeys(all_sentences))  # 순서 유지하며 중복 제거

if not unique_sentences:
    raise ValueError("'sentence' 항목을 찾지 못했습니다.")

# ── 토큰 계산 (고유 문장 기준) ──────────────────────────
encoding   = tiktoken.encoding_for_model("gpt-4o-mini")
token_cnt  = len(encoding.encode("\n".join(unique_sentences)))

print(f"✅ 총 문장 수 (중복 포함) : {len(all_sentences)}")
print(f"✅ 고유 문장 수          : {len(unique_sentences)}")
print(f"🧠 고유 문장 기준 토큰 수 : {token_cnt}")
