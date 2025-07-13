#!/usr/bin/env python3
import json, re
from pathlib import Path

SRC  = Path("InfiniteChoice/infinitechoice.jsonl")
DEST = Path("InfiniteChoice/qa.json")

def deepest_string(x: str) -> str:
    """
    따옴표와 백슬래시로 여러 번 이스케이프된 문자열을
    더 이상 json.loads() 가 실패할 때까지 역직렬화.
    """
    while True:
        if not isinstance(x, str):
            break
        x_strip = x.strip()
        # 양쪽 따옴표가 있고, 내부에 최소 하나 \" 또는 \' 가 있으면 더 파보기
        if len(x_strip) >= 2 and x_strip[0] == '"' and x_strip[-1] == '"':
            inner = x_strip[1:-1]
        else:
            inner = x_strip
        try:
            x_next = json.loads(x_strip)
            if x_next == x or not isinstance(x_next, str):
                # 더 이상 변화 없거나 문자열이 아니면 끝
                return inner.strip()
            x = x_next
        except json.JSONDecodeError:
            # 더 파싱 불가 → 수동 치환 후 종료
            return re.sub(r'\\(["\'])', r'\1', inner).strip()

def clean_answer(raw):
    # 리스트면 첫 항목
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    # None → 빈 문자열
    if raw is None:
        return ""
    # 최종 문자열 정리
    return deepest_string(str(raw))

def transform(rec):
    return {
        "question": rec.get("input", rec.get("query", "")),
        "answer": clean_answer(rec.get("answer", ""))
    }

records = []
with SRC.open("r", encoding="utf-8") as f:
    for ln in f:
        if not ln.strip():
            continue
        records.append(transform(json.loads(ln)))

DEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✅ {len(records)} records saved → {DEST}")
