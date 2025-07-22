#!/usr/bin/env python
# sample_200_from_1000_jsonl.py
import json, random
from pathlib import Path

SRC_PATH  = Path("hotpotQA/sampled_1000.jsonl")     # 원본 1,000개
DEST_PATH = Path("hotpotQA/sampled_200.jsonl")      # 출력 200개
NUM_SAMPLES = 200
SEED = 42               # 재현 가능하도록 고정(원하면 None)

def main():
    # 1) 원본 읽기
    with SRC_PATH.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"🔢 loaded {len(data):,} items")

    # 2) 샘플링
    random.seed(SEED)
    sampled = random.sample(data, min(NUM_SAMPLES, len(data)))
    print(f"🎲 sampled {len(sampled)} items")

    # 3) 저장 (jsonl)
    with DEST_PATH.open("w", encoding="utf-8") as f:
        for obj in sampled:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ saved → {DEST_PATH}")

if __name__ == "__main__":
    main()
