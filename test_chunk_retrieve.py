import json
from pathlib import Path
from typing import Dict, List, Set

# ── 경로 설정 ─────────────────────────────────────────────────────────────
HOTPOT_PATH          = Path("hotpotQA/sampled_1000.jsonl")          # ↩︎ 1 000개 샘플
CHUNK_STORE_PATH     = Path("hotpotQA/hotpot_kv_store_text_chunks.json")     # ↩︎ chunk‑ID → text
OUT_PATH             = Path("Result/hotpot_gold_chunk_ids.json")    # ↩︎ 결과 저장

# ── 유틸 함수 ────────────────────────────────────────────────────────────
def load_chunk_store() -> Dict[str, str]:
    """chunk‑ID ➜ text 로 이루어진 딕셔너리 반환"""
    with CHUNK_STORE_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    # 필요한 건 'content' 필드뿐이므로 가볍게 변환
    return {cid: obj["content"] for cid, obj in raw.items()}

def find_chunk_id(sentence: str, chunk_store: Dict[str, str]) -> str | None:
    """
    주어진 문장을 포함하는 첫 번째 chunk‑ID 반환.
    (중복 문장이 여러 청크에 들어 있을 수 있지만, 일반적으로 1개면 충분)
    """
    for cid, text in chunk_store.items():
        if sentence in text:
            return cid
    return None

def get_gold_chunk_ids(item: dict, chunk_store: Dict[str, str]) -> Set[str]:
    """
    Hotpot 항목 하나에 대해 supporting fact 문장이 들어 있는
    chunk‑ID 집합을 반환
    """
    ctx = {title: sents for title, sents in item["context"]}
    gold_chunks: Set[str] = set()

    for title, sent_idx in item["supporting_facts"]:
        sents: List[str] = ctx.get(title, [])
        if 0 <= sent_idx < len(sents):
            sentence = sents[sent_idx]
            cid = find_chunk_id(sentence, chunk_store)
            if cid:
                gold_chunks.add(cid)
    return gold_chunks

# ── 메인 로직 ────────────────────────────────────────────────────────────
def main() -> None:
    chunk_store = load_chunk_store()
    results = []

    with HOTPOT_PATH.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item["question"]
            gold_cids = list(get_gold_chunk_ids(item, chunk_store))
            results.append({
                "query": query,
                "gold_chunk_ids": gold_cids
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 매핑 완료: {len(results):,}개 쿼리 → {OUT_PATH}")

if __name__ == "__main__":
    main()
