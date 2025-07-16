# import json
# from pathlib import Path
# from typing import List, Dict, Any
# import tiktoken
# from tqdm import tqdm
# from collections.abc import Iterable

# # ───── 사용자 설정 ─────
# TXT_PATH   = "MultihopRAG/contexts.txt"
# GRAPH_PATH = "MultihopRAG/graph_v1.json"
# OUTPUT_PATH = "MultihopRAG/graph_with_chunks.json"

# MODEL_NAME = "gpt-4o-mini"
# MAX_TOKENS = 1200
# OVERLAP    = 100
# # ──────────────────────

# def chunk_text(text: str, max_tokens: int, overlap: int, model: str) -> List[str]:
#     enc = tiktoken.encoding_for_model(model)
#     tokens = enc.encode(text)
#     chunks, start = [], 0
#     while start < len(tokens):
#         chunk_tokens = tokens[start : start + max_tokens]
#         chunks.append(enc.decode(chunk_tokens))
#         start += max_tokens - overlap
#     return chunks

# def normalize(s: str) -> str:
#     return " ".join(s.replace("\n", " ").split())

# def find_sentence_chunks(sentence: str, chunks: List[str]) -> List[int]:
#     target = normalize(sentence)
#     return [i for i, ch in enumerate(chunks) if target in normalize(ch)]

# def flatten_until_dict(seq: Iterable) -> List[Dict[str, Any]]:
#     """리스트 중첩을 dict 레벨까지 전부 평탄화"""
#     flat: List[Dict[str, Any]] = []
#     stack = list(seq)
#     while stack:
#         cur = stack.pop()
#         if isinstance(cur, dict):
#             flat.append(cur)
#         elif isinstance(cur, list):
#             stack.extend(cur)        # 더 풀어야 함 → 스택에 추가
#         else:
#             # dict도 list도 아니면 무시(원하지 않는 타입)
#             pass
#     return flat[::-1]                # 원래 순서 유지

# # 1) 텍스트 → 청크
# full_text = Path(TXT_PATH).read_text(encoding="utf-8")
# chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)
# print(f"✅ 청크 {len(chunks)}개 생성됨")

# # 2) 그래프 로드 및 평탄화
# with open(GRAPH_PATH, encoding="utf-8") as f:
#     raw_graph = json.load(f)

# graph = flatten_until_dict(raw_graph)
# print(f"✅ 평탄화 후 그래프 항목 {len(graph)}개")

# # 3) 문장 ↔ 청크 매칭
# for item in tqdm(graph, desc="📌 매칭 중"):
#     sent = item.get("sentence", "")
#     item["chunk_ids"] = find_sentence_chunks(sent, chunks)
#     if not item["chunk_ids"]:
#         tqdm.write(f"⚠️ 매칭 실패: {sent[:70]}...")

# # 4) 저장
# with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#     json.dump(graph, f, indent=2, ensure_ascii=False)
# print(f"💾 저장 완료 → {OUTPUT_PATH}")

import json

# 예시: JSON 데이터를 문자열로 불러온 경우 (실제로는 파일에서 불러올 수도 있음)
with open('MultihopRAG\graph_v1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# chunk_id 추가
for chunk_id, chunk in enumerate(data):
    for item in chunk:
        item["chunk_id"] = chunk_id

# 결과 저장 (선택사항)
with open('dataset_with_chunk_id.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# 확인 출력 (선택사항)
print(json.dumps(data[:1], indent=2, ensure_ascii=False))  # 첫 번째 청크만 확인
