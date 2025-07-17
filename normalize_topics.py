#!/usr/bin/env python3
from __future__ import annotations

import re
import string
import sys
from pathlib import Path
from typing import Dict
import spacy
import inflect

import networkx as nx

# ──────────────────────────────────────────
# 텍스트 정규화
# ──────────────────────────────────────────
# 필요한 리소스 로드
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
inflector = inflect.engine()

def normalize_text(text: str) -> str:
    text = text.lower()

    # ① 괄호 후치 제거 → e.g., "battery (chemistry)" → "battery"
    text = re.sub(r"\s*\(.*?\)$", "", text)

    # ② 하이픈 접두사/접미사 정리 → e.g., "non-smoker" → "smoker"
    text = re.sub(r"^(ex|pre|non)-", "", text)
    text = re.sub(r"-(like|type|based)$", "", text)

    # ③ 특수문자 제거 → e.g., "battery!" → "battery"
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # ④ 공백 정리
    text = " ".join(text.split())

    # ⑤ spaCy 기반 어근화 (lemmatization)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_space]

    # ⑥ 복수형 처리 → e.g., "companies" → "company"
    singulars = [inflector.singular_noun(w) or w for w in lemmas]

    # ⑦ 최종 정리
    return " ".join(singulars).strip()

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# ──────────────────────────────────────────
# 구분자 패턴
# ──────────────────────────────────────────
DELIM_REGEX = re.compile(
    r"\s*/\s*"          # slash
    r"|\s+and\s+"       # and
    r"|\s*&\s*"         # &
    r"|\s*,\s*"         # comma
    r"|\s*,\s*and\s+",  # , and
    flags=re.I,
)

# ──────────────────────────────────────────
# 보조 인덱스
# ──────────────────────────────────────────
def build_norm_label_index(G: nx.Graph) -> Dict[str, str]:
    """정규화 라벨 → 노드 ID"""
    idx: Dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        if data.get("type") == "entity":
            continue
        norm = normalize_text(data.get("label", ""))
        if norm and norm not in idx:
            idx[norm] = nid
    return idx

# ──────────────────────────────────────────
# 메인 변환
# ──────────────────────────────────────────
def normalize_composite_subtopics(G: nx.Graph) -> None:
    label2nid = build_norm_label_index(G)

    # 대상 노드 수집
    targets = [
        (nid, data["label"])
        for nid, data in G.nodes(data=True)
        if data.get("type") != "entity" # 엔티티 제외
        and DELIM_REGEX.search(str(data.get("label", "")))
    ]

    for orig_nid, raw_label in targets:
        parts = [p.strip() for p in DELIM_REGEX.split(raw_label) if p.strip()]
        if len(parts) < 2:
            continue

        incident_edges = list(G.edges(orig_nid, data=True))

        for part in parts:
            norm = normalize_text(part)
            if not norm:
                continue

            # 1) 기존 노드 재사용 or 신규 생성
            if norm in label2nid:
                dest = label2nid[norm]
            else:
                dest = f"subtopic_{snake(norm)}"
                if dest not in G:
                    attrs = G.nodes[orig_nid].copy()
                    attrs["label"] = part          # 원본 그대로 표시
                    G.add_node(dest, **attrs)
                label2nid[norm] = dest

            # 2) 엣지 복사 (중복 방지)
            for u, v, edata in incident_edges:
                nbr = v if u == orig_nid else u
                if not G.has_edge(dest, nbr):
                    G.add_edge(dest, nbr, **edata)

        # 3) 복합 노드 삭제
        G.remove_node(orig_nid)

        for i, (u, v, data) in enumerate(G.edges(data=True)):
            data['id'] = str(i)

# input/output 경로 직접 지정 (또는 sys.argv 사용 유지 가능)
src = Path("hotpotQA/graph_v1.gexf")
dst = Path("hotpotQA/graph_v1_processed.gexf")

if not src.exists():
    print(f"❌  input file not found: {src}")
    sys.exit(1)

print(f"📖  loading graph: {src}")
graph = nx.read_gexf(src)

normalize_composite_subtopics(graph)

print(f"💾  writing graph → {dst}")
nx.write_gexf(graph, dst)
print("✅  done.")