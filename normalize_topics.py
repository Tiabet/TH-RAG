#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_topics.py · topic / subtopic node cleanup
---------------------------------------------------
* composite split → label normalize (lemma + singular)
* filler stripping (role of…, impact of…)
* head‑noun 클러스터로 표준 subtopic ID 부여 (ID 정렬용)
* **head‑only subtopic 분해**
  · 연결된 각 topic 에 대해 `<topic> <head>` 서브토픽으로 모든 엣지 이관
  · entity‑subtopic 엣지도 동일하게 재배선
  · 이관이 하나도 성공하지 못한 head‑only 노드는 그대로 둠 (정보 손실 방지)
  · 모든 엣지를 옮기고 degree==0 이면 노드 제거
* edge 속성(label / sentence / weight) 안전히 합침

‣ I/O 경로는 하드코딩(스크립트 상단 상수)
"""

from __future__ import annotations
import re, csv, collections, sys
from pathlib import Path

import networkx as nx
import spacy, inflect

# ──────────────────────────────────────────
# 🛠️  경로 하드코딩 – 필요 시만 수정!
# ──────────────────────────────────────────
INPUT_GEXF  = Path("hotpotQA/graph_v1.gexf")
OUTPUT_GEXF = Path("hotpotQA/normalized_graph_v1.gexf")
SUB_CSV     = Path("hotpotQA/hotpot.csv")

# ──────────────────────────────────────────
# resources & 기본 설정
# ──────────────────────────────────────────
nlp        = spacy.load("en_core_web_sm", disable=["ner", "parser"])
inflector  = inflect.engine()
MIN_SIZE   = 1   # head‑noun 클러스터 최소 크기

FILLER_PAT = [
    r"^(type|kind|form|class|classification|list|number|count|history|study|future|status|role|impact|effect|origin|case study) of ",
    r" (in|for|at|within|by|about|among|between|under) .*$",
]

DELIM_RE = re.compile(r"\s*/\s*|\s+and\s+|\s*&\s*|\s*,\s*|\s*,\s*and\s+", re.I)

# ──────────────────────────────────────────
# helper 함수
# ──────────────────────────────────────────

def normalize_text(txt: str) -> str:
    txt = txt.lower().replace("_", " ")
    txt = re.sub(r"\s*\(.*?\)$", "", txt)
    txt = re.sub(r"[^a-z0-9\s\-]", " ", txt)
    txt = " ".join(txt.split())
    lemmas = [t.lemma_ for t in nlp(txt) if not t.is_space]
    singulars = [inflector.singular_noun(w) or w for w in lemmas]
    return " ".join(singulars).strip()

def strip_fillers(txt: str) -> str:
    for pat in FILLER_PAT:
        txt = re.sub(pat, "", txt)
    return txt.strip()

def head_noun(label: str) -> str:
    toks = label.split()
    for w in reversed(toks):
        if len(w) > 3 and w not in {"of", "for", "and", "the", "in"}:
            return w
    return toks[-1]

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def merge_edge_attrs(dst: dict, src: dict, sep: str = " / ") -> None:
    for k in ("label", "sentence"):
        if src.get(k):
            if not dst.get(k):
                dst[k] = src[k]
            elif src[k] not in dst[k]:
                dst[k] += sep + src[k]
    dst["weight"] = dst.get("weight", 1) + src.get("weight", 1)

# ──────────────────────────────────────────
# 1) head‑noun 클러스터 매핑 (ID 표준화용)
# ──────────────────────────────────────────

def build_cluster_map(sub_csv: Path) -> dict[str, str]:
    subs_raw = [r[0] for r in csv.reader(sub_csv.open(encoding="utf-8-sig")) if r]
    clusters: dict[str, list[str]] = collections.defaultdict(list)
    for s in subs_raw:
        clusters[head_noun(strip_fillers(normalize_text(s)))].append(s)
    mapping: dict[str, str] = {}
    for hn, items in clusters.items():
        canon = f"subtopic_{snake(hn)}"
        for it in items:
            mapping[normalize_text(it)] = canon
    return mapping

# ──────────────────────────────────────────
# 2) head‑only 서브토픽 분해
# ──────────────────────────────────────────

def split_head_only_subtopics(G: nx.Graph):
    """Break a head‑only subtopic (e.g. `director`) into `<topic> <head>` variants.
    Entity edges are migrated *only* when their original edge carries the same
    topic attribute, preventing duplicate edges to unrelated topics.  The
    original node is removed when all of its edges are re‑wired.
    """

    sub_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "subtopic"]
    norm_lbl  = {n: normalize_text(G.nodes[n]["label"]) for n in sub_nodes}
    sub_norm_set = set(norm_lbl.values())  # fast existence test

    for nid in sub_nodes:
        head = norm_lbl[nid]
        if " " in head:
            continue  # already has a prefix

        # neighbours
        topic_nodes   = [t for t in G[nid] if G.nodes[t].get("type") == "topic"]
        entity_nodes  = [e for e in G[nid] if G.nodes[e].get("type") == "entity"]
        if not topic_nodes or not entity_nodes:
            continue

        moved_any = False
        moved_entities : set[str] = set()  # track moved entities

        for t in topic_nodes:
            topic_norm = normalize_text(G.nodes[t]["label"])  # e.g. "film"
            cand_label = f"{topic_norm} {head}"
            if cand_label not in sub_norm_set:
                continue  # the `${topic} ${head}` subtopic does not exist
            cand_id = f"subtopic_{snake(cand_label)}"

            # topic ↔ candidate edge
            if not G.has_edge(cand_id, t):
                G.add_edge(cand_id, t, label="has_topic", relation_type="topic_relation")

            # entity ↔ candidate edges (topic attr check)
            for e in entity_nodes:
                edge_topic = normalize_text(G[nid][e].get("topic", ""))
                if edge_topic and edge_topic != topic_norm:
                    continue  # entity belongs to different topic
                if not G.has_edge(cand_id, e):
                    new_attrs = G[nid][e].copy()
                    new_attrs["topic"] = topic_norm
                    G.add_edge(cand_id, e, **new_attrs)
                else:
                    merge_edge_attrs(G[cand_id][e], G[nid][e])
                moved_entities.add(e)
            moved_any = True

        # delete original edges / node only if something was moved
        # 옮겨 놓은 엣지들만 삭제
        if moved_any:
            for e in moved_entities:
                if G.has_edge(nid, e):
                    G.remove_edge(nid, e)
            # topic 엣지는 모두 옮겼으므로 안전히 삭제
            for t in topic_nodes:
                if G.has_edge(nid, t):
                    G.remove_edge(nid, t)

            # 아직 남아 있는 엔티티‑엣지가 없으면 노드 제거
            if G.degree(nid) == 0:
                G.remove_node(nid)

# ──────────────────────────────────────────
# 그래프 정규화 메인
# ──────────────────────────────────────────

def normalize_graph(src: Path, dst: Path, sub_csv: Path):
    if not src.exists():
        sys.exit("❌ input not found: {}".format(src))
    if not src.exists():
        sys.exit(f"❌ input not found: {src}")
    print("📖 loading graph …")
    G = nx.read_gexf(src)

    # A) ID 표준화 매핑
    mapping = build_cluster_map(sub_csv)
    print(f"🗂️  cluster map size: {len(mapping):,}")

    # B) composite split
    for nid, data in list(G.nodes(data=True)):
        if data.get("type") == "entity":
            continue
        label = str(data.get("label", ""))
        if not DELIM_RE.search(label):
            continue
        parts = [p.strip() for p in DELIM_RE.split(label) if p.strip()]
        if len(parts) < 2:
            continue
        inc_edges = list(G.edges(nid, data=True))
        for part in parts:
            norm = normalize_text(part)
            if not norm:
                continue
            prefix = "topic" if data.get("type") == "topic" else "subtopic"
            new_id = f"{prefix}_{snake(norm)}"
            if new_id not in G:
                attrs = data.copy()
                attrs["label_raw"], attrs["label"] = part, norm
                G.add_node(new_id, **attrs)
            for u, v, edata in inc_edges:
                other = v if u == nid else u
                if not G.has_edge(new_id, other):
                    G.add_edge(new_id, other, **edata)
        G.remove_node(nid)

    # C) label 정규화 & 병합
    for nid in list(G.nodes):
        d = G.nodes[nid]
        ntype = d.get("type")
        if ntype not in {"topic", "subtopic"}:
            continue
        raw = str(d.get("label", "")) or str(nid)
        norm = normalize_text(raw)
        d["label_raw"], d["label"] = raw, norm
        target_id = f"{ntype}_{snake(norm)}"
        if ntype == "subtopic" and norm in mapping:
            target_id = mapping[norm]
        if target_id == nid:
            continue
        if target_id in G:
            # 속성 병합 + 엣지 재배선
            for nbr, edata in list(G[nid].items()):
                if nbr == target_id:
                    continue
                if G.has_edge(target_id, nbr):
                    merge_edge_attrs(G[target_id][nbr], edata)
                else:
                    G.add_edge(target_id, nbr, **edata)
            G.remove_node(nid)
        else:
            nx.relabel_nodes(G, {nid: target_id}, copy=False)

    # D) head‑only subtopic 분해
    split_head_only_subtopics(G)

    # E) edge id 재부여
    for i, (_, _, d) in enumerate(G.edges(data=True)):
        d["id"] = str(i)

    print("💾 writing →", dst)
    nx.write_gexf(G, dst)
    print("✅ done")

# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────
if __name__ == "__main__":
    normalize_graph(INPUT_GEXF, OUTPUT_GEXF, SUB_CSV)
