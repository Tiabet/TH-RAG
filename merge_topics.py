# merge_low_degree_subtopics_only.py
import networkx as nx, re, itertools
from pathlib import Path

# ─────────────────── 설치 여부에 따라 rapidfuzz / difflib ───────────────────
try:
    from rapidfuzz import fuzz        # pip install rapidfuzz
    str_sim = lambda a, b: fuzz.ratio(a, b)          # 0‒100
except ImportError:
    from difflib import SequenceMatcher
    str_sim = lambda a, b: SequenceMatcher(None, a, b).ratio() * 100

# ────────────────────────────── 설정값 ─────────────────────────────────────────
GEXF_IN   = "hotpotQA/normalized_graph.gexf"
GEXF_OUT  = "hotpotQA/merged_graph.gexf"

DEG_THRESHOLD     = 5   # 저차수 subtopic 기준 (== 5 이면 사실상 topic+entity 두 엣지)
LEVENSHTEIN_TH    = 30     # 문자열 유사도(%) 최소값
JACCARD_TH        = 0.3    # 이웃 Jaccard 최소값

# ────────────────────────────── 유틸 함수 ─────────────────────────────────────
def normalize(text: str) -> str:
    """소문자·숫자·공백만 남김."""
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).strip()

def jaccard(set_a, set_b) -> float:
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

# ──────────────────────── 0) 그래프 로드 (MultiGraph) ─────────────────────────
G0 = nx.read_gexf(GEXF_IN)
G   = nx.MultiGraph(G0) if not isinstance(G0, (nx.MultiGraph, nx.MultiDiGraph)) else G0
edge_before = G.number_of_edges()

# ──────────────────────── 1) 저차수 subtopic 추출 ────────────────────────────
low_subs = [
    n for n, d in G.nodes(data=True)
    if d.get("type") == "subtopic" and G.degree(n) <= DEG_THRESHOLD
]
print(f"Low‑degree subtopics: {len(low_subs)}")

# ──────────────────────── 2) 병합 대상 찾기 (B→C) ────────────────────────────
mapping = {}                        # subtopic → target subtopic
labels  = {n: normalize(G.nodes[n].get("label", n)) for n in G.nodes}

# ——— B. 문자열(Levenshtein) ≥ threshold ———
for sub in low_subs:
    if sub in mapping:          # 이미 매핑된 경우 skip
        continue
    best, best_score = None, 0
    for cand in G.nodes:
        if cand == sub or G.nodes[cand].get("type") != "subtopic":
            continue
        score = str_sim(labels[sub], labels[cand])
        if score >= LEVENSHTEIN_TH and score > best_score:
            best, best_score = cand, score
    if best:
        mapping[sub] = best

# ——— C. 이웃 Jaccard ≥ threshold ———
neighbors = {
    n: {nbr for nbr in G.neighbors(n) if G.nodes[nbr].get("type") == "entity"}
    for n in G.nodes
}
for sub in low_subs:
    if sub in mapping:           # 이미 매핑됐으면 패스
        continue
    best, best_j = None, 0.0
    for cand in G.nodes:
        if cand == sub or G.nodes[cand].get("type") != "subtopic":
            continue
        j = jaccard(neighbors[sub], neighbors[cand])
        if j >= JACCARD_TH and j > best_j:
            best, best_j = cand, j
    if best:
        mapping[sub] = best

print(f"Subtopics to merge: {len(mapping)}")

# ──────────────────────── 3) 엣지 복사 + 노드 삭제 ────────────────────────────
for sub, tgt in mapping.items():
    if sub == tgt or not G.has_node(sub) or not G.has_node(tgt):
        continue

    # 3‑1) 모든 엣지를 target으로 복사 (MultiGraph → 중복 허용)
    for u, v, k, data in list(G.edges(sub, keys=True, data=True)):
        other = v if u == sub else u
        G.add_edge(tgt, other, **data)

    # 3‑2) 원본 subtopic 노드 제거
    G.remove_node(sub)

# ──────────────────────── 4) 무결성 검증 ───────────────────────────────────────
edge_after = G.number_of_edges()
assert edge_before == edge_after, f"Edge count changed! {edge_before} → {edge_after}"
print("Edge count preserved ✔")

# ──────────────────────── 5) 저장 ────────────────────────────────────────────
nx.write_gexf(G, GEXF_OUT)
print(f"Done! Merged graph saved to {GEXF_OUT}")
