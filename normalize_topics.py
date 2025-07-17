#!/usr/bin/env python3
"""
Graph preprocessing – topics & subtopics only
-------------------------------------------
* **Split composite labels** into atomic subtopics.
* **Normalise labels** (lower‑case, lemmatised, singular).
* **Relabel IDs _only for topic/subtopic_ nodes** so that
    topic_sports  ➜  topic_sport
    subtopic_sports teams  ➜  subtopic_sport_team
* **Merge duplicates**: if the target ID already exists, the current node is deleted and all its incident edges are rewired to the target.
* Entity nodes are left untouched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import inflect
import networkx as nx
import spacy

# ──────────────────────────────────────────
# Resources
# ──────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
inflector = inflect.engine()

# I/O paths
src = Path("hotpotQA/graph_v1.gexf")
dst = Path("hotpotQA/graph_v1_processed.gexf")

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lower‑case, strip brackets/punct, lemmatise, singularise."""
    text = text.lower()
    text = re.sub(r"\s*\(.*?\)$", "", text)          # trailing parentheses
    text = re.sub(r"^(ex|pre|non)-", "", text)          # prefixes
    text = re.sub(r"-(like|type|based)$", "", text)    # suffixes
    text = re.sub(r"[^a-z0-9\s]", "", text)           # non‑alnum
    text = " ".join(text.split())                        # collapse ws
    doc = nlp(text)
    lemmas = [t.lemma_ for t in doc if not t.is_space]
    singulars = [inflector.singular_noun(w) or w for w in lemmas]
    return " ".join(singulars).strip()

def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

DELIM_REGEX = re.compile(
    r"\s*/\s*|\s+and\s+|\s*&\s*|\s*,\s*|\s*,\s*and\s+",
    re.I
)

# ──────────────────────────────────────────
# 1) Split composite labels
# ──────────────────────────────────────────

def split_composites(G: nx.Graph) -> None:
    """Break composite topic/subtopic labels (A & B) into separate nodes."""
    for nid, data in list(G.nodes(data=True)):
        if data.get("type") == "entity":
            continue
        label = str(data.get("label", ""))
        if not DELIM_REGEX.search(label):
            continue
        parts = [p.strip() for p in DELIM_REGEX.split(label) if p.strip()]
        if len(parts) < 2:
            continue
        incident = list(G.edges(nid, data=True))
        for part in parts:
            norm = normalize_text(part)
            if not norm:
                continue
            prefix = "topic" if data.get("type") == "topic" else "subtopic"
            new_id = f"{prefix}_{snake(norm)}"
            if new_id not in G:
                attrs = data.copy()
                attrs["label_raw"] = part
                attrs["label"] = norm
                G.add_node(new_id, **attrs)
            for u, v, edata in incident:
                nbr = v if u == nid else u
                if not G.has_edge(new_id, nbr):
                    G.add_edge(new_id, nbr, **edata)
        G.remove_node(nid)

# ──────────────────────────────────────────
# 2) Normalise labels + merge duplicates with ID rewiring
# ──────────────────────────────────────────

def normalise_and_merge(G: nx.Graph) -> None:
    """Operate only on topic/subtopic nodes, relabel IDs & merge duplicates."""
    for nid in list(G.nodes):
        data = G.nodes[nid]
        ntype = data.get("type")
        if ntype not in {"topic", "subtopic"}:
            continue

        raw_label = str(data.get("label", "")) or str(nid)
        norm = normalize_text(raw_label)
        data["label_raw"] = raw_label
        data["label"] = norm

        target_id = f"{ntype}_{snake(norm)}"
        if target_id == nid:
            continue  # already standardised

        if target_id in G:
            # merge: redirect edges then drop current node
            for nbr, edata in list(G[nid].items()):
                if nbr == target_id:
                    continue  # avoid self‑loop
                if not G.has_edge(target_id, nbr):
                    G.add_edge(target_id, nbr, **edata)
            G.remove_node(nid)
        else:
            # simple relabel
            nx.relabel_nodes(G, {nid: target_id}, copy=False)

# ──────────────────────────────────────────
# 3) Re‑index edge IDs
# ──────────────────────────────────────────

def reset_edge_ids(G: nx.Graph) -> None:
    for i, (_, _, data) in enumerate(G.edges(data=True)):
        data["id"] = str(i)

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

if not src.exists():
    print(f"❌  input file not found: {src}")
    sys.exit(1)

print(f"📖  loading graph: {src}")
G = nx.read_gexf(src)

split_composites(G)
normalise_and_merge(G)
reset_edge_ids(G)

print(f"💾  writing graph → {dst}")
nx.write_gexf(G, dst)
print("✅  done.")
