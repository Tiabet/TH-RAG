#!/usr/bin/env python3
"""Split subtopic nodes whose *label* contains a slash ("/") and **merge** them
into pre‑existing subtopic nodes when possible.

✔️  If a clean part already exists as a subtopic (same label, type="subtopic"),
    we **reuse** that node and just copy edges.
✔️  Otherwise, we create a new node with id pattern ``subtopic_<clean>``.

For example, a node
    id="subtopic_comedian/actress"  label="comedian/actress"
will be removed and its edges duplicated to:
    id="subtopic_comedian"   (existing or new)  label="comedian"
    id="subtopic_actress"    (existing or new)  label="actress"

All incident edges of the original node are duplicated so connectivity is
preserved.  Attribute merging for duplicate edges is *not* attempted; if the
edge already exists, we skip adding a second copy.

Usage
-----
$ python split_subtopic_slash.py hotpotQA/graph_v1.gexf hotpotQA/graph_v1_processed.gexf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import networkx as nx

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_label_index(graph: nx.Graph) -> Dict[str, str]:
    """Return mapping *label → node-id* for existing subtopic nodes."""
    return {
        data.get("label", ""): nid
        for nid, data in graph.nodes(data=True)
        if data.get("type") == "subtopic"
    }


def split_slash_subtopics(G: nx.Graph) -> None:
    """In‑place transformation as described above."""
    label_to_nid = build_label_index(G)

    # Identify slash‑label subtopic nodes first
    candidates = [
        (nid, data["label"])  # type: ignore[index]
        for nid, data in G.nodes(data=True)
        if data.get("type") == "subtopic" and "/" in str(data.get("label", ""))
    ]

    for orig_nid, orig_label in candidates:
        parts = [p.strip() for p in orig_label.split("/") if p.strip()]
        if len(parts) < 2:
            continue  # nothing to do

        # Snapshot of incident edges (u,v,data)
        incident = list(G.edges(orig_nid, data=True))

        for part in parts:
            clean = part.replace(" ", "_")

            # 1) Find or create destination node
            if part in label_to_nid:
                dest_nid = label_to_nid[part]
            else:
                dest_nid = f"subtopic_{clean}"
                if dest_nid not in G:
                    attrs = G.nodes[orig_nid].copy()
                    attrs["label"] = part
                    G.add_node(dest_nid, **attrs)
                label_to_nid[part] = dest_nid  # update index

            # 2) Duplicate edges (avoid duplicates)
            for u, v, edata in incident:
                nbr = v if u == orig_nid else u
                if not G.has_edge(dest_nid, nbr):
                    G.add_edge(dest_nid, nbr, **edata)

        # 3) Remove original slash node
        G.remove_node(orig_nid)

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: split_subtopic_slash.py input.gexf output.gexf")
        sys.exit(1)

    in_path, out_path = map(Path, sys.argv[1:])
    if not in_path.exists():
        print(f"❌ Input file not found: {in_path}")
        sys.exit(1)

    print(f"📖  Loading graph: {in_path}")
    G = nx.read_gexf(in_path)

    split_slash_subtopics(G)

    print(f"💾  Writing modified graph → {out_path}")
    nx.write_gexf(G, out_path)
    print("✅  Done.")
