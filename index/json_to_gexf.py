import json
import networkx as nx
import os
import html
import re
import argparse

def clean_id(text: str) -> str:
    """노드 ID용: 공백, 특수문자 제거 및 소문자화"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    return re.sub(r"[^\w\-\.]", "_", text)

def is_valid(item: dict) -> bool:
    """triplet 형식이 맞는지 검증"""
    return (
        isinstance(item, dict)
        and "triple" in item and isinstance(item["triple"], list) and len(item["triple"]) == 3
        and "subject" in item and "object" in item
    )

# ─────────────────────────────────────────────────────────────────────
# ✅ 메인 실행부: JSON → GEXF 변환
# ─────────────────────────────────────────────────────────────────────

def convert_json_to_gexf(input_file: str, output_file: str = None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []

    # Case A: 단일 dict 구조
    if isinstance(data, dict) and "triples" in data:
        entries.extend([item for item in data["triples"] if is_valid(item)])

    # Case B: 블록 리스트 구조
    elif isinstance(data, list):
        for block in data:
            if isinstance(block, dict) and "triples" in block:
                entries.extend([item for item in block["triples"] if is_valid(item)])

    print(f"✅ usable triples: {len(entries)}")
    if not entries:
        raise ValueError("No valid triples found—check JSON structure.")

    # 그래프 생성
    G = nx.Graph()

    for entry in entries:
        subj, pred, obj = entry['triple']
        subj, pred, obj = subj.lower(), pred.lower(), obj.lower()
        subj_st = entry['subject']['subtopic'].lower()
        subj_mt = entry['subject']['main_topic'].lower()
        obj_st = entry['object']['subtopic'].lower()
        obj_mt = entry['object']['main_topic'].lower()
        raw_sentence = entry.get('sentence', '')
        if isinstance(raw_sentence, list) and len(raw_sentence) > 0:
            sentence = raw_sentence[0].strip()
        elif isinstance(raw_sentence, str):
            sentence = raw_sentence.strip()
        else:
            sentence = ""

        # 노드 ID 생성
        subj_node     = f"entity_{clean_id(subj)}"
        subj_st_node  = f"subtopic_{clean_id(subj_st)}"
        subj_mt_node  = f"topic_{clean_id(subj_mt)}"
        obj_node      = f"entity_{clean_id(obj)}"
        obj_st_node   = f"subtopic_{clean_id(obj_st)}"
        obj_mt_node   = f"topic_{clean_id(obj_mt)}"

        # 노드 추가
        for node, label, typ in [
            (subj_node, subj, 'entity'), (subj_st_node, subj_st, 'subtopic'), (subj_mt_node, subj_mt, 'topic'),
            (obj_node, obj, 'entity'), (obj_st_node, obj_st, 'subtopic'), (obj_mt_node, obj_mt, 'topic')
        ]:
            safe_label = label
            G.add_node(node, label=safe_label.lower(), type=typ)

        # 계층 엣지
        G.add_edge(subj_node, subj_st_node, label='has_subtopic', relation_type='subtopic_relation', topic=subj_mt)
        G.add_edge(subj_st_node, subj_mt_node, label='has_topic', relation_type='topic_relation')
        G.add_edge(obj_node, obj_st_node, label='has_subtopic', relation_type='subtopic_relation', topic=obj_mt)
        G.add_edge(obj_st_node, obj_mt_node, label='has_topic', relation_type='topic_relation')

        # 문장 및 predicate 엣지
        safe_sentence = sentence
        safe_pred     = pred.lower()
        
        if G.has_edge(subj_node, obj_node):
            existing = G[subj_node][obj_node]
            if safe_pred and safe_pred not in existing['label']:
                existing['label'] += f" / {safe_pred}"
            if safe_sentence and safe_sentence not in existing['sentence']:
                existing['sentence'] += f" / {safe_sentence}"
            existing['weight'] = existing.get('weight', 1) + 1
        else:
            G.add_edge(
                subj_node,
                obj_node,
                label=safe_pred,
                relation_type='predicate_relation',
                sentence=safe_sentence,
                weight=1
            )

    # 디버그: 엣지 속성 검사
    print("=== Edge Attributes Check ===")
    for u, v, d in G.edges(data=True):
        for key, value in d.items():
            if isinstance(value, tuple):
                print(f"[TUPLE] Edge {u} - {v} has tuple in '{key}': {value}")
            elif not isinstance(value, (str, int, float)):
                print(f"[WARN] Edge {u} - {v} has non-serializable '{key}': {value} ({type(value)})")

    # 저장
    output_file = output_file or input_file.replace('.json', '.gexf')
    nx.write_gexf(G, output_file)
    print("📁 File saved:", output_file)


# ─────────────────────────────────────────────────────────────────────
# ✅ CLI에서 직접 실행할 경우만 작동
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    convert_json_to_gexf("UltraDomain/Mix/graph_v1.json")
