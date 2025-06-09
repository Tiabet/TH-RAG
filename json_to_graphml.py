import json
import networkx as nx
import sys

def load_json_items(json_path):
    """
    JSON array 또는 JSON Lines 형식 파일을 읽고, items 리스트로 반환합니다.
    배열 안에 중첩된 리스트가 있으면 한 단계 flatten 처리합니다.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            items = []
            for el in data:
                if isinstance(el, list):
                    items.extend(el)
                else:
                    items.append(el)
            return items
        else:
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        items = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items


def build_graph(json_path, graphml_path, directed=False):
    # JSON 로드
    items = load_json_items(json_path)

    # directed 여부에 따른 그래프 생성
    G = nx.MultiDiGraph() if directed else nx.MultiGraph()
    seen_hierarchy = set()
    edge_id = 0  # GraphML용 고유 edge id 카운터

    for item in items:
        if not isinstance(item, dict):
            continue
        subj = item.get('subject', {}) or {}
        obj  = item.get('object', {})  or {}
        subj_sub  = subj.get('subtopic')
        subj_main = subj.get('main_topic')
        obj_sub   = obj.get('subtopic')
        obj_main  = obj.get('main_topic')
        rel_label = item.get('sentence', '')

        # 노드 추가
        if subj_main:  G.add_node(subj_main,  type='topic')
        if subj_sub:   G.add_node(subj_sub,   type='subtopic')
        if obj_main:   G.add_node(obj_main,   type='topic')
        if obj_sub:    G.add_node(obj_sub,    type='subtopic')

        # 계층적 엣지 (belongs_to)
        if subj_sub and subj_main and (subj_sub, subj_main) not in seen_hierarchy:
            edge_id += 1
            G.add_edge(subj_sub, subj_main,
                       id=str(edge_id), label='belongs_to', hierarchy=True)
            seen_hierarchy.add((subj_sub, subj_main))
        if obj_sub and obj_main and (obj_sub, obj_main) not in seen_hierarchy:
            edge_id += 1
            G.add_edge(obj_sub, obj_main,
                       id=str(edge_id), label='belongs_to', hierarchy=True)
            seen_hierarchy.add((obj_sub, obj_main))

        # 문장 기반 관계 엣지
        if subj_sub and obj_sub:
            edge_id += 1
            G.add_edge(subj_sub, obj_sub,
                       id=str(edge_id), label=rel_label, hierarchy=False)

    # GraphML로 저장 (각 edge에 id 속성 포함)
    nx.write_graphml(G, graphml_path)
    default = 'directed' if directed else 'undirected'
    print(f"Saved GraphML to '{graphml_path}' (edgedefault={default})")


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python json_to_graphml.py <input.json|jsonl> <output.graphml> [--directed]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]
    directed    = (len(sys.argv) == 4 and sys.argv[3] == '--directed')

    build_graph(input_path, output_path, directed=directed)
