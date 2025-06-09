import json
import networkx as nx

def build_graph(json_path, gml_path):
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract items list, flattening if nested lists
    if isinstance(data, dict):
        # Prefer common keys or first list value
        if 'items' in data and isinstance(data['items'], list):
            raw = data['items']
        elif 'data' in data and isinstance(data['data'], list):
            raw = data['data']
        else:
            raw = next((v for v in data.values() if isinstance(v, list)), [])
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError('Expected JSON to be a list or dict containing a list.')

    # Flatten one level of nested lists
    items = []
    for element in raw:
        if isinstance(element, list):
            items.extend(element)
        else:
            items.append(element)

    # Build graph
    G = nx.MultiDiGraph()
    seen_hierarchy = set()

    for item in items:
        if not isinstance(item, dict):
            continue  # skip non-dict entries
        subj_sub = item['subject']['subtopic']
        subj_main = item['subject']['main_topic']
        obj_sub = item['object']['subtopic']
        obj_main = item['object']['main_topic']
        relation = item.get('sentence', '')

        # Add nodes
        G.add_node(subj_main, type='topic')
        G.add_node(subj_sub, type='subtopic')
        G.add_node(obj_main, type='topic')
        G.add_node(obj_sub, type='subtopic')

        # Hierarchical edges
        if (subj_sub, subj_main) not in seen_hierarchy:
            G.add_edge(subj_sub, subj_main, label='belongs_to', hierarchy=True)
            seen_hierarchy.add((subj_sub, subj_main))
        if (obj_sub, obj_main) not in seen_hierarchy:
            G.add_edge(obj_sub, obj_main, label='belongs_to', hierarchy=True)
            seen_hierarchy.add((obj_sub, obj_main))

        # Relation edge
        G.add_edge(subj_sub, obj_sub, label=relation, hierarchy=False)

    # Write GML
    nx.write_gml(G, gml_path)
    print(f"GML file saved to {gml_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python json_to_gml.py <input.json> <output.gml>')
        sys.exit(1)
    build_graph(sys.argv[1], sys.argv[2])
