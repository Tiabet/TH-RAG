import json
import networkx as nx

# Load JSON data
with open('DB/graph_v2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten and filter entries (triples of length 3)
entries = [
    item for sublist in data if isinstance(sublist, list)
    for item in sublist
    if isinstance(item, dict)
    and 'triple' in item
    and isinstance(item['triple'], list)
    and len(item['triple']) == 3
]

# Create undirected graph (or DiGraph if direction matters)
G = nx.Graph()

for entry in entries:
    subj, pred, obj = entry['triple']
    subj_st = entry['subject']['subtopic']
    subj_mt = entry['subject']['main_topic']
    obj_st = entry['object']['subtopic']
    obj_mt = entry['object']['main_topic']
    sentence = entry.get('sentence', '')

    # Define node IDs with prefixes
    subj_node = f"entity_{subj}"
    subj_st_node = f"subtopic_{subj_st}"
    subj_mt_node = f"topic_{subj_mt}"
    obj_node = f"entity_{obj}"
    obj_st_node = f"subtopic_{obj_st}"
    obj_mt_node = f"topic_{obj_mt}"

    # Add nodes with attributes
    for node, label, typ in [
        (subj_node, subj, 'entity'), (subj_st_node, subj_st, 'subtopic'), (subj_mt_node, subj_mt, 'topic'),
        (obj_node, obj, 'entity'), (obj_st_node, obj_st, 'subtopic'), (obj_mt_node, obj_mt, 'topic')
    ]:
        G.add_node(node, label=label, type=typ)

    # Hierarchical edges
    G.add_edge(subj_node, subj_st_node, label='has_subtopic', relation_type='subtopic_relation')
    G.add_edge(subj_st_node, subj_mt_node, label='has_topic', relation_type='topic_relation')
    G.add_edge(obj_node, obj_st_node, label='has_subtopic', relation_type='subtopic_relation')
    G.add_edge(obj_st_node, obj_mt_node, label='has_topic', relation_type='topic_relation')

    # Predicate edge with merging
    if G.has_edge(subj_node, obj_node):
        existing = G[subj_node][obj_node]
        existing['label'] = f"{existing['label']} / {pred}" if pred not in existing['label'] else existing['label']
        existing['sentence'] = f"{existing['sentence']} / {sentence}" if sentence and sentence not in existing['sentence'] else existing['sentence']
        existing['weight'] = existing.get('weight', 1) + 1
    else:
        G.add_edge(subj_node, obj_node,
                   label=pred,
                   relation_type='predicate_relation',
                   sentence=sentence,
                   weight=1)

# Save as GEXF for Gephi
nx.write_gexf(G, 'DB/graph_v7.gexf')

print("File saved:")
print(" - DB/graph_v7.gexf")
