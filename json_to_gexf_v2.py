import json
import networkx as nx

# Load JSON data
with open('graph.json', 'r', encoding='utf-8') as f:
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

# Create directed graph
G = nx.Graph()

for entry in entries:
    subj, pred, obj = entry['triple']
    subj_st = entry['subject']['subtopic']
    subj_mt = entry['subject']['main_topic']
    obj_st = entry['object']['subtopic']
    obj_mt = entry['object']['main_topic']

    # Define node IDs with prefixes to avoid naming conflicts
    subj_node = f"entity_{subj}"
    subj_st_node = f"subtopic_{subj_st}"
    subj_mt_node = f"topic_{subj_mt}"
    obj_node = f"entity_{obj}"
    obj_st_node = f"subtopic_{obj_st}"
    obj_mt_node = f"topic_{obj_mt}"

    # Add nodes with attributes
    G.add_node(subj_node, label=subj, type='entity')
    G.add_node(subj_st_node, label=subj_st, type='subtopic')
    G.add_node(subj_mt_node, label=subj_mt, type='topic')
    G.add_node(obj_node, label=obj, type='entity')
    G.add_node(obj_st_node, label=obj_st, type='subtopic')
    G.add_node(obj_mt_node, label=obj_mt, type='topic')

    # Add hierarchical edges
    G.add_edge(subj_node, subj_st_node, label='has_subtopic', relation_type='subtopic_relation')
    G.add_edge(subj_st_node, subj_mt_node, label='has_topic', relation_type='topic_relation')
    G.add_edge(obj_node, obj_st_node, label='has_subtopic', relation_type='subtopic_relation')
    G.add_edge(obj_st_node, obj_mt_node, label='has_topic', relation_type='topic_relation')

    # Add predicate edge between entities
    G.add_edge(subj_node, obj_node, label=pred, relation_type='predicate_relation')

# Save as GEXF for Gephi
nx.write_gexf(G, 'graph_v4.gexf')

print("Files saved:")
print(" - graph_v4.gexf")
