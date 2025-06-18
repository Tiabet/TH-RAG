import json
import networkx as nx
import pandas as pd

# Load JSON data
with open('DB/graph.json', 'r') as f:
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
G = nx.DiGraph()

for entry in entries:
    subj, pred, obj = entry['triple']
    subj_st, subj_mt = entry['subject']['subtopic'], entry['subject']['main_topic']
    obj_st, obj_mt = entry['object']['subtopic'], entry['object']['main_topic']
    
    # Add nodes with types
    G.add_node(subj, label=subj, type='entity')
    G.add_node(subj_st, label=subj_st, type='subtopic')
    G.add_node(subj_mt, label=subj_mt, type='topic')
    G.add_node(obj, label=obj, type='entity')
    G.add_node(obj_st, label=obj_st, type='subtopic')
    G.add_node(obj_mt, label=obj_mt, type='topic')
    
    # Add edges: entity -> subtopic -> topic, and the predicate edge
    G.add_edge(subj, subj_st, label='has_subtopic', relation_type='subtopic_relation')
    G.add_edge(subj_st, subj_mt, label='has_topic', relation_type='topic_relation')
    G.add_edge(obj, obj_st, label='has_subtopic', relation_type='subtopic_relation')
    G.add_edge(obj_st, obj_mt, label='has_topic', relation_type='topic_relation')
    G.add_edge(subj, obj, label=pred, relation_type='predicate_relation')

# Save as GEXF for Gephi
nx.write_gexf(G, 'graph_v2.gexf')

# # Also save CSVs for optional import
# nodes = [{'id': n, 'label': attr.get('label', ''), 'type': attr.get('type', '')} for n, attr in G.nodes(data=True)]
# edges = [{'source': u, 'target': v, 'label': attr.get('label', '')} for u, v, attr in G.edges(data=True)]

# pd.DataFrame(nodes).to_csv('/mnt/data/nodes_extended.csv', index=False)
# pd.DataFrame(edges).to_csv('/mnt/data/edges_extended.csv', index=False)

print("Files saved:")
print(" - graph_v2.gexf")
# print(" - nodes_extended.csv")
# print(" - edges_extended.csv")
