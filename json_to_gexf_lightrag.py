import json
import networkx as nx

# Load the JSON data
with open('hotpotQA/light_graph_v1.json', 'r') as f:
    data = json.load(f)

# Flatten and filter entries with valid triples
entries = [
    item for sublist in data if isinstance(sublist, list)
    for item in sublist
    if isinstance(item, dict) and 'triple' in item and isinstance(item['triple'], list) and len(item['triple']) == 3
]

# Create a directed graph
G = nx.Graph()

for entry in entries:
    subj, pred, obj = entry['triple']
    subj_st = entry.get('subject', {}).get('subtopic')
    subj_mt = entry.get('subject', {}).get('main_topic')
    obj_st = entry.get('object', {}).get('subtopic')
    obj_mt = entry.get('object', {}).get('main_topic')
    
    # Add nodes with unique IDs
    G.add_node(f"entity_{subj}", label=subj, type='entity')
    if subj_st:
        G.add_node(f"subtopic_{subj_st}", label=subj_st, type='subtopic')
        G.add_edge(f"entity_{subj}", f"subtopic_{subj_st}", label='has_subtopic')
    if subj_mt:
        G.add_node(f"topic_{subj_mt}", label=subj_mt, type='topic')
        if subj_st:
            G.add_edge(f"subtopic_{subj_st}", f"topic_{subj_mt}", label='has_topic')
    
    G.add_node(f"entity_{obj}", label=obj, type='entity')
    if obj_st:
        G.add_node(f"subtopic_{obj_st}", label=obj_st, type='subtopic')
        G.add_edge(f"entity_{obj}", f"subtopic_{obj_st}", label='has_subtopic')
    if obj_mt:
        G.add_node(f"topic_{obj_mt}", label=obj_mt, type='topic')
        if obj_st:
            G.add_edge(f"subtopic_{obj_st}", f"topic_{obj_mt}", label='has_topic')
    
    # Add the original predicate edge between entity nodes
    G.add_edge(f"entity_{subj}", f"entity_{obj}", label=pred)

# Save the graph in GEXF format for Gephi
nx.write_gexf(G, 'graph_lightrag_v2.gexf')

print("GEXF file saved to graph_lightrag.gexf")
