import networkx as nx

# Load the GraphML file
path = '/graph.graphml'
try:
    G = nx.read_graphml(path)
except Exception as e:
    raise RuntimeError(f"Error reading GraphML: {e}")

# Basic properties
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
directed = G.is_directed()

# Identify isolates (nodes with degree 0)
isolates = list(nx.isolates(G))

# Identify self-loops
self_loops = list(nx.nodes_with_selfloops(G))

# Identify connected components
if directed:
    # consider weakly connected components
    components = list(nx.weakly_connected_components(G))
else:
    components = list(nx.connected_components(G))
num_components = len(components)
# sizes of components
component_sizes = sorted([len(c) for c in components], reverse=True)

# Node attributes
node_attrs = {}
for n, attrs in G.nodes(data=True):
    for key, val in attrs.items():
        node_attrs.setdefault(key, []).append(val)

# Count nodes missing certain attributes (if any attrs expected)
# Let's list all attribute keys
attr_keys = list(node_attrs.keys())

# Check attribute consistency: if attribute values have varying types or missing
attr_stats = {}
for key, vals in node_attrs.items():
    types = set(type(v).__name__ for v in vals)
    missing = sum(1 for v in vals if v is None or v == '')
    attr_stats[key] = {
        'count': len(vals),
        'unique_values': len(set(vals)),
        'types': types,
        'missing': missing
    }

# Edge attributes
edge_attrs = {}
for u, v, attrs in G.edges(data=True):
    for key, val in attrs.items():
        edge_attrs.setdefault(key, []).append(val)
edge_attr_keys = list(edge_attrs.keys())
edge_attr_stats = {}
for key, vals in edge_attrs.items():
    types = set(type(v).__name__ for v in vals)
    missing = sum(1 for v in vals if v is None or v == '')
    edge_attr_stats[key] = {
        'count': len(vals),
        'unique_values': len(set(vals)),
        'types': types,
        'missing': missing
    }

# Summarize anomalies
anomalies = {
    'num_nodes': num_nodes,
    'num_edges': num_edges,
    'directed': directed,
    'num_components': num_components,
    'component_sizes': component_sizes[:5],  # top 5
    'num_isolates': len(isolates),
    'isolates_sample': isolates[:5],
    'num_self_loops': len(self_loops),
    'self_loops_sample': self_loops[:5],
    'attr_keys': attr_keys,
    'node_attr_stats': attr_stats,
    'edge_attr_keys': edge_attr_keys,
    'edge_attr_stats': edge_attr_stats
}
print(anomalies)