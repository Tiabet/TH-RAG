import networkx as nx
import pandas as pd

# GEXF 파일 읽기
G = nx.read_gexf("DB/graph_v7.gexf")

# 노드와 엣지를 DataFrame으로 변환
nodes = pd.DataFrame(G.nodes(data=True)).T.reset_index().rename(columns={"index": "id"})
edges = nx.to_pandas_edgelist(G)

# CSV로 저장
nodes.to_csv("DB/v7_nodes.csv", index=False)
edges.to_csv("DB/v7_edges.csv", index=False)