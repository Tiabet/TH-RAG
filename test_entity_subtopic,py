import networkx as nx

# GEXF 파일 불러오기
G = nx.read_gexf("hotpotQA/graph_v2.gexf")

# 엔티티 노드 중에서 연결된 서브토픽이 2개 이상인 것 찾기
result = []

for node, data in G.nodes(data=True):
    if data.get('type') == 'entity':
        connected_nodes = G.neighbors(node)
        subtopic_neighbors = [
            n for n in connected_nodes
            if G.nodes[n].get('type') == 'subtopic'
        ]
        if len(subtopic_neighbors) >= 2:
            result.append({
                'entity': node,
                'subtopics': subtopic_neighbors
            })

# 결과 출력
for item in result:
    print(f"Entity: {item['entity']}")
    print(f"Connected Subtopics: {item['subtopics']}")
    print("-----")
