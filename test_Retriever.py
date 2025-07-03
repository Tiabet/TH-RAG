# run_retrieve_and_visualize.py

import os
import matplotlib.pyplot as plt
import networkx as nx
import openai
from dotenv import load_dotenv
from Retriever_v2 import Retriever  # 너가 작성한 retrieve.py에서 import

# === 사용자 설정 ===
GEXF_PATH = "hotpotQA/graph_v1.gexf"
INDEX_PATH = "hotpotQA/edge_index_v1.faiss"
PAYLOAD_PATH = "hotpotQA/edge_payloads_v1.npy"
EMBEDDING_MODEL = "text-embedding-3-small"
OUTPUT_GEXF = "hotpotQA/test/retrieved_subgraph_v1.gexf"
QUERY = "Which American comedian born on March 21, 1962, appeared in the movie \"Sleepless in Seattle?\""

# 환경 변수에서 OpenAI API 키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("환경 변수 'OPENAI_API_KEY'가 설정되지 않았습니다.")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
# === 리트리버 초기화 ===
retriever = Retriever(
    gexf_path=GEXF_PATH,
    embedding_model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    index_path=INDEX_PATH,
    payload_path=PAYLOAD_PATH,
    client=client
)

# === 리트리브 실행 ===
results = retriever.retrieve(query=QUERY, top_n=10000)  # top_n은 조정 가능

# === 노드/엣지 추출 ===
selected_nodes = set()
selected_edges = []

for hit in results["faiss_results"]:
    source = hit["source"]
    target = hit["target"]
    sentence = hit["sentence"]
    label = hit.get("label", "")

    # 노드 추가
    selected_nodes.update([source, target])

    # FAISS 결과 기반 엣지 구성
    selected_edges.append((source, target, {
        "label": label,
        "sentence": sentence,
        "faiss_score": hit["score"],
        "rank": hit["rank"]
    }))


# === ✅ 추출된 subtopic 노드도 포함시킴 ===
selected_nodes.update(retriever.seen_sub_nodes)
# 기존 selected_edges 이후에 추가
selected_edges.extend(retriever.subtopic_entity_edges)


# === 3. 서브그래프 생성 ===
subgraph = nx.Graph()
subgraph.add_nodes_from(selected_nodes)
subgraph.add_edges_from(selected_edges)

# === 5. 저장 ===
print(f"✅ 노드 수: {len(subgraph.nodes)} / 엣지 수: {len(subgraph.edges)}")
nx.write_gexf(subgraph, OUTPUT_GEXF)
print(f"📁 저장 완료: {OUTPUT_GEXF}")