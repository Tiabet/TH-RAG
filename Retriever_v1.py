import os
import networkx as nx
from typing import List, Dict, Set
import openai
from dotenv import load_dotenv

# your_module 경로에 맞춰 아래 두 줄 경로 수정
from edge_embedding import EdgeEmbedderFAISS
from edge_topic import extract_topics_subtopics

# === 사전정의된 토픽 및 서브토픽 ===
# PREDEFINED_TOPICS_INFO = [
#     {"topic": "Beekeeping", "subtopics": ["Extraction", "Straining", "Processing", "Harvesting", "Filtering", "Storage", "Quality", "Nutritional", "Pests", "Equipment"]},
# ]
# === Retriever 클래스 정의 ===
class Retriever:
    def __init__(self,
                 gexf_path: str,
                 embedding_model: str,
                 openai_api_key: str,
                 index_path: str,
                 payload_path: str,
                 client=None):
        print(f"Loading graph from {gexf_path}")
        self.graph = nx.read_gexf(gexf_path)
        self.client = client

        print("Initializing EdgeEmbedderFAISS")
        self.embedder = EdgeEmbedderFAISS(
            gexf_path=gexf_path,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            index_path=index_path,
            payload_path=payload_path,
        )
        if os.path.exists(index_path):
            self.embedder.load_index()
            print("FAISS index & payloads loaded.")
        else:
            self.embedder.build_index()
            print("FAISS index & payloads built.")

        print(f"Total graph nodes: {len(self.graph.nodes)}")

    def _subtopic_is_linked_to_topic(self, sub_nid: str, topic_terms: Set[str]) -> bool:
        sub_label = self.graph.nodes[sub_nid].get("label", "").lower()
        sub_words = set(sub_label.split())

        for topic_label in topic_terms:
            topic_words = set(topic_label.lower().split())
            if sub_words & topic_words:  # 교집합이 존재하면 관련 있음
                return True
        return False


    def _entities_of_subtopic(self, sub_nid: str) -> Set[str]:
        ents = set()
        for nbr in self.graph.neighbors(sub_nid):
            if any(prefix in nbr for prefix in ["entity_", "ent_"]):
                ents.add(nbr)
        return ents

    def retrieve(self, query, top_n: int = 50) -> Dict[str, List[str]]:
        print("--- Retrieval Start ---")
        MAX_RETRIES = 10  # 무한 루프 방지
        attempt = 0
        total_sent_count = 0
        entity_sentences = {}  # 누적 저장
        seen_entities = set()
        seen_sub_nodes = set()

        while total_sent_count < 200 and attempt < MAX_RETRIES:
            print(f"\n[Attempt {attempt+1}]")
            # 1. 토픽/서브토픽
            topics_info = extract_topics_subtopics(query, self.client)
            # print(f"Topics & Subtopics: {topics_info}")
            for item in topics_info:
                item["topic"] = item["topic"].lower()
                item["subtopics"] = [sub.lower() for sub in item["subtopics"]]
            topic_terms = {t["topic"] for t in topics_info}
            subtopic_terms = {sub for t in topics_info for sub in t["subtopics"]}

            # 2. subtopic 노드 매핑
            sub_nodes = set()
            for sub in subtopic_terms:
                matched = [
                    nid for nid, data in self.graph.nodes(data=True)
                    if data.get("type") == "subtopic" and (sub in nid or sub in data.get("label", ""))
                ]
                if matched:
                    sub_nodes.update(matched)
                else:
                    print(f"Warning: No match for subtopic '{sub}'")
            print(f"Total subtopic nodes: {len(sub_nodes)}")

            new_sub_nodes = sub_nodes - seen_sub_nodes
            seen_sub_nodes.update(new_sub_nodes)

            # 3. 연결 검증
            valid_subs = [nid for nid in new_sub_nodes if self._subtopic_is_linked_to_topic(nid, topic_terms)]
            # print(f"New valid subtopic nodes: {valid_subs}")


            # 4. 엔티티 추출
            new_entities = set()
            for sub in valid_subs:
                ents = self._entities_of_subtopic(sub)
                new_entities.update(ents)
            new_entities -= seen_entities  # 중복 제거
            seen_entities.update(new_entities)

            if not new_entities:
                print("No new entities found. Retrying...\n")
                attempt += 1
                continue
            else:
                print(f"New entities: {len(new_entities)}")

            # 5. 엔티티 간선 문장 추출
            for ent in new_entities:
                sents = []
                for u, v, data in self.graph.edges(ent, data=True):
                    sentence_block = data.get("sentence", "")
                    if sentence_block:
                        parts = [s.strip() for s in sentence_block.split('/') if s.strip()]
                        sents.extend(parts)
                if sents:
                    entity_sentences[ent] = sents
                    total_sent_count += len(sents)

            print(f"Accumulated entity-edge sentences: {total_sent_count}")

            if total_sent_count >= 200:
                print("✅ Minimum sentence threshold reached.")
                break
            else:
                print("🔁 Continuing to collect more...\n")
                attempt += 1
        print(len(entity_sentences), "entities with sentences collected.")
        # 6. FAISS 검색 (필터링)
        results = self.embedder.search(
            query=query,
            top_k=top_n,
            filter_entities=seen_entities if seen_entities else None,
        )
        print(f"Retrieved {len(results)} edges from FAISS.")
        print("--- Retrieval End ---")

        # === 이 두 줄 추가 ===
        self.seen_sub_nodes = seen_sub_nodes
        self.seen_entities = seen_entities

        # 7. 결과 반환
        return {
            # "entity_sentences": entity_sentences,
            "faiss_results": results
        }


# # === 설정 부분만 수정하세요 ===
# GEXF_PATH = "hotpotQA/graph_v6.gexf"
# INDEX_PATH = "hotpotQA/edge_index_v6.faiss"
# PAYLOAD_PATH = "hotpotQA/edge_payloads_v6.npy"
# EMBEDDING_MODEL = "text-embedding-3-small"
# OUTPUT_GEXF = "hotpotQA/test/retrieved_subgraph.gexf"
# QUERY = "\"When a Killer Calls\" was released un 2006 to  coincide with the theatrical release of a remake directed by who?"  # <- 원하는 쿼리로 수정
# load_dotenv()
# # 환경변수 확인
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     print("Error: 환경 변수 OPENAI_API_KEY를 설정해야 합니다.")
#     exit(1)

# client = openai.OpenAI(api_key=OPENAI_API_KEY)
# # Retriever 인스턴스화 및 실행
# retriever = Retriever(
#     gexf_path=GEXF_PATH,
#     embedding_model=EMBEDDING_MODEL,
#     openai_api_key=OPENAI_API_KEY,
#     index_path=INDEX_PATH,
#     payload_path=PAYLOAD_PATH,
#     client = client
# )

# # retrieve 메서드 호출
# results = retriever.retrieve(query=QUERY, top_n=100000)

# # 결과 출력
# print("\n=== 상위 10개 검색 결과 ===")
# for hit in results["faiss_results"]:
#     print(f"[{hit['rank']}] ({hit['score']:.4f}) {hit['source']}→{hit['target']} “{hit['sentence']}”")