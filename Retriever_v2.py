import os
import networkx as nx
from typing import List, Dict, Set

# your_module 경로에 맞춰 아래 두 줄 경로 수정
from edge_embedding import EdgeEmbedderFAISS
from edge_topic import extract_topics_subtopics

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
        self.subtopic_entity_edges = []

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
        found = False
        for nbr in self.graph.neighbors(sub_nid):
            if any(prefix in nbr for prefix in ["entity_", "ent_"]):
                ents.add(nbr)
                found = True
                if self.graph.has_edge(sub_nid, nbr):
                    edge_data = self.graph.get_edge_data(sub_nid, nbr)
                    self.subtopic_entity_edges.append((
                        sub_nid, nbr, {
                            "label": edge_data.get("label", ""),
                            "sentence": edge_data.get("sentence", ""),
                            "relation": "subtopic-entity"
                        }
                    ))
        if not found:
            print(f"⚠️ No entity neighbors for subtopic: {sub_nid} ({self.graph.nodes[sub_nid].get('label', '')})")
        return ents



    # def _entities_of_subtopic(self, sub_nid: str) -> Set[str]:
    #     ents = set()
    #     for nbr in self.graph.neighbors(sub_nid):
    #         if any(prefix in nbr for prefix in ["entity_", "ent_"]):
    #             ents.add(nbr)
    #     return ents

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
        self.seen_sub_nodes = valid_subs
        self.seen_entities = seen_entities

        # 7. 결과 반환
        return {
            # "entity_sentences": entity_sentences,
            "faiss_results": results
        }
    