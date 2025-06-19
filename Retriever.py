import os
import networkx as nx
import openai
from typing import List, Dict, Set
from edge_topic import extract_topics_subtopics
from edge_embedding import EdgeEmbedderFAISS

class Retriever:
    def __init__(self,
                 gexf_path: str,
                 embedding_model: str,
                 openai_api_key: str,
                 index_path: str,
                 payload_path: str):
        print(f"Loading graph from {gexf_path}")
        self.graph = nx.read_gexf(gexf_path)

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

        self.client = openai.OpenAI(api_key=openai_api_key) 

        print(f"Total graph nodes: {len(self.graph.nodes)}")

    def _subtopic_is_linked_to_topic(self, sub_nid: str, topic_terms: Set[str]) -> bool:
        for nbr in self.graph.neighbors(sub_nid):
            lbl = self.graph.nodes[nbr].get("label", "")
            if any(topic in nbr or topic in lbl for topic in topic_terms):
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

        # 1. 토픽/서브토픽
        topics_info = extract_topics_subtopics(query, self.client)
        print(f"Topics & Subtopics: {topics_info}")
        topic_terms = {t["topic"] for t in topics_info}
        subtopic_terms = {sub for t in topics_info for sub in t["subtopics"]}

        # 2. subtopic 노드 매핑
        sub_nodes = set()
        for sub in subtopic_terms:
            matched = [nid for nid, data in self.graph.nodes(data=True)
                       if sub in nid or sub in data.get("label", "")]
            if matched:
                # print(f"Matched subtopic '{sub}' -> {matched}")
                sub_nodes.update(matched)
            else:
                print(f"Warning: No match for subtopic '{sub}'")
        print(f"Total subtopic nodes: {len(sub_nodes)}")

        # 3. 연결 검증
        valid_subs, invalid_subs = [], []
        for nid in sub_nodes:
            if self._subtopic_is_linked_to_topic(nid, topic_terms):
                valid_subs.append(nid)
            else:
                invalid_subs.append(nid)
        if invalid_subs:
            print(f"Ignored subtopics (no topic link): {invalid_subs}")
        print(f"Valid subtopic nodes: {valid_subs}")

        # 4. 엔티티 추출
        entities = set()
        for sub in valid_subs:
            ents = self._entities_of_subtopic(sub)
            # print(f"Entities for '{sub}': {ents}")
            entities.update(ents)
        if not entities:
            print("Warning: No entities found; unfiltered search will be used.")
            entities = None
        else:
            print(f"Collected entities: {entities}")

        # 5. 엔티티 간선 문장 추출
        entity_sentences = {}
        for ent in entities or []:
            sents = []
            for u, v, data in self.graph.edges(ent, data=True):
                sentence_block = data.get("sentence", "")
                if sentence_block:
                    parts = [s.strip() for s in sentence_block.split('/') if s.strip()]
                    sents.extend(parts)
            if sents:
                entity_sentences[ent] = sents
                # print(f"Entity '{ent}' has {len(sents)} sentence(s)")
        if entity_sentences:
            print(f"{len(entity_sentences)}Entity-edge sentences collected.")
        else:
            print("Warning: No sentences found on entity edges.")

        # 6. FAISS 검색 (필터링)
        results = self.embedder.search(
            query=query,
            top_k=top_n,
            filter_entities=entities,
        )
        print(f"Retrieved {len(results)} edges from FAISS.")
        print("--- Retrieval End ---")

        # 7. 결과 반환
        return {
            "entity_sentences": entity_sentences,
            "faiss_results": results
        }