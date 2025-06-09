import json
import networkx as nx

def json_to_hierarchical_gml(json_path, gml_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    G = nx.DiGraph()
    node_id_map = {}
    current_id = 0

    def add_node(label):
        nonlocal current_id
        if label not in node_id_map:
            G.add_node(current_id, label=label)
            node_id_map[label] = current_id
            current_id += 1
        return node_id_map[label]

    for item in data:
        subj_label, pred_label, obj_label = item["triple"]

        subj_id = add_node(subj_label)
        obj_id = add_node(obj_label)

        G.add_edge(subj_id, obj_id, label=pred_label, sentence=item["sentence"])

        # Subject topic hierarchy
        subj_subtopic = item["subject"]["subtopic"]
        subj_main_topic = item["subject"]["main_topic"]
        subtopic_id = add_node(subj_subtopic)
        main_topic_id = add_node(subj_main_topic)
        G.add_edge(subj_id, subtopic_id, label="is_a_subtopic")
        G.add_edge(subtopic_id, main_topic_id, label="belongs_to_main_topic")

        # Object topic hierarchy
        obj_subtopic = item["object"]["subtopic"]
        obj_main_topic = item["object"]["main_topic"]
        subtopic_id = add_node(obj_subtopic)
        main_topic_id = add_node(obj_main_topic)
        G.add_edge(obj_id, subtopic_id, label="is_a_subtopic")
        G.add_edge(subtopic_id, main_topic_id, label="belongs_to_main_topic")

    nx.write_gml(G, gml_path)

# 🔽 여기서 직접 실행 시에만 동작하게!
if __name__ == "__main__":
    json_to_hierarchical_gml("graph.json", "graph.gml")
