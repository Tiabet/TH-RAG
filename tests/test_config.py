from config import THRAGConfig


def test_dataset_paths_are_consistent() -> None:
    config = THRAGConfig("demo")

    assert config.get_contexts_file().name == "contexts.txt"
    assert config.get_questions_file().name == "qa.json"
    assert config.get_graph_json_file().name == "demo_graph.json"
    assert config.get_graph_gexf_file().name == "demo_graph.gexf"
    assert config.get_kv_store_file().name == "demo_kv_store.json"
    assert config.get_answer_file(answer_type="short").name == "demo_answers_short.json"
    assert config.get_evaluation_file(eval_method="f1").name == "demo_eval_f1.json"
