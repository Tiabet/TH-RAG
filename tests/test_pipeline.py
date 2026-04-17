from pipeline import resolve_steps


def test_resolve_steps_preserves_canonical_order() -> None:
    steps = resolve_steps(["answer_generation_short", "graph_build", "evaluation"])
    assert steps == [
        "graph_construction",
        "json_to_gexf",
        "edge_embedding",
        "answer_generation_short",
        "evaluation_f1",
    ]



def test_resolve_steps_expands_groups_without_duplicates() -> None:
    steps = resolve_steps(["graph_build", "graph_construction", "answer_generation"])
    assert steps == [
        "graph_construction",
        "json_to_gexf",
        "edge_embedding",
        "answer_generation_short",
        "answer_generation_long",
    ]
