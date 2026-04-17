import json

import networkx as nx

from index.graph_construction import parse_triples_response
from index.json_to_gexf import convert_json_to_gexf


def test_parse_triples_response_accepts_top_level_list() -> None:
    payload = json.dumps(
        [
            {
                "triple": ["TH-RAG", "uses", "FAISS"],
                "sentence": "TH-RAG uses FAISS.",
                "subject": {"subtopic": "System", "main_topic": "Research"},
                "object": {"subtopic": "Index", "main_topic": "Infrastructure"},
            }
        ]
    )

    triples = parse_triples_response(payload)
    assert len(triples) == 1
    assert triples[0]["triple"][2] == "FAISS"



def test_convert_json_to_gexf_writes_graph(tmp_path) -> None:
    graph_json = tmp_path / "graph.json"
    graph_json.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk-00000",
                    "triples": [
                        {
                            "triple": ["TH-RAG", "uses", "FAISS"],
                            "sentence": "TH-RAG uses FAISS.",
                            "subject": {"subtopic": "System", "main_topic": "Research"},
                            "object": {"subtopic": "Index", "main_topic": "Infrastructure"},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "graph.gexf"
    convert_json_to_gexf(str(graph_json), str(output_path))

    graph = nx.read_gexf(output_path)
    labels = {data["label"] for _, data in graph.nodes(data=True)}
    assert "TH-RAG" in labels
    assert "FAISS" in labels
