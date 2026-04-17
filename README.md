# TH-RAG

Reference implementation for the ACL 2026 paper **TH-RAG: Topic-Based Hierarchical Knowledge Graphs for Robust Multi-hop Reasoning in Graph-based RAG Systems**.

This repository contains the end-to-end research pipeline used to:

1. build hierarchical knowledge-graph artifacts from raw contexts,
2. index graph evidence with FAISS,
3. generate short-form and long-form answers, and
4. evaluate model outputs.

## Repository Status

The codebase has been cleaned for public release with these constraints:

- English-only comments, documentation, and user-facing messages
- no decorative emoji or informal console output
- removal of empty placeholder files and unrelated cleanup scripts
- unified configuration and path handling across graph construction, retrieval, generation, and evaluation

## Requirements

- Python 3.10+
- An OpenAI API key exposed through `OPENAI_API_KEY`
- Platform support: Windows, Linux, and macOS through the Python CLI

Install dependencies:

```bash
pip install -r requirements.txt
```

Optionally, install the project metadata and test dependencies:

```bash
pip install -e .[dev]
```

## Dataset Format

Each dataset lives under `data/<dataset_name>/`.

Required files:

- `contexts.txt`: source passages used to build the hierarchical graph
- `qa.json`: evaluation questions and gold answers

Expected `qa.json` format:

```json
[
  {
    "query": "What does TH-RAG use to index graph evidence?",
    "answer": "FAISS"
  }
]
```

An example dataset is included at `data/test_dataset/`.

## Quick Start

1. Copy the environment template.

```bash
cp .env.example .env
```

2. Add your OpenAI API key to `.env`.

3. Validate the configuration.

```bash
python test_config.py
```

4. List available datasets.

```bash
python pipeline.py --list-datasets
```

5. Run the full pipeline.

```bash
python pipeline.py --dataset test_dataset
```

## Pipeline Stages

The unified pipeline exposes the following canonical steps:

- `graph_construction`: chunk `contexts.txt`, extract triples, and create the chunk KV store
- `json_to_gexf`: convert extracted triples into a hierarchical GEXF graph
- `edge_embedding`: embed predicate-edge evidence and build the FAISS index
- `answer_generation_short`: produce concise answers
- `answer_generation_long`: produce detailed answers
- `evaluation_f1`: score the short-answer output against the gold answers in `qa.json`

You can also use step aliases:

- `graph_build` -> `graph_construction json_to_gexf edge_embedding`
- `answer_generation` -> `answer_generation_short answer_generation_long`
- `evaluation` -> `evaluation_f1`

Examples:

```bash
python pipeline.py --dataset test_dataset --steps graph_build
python pipeline.py --dataset test_dataset --steps answer_generation_short
python pipeline.py --dataset test_dataset --steps evaluation
python pipeline.py --dataset test_dataset --steps graph_build answer_generation_short --force
```

## Output Layout

Generated artifacts are written under `results/`.

- `results/index/`: extracted graph JSON, KV store, GEXF graph, FAISS index, and payloads
- `results/generated/`: model answers
- `results/chunks/`: chunk usage logs for answer generation
- `results/evaluated/`: evaluation summaries
- `temp/`: pipeline state bookkeeping

## Pairwise Evaluation

For pairwise LLM-based comparison between two answer files, use the UltraDomain-style evaluator:

```bash
python evaluate/judge_Ultradomain.py \
  --answer-a results/generated/test_dataset_answers_long.json \
  --answer-b other_system_answers.json \
  --output results/evaluated/test_dataset_pairwise.json \
  --label-a TH-RAG \
  --label-b Baseline \
  --plot results/evaluated/test_dataset_pairwise.png
```

## Windows Helper

A menu-driven Windows launcher is available:

```cmd
run_pipeline.bat
```

## Repository Structure

```text
THRAG/
|-- config.py
|-- pipeline.py
|-- test_config.py
|-- README.md
|-- SETUP_GUIDE.md
|-- requirements.txt
|-- pyproject.toml
|-- run_pipeline.bat
|-- data/
|-- index/
|   |-- build_graph.py
|   |-- graph_construction.py
|   |-- json_to_gexf.py
|   |-- edge_embedding.py
|   |-- topic_choice.py
|   |-- subtopic_choice.py
|-- generate/
|   |-- Retriever.py
|   |-- graph_rag.py
|   |-- graph_based_rag_short.py
|   |-- graph_based_rag_long.py
|   |-- answer_generation_short.py
|   |-- answer_generation_long.py
|-- evaluate/
|   |-- judge_F1.py
|   |-- judge_Ultradomain.py
|-- prompt/
|-- tests/
```

## Notes

- The pipeline expects `qa.json` to contain gold answers if you want to run `evaluation_f1`.
- `answer_generation_short` is the output consumed by the F1 evaluator by default.
- Public-release cleanup removed unfinished wrappers and utility files that were not part of the research pipeline.

## License

Apache License 2.0
