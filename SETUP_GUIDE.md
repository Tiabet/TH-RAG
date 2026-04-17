# TH-RAG Setup Guide

This guide documents the public-release workflow for the TH-RAG codebase.

## 1. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before running the pipeline.

## 2. Prepare a Dataset

Create a dataset directory under `data/`:

```text
data/
|-- my_dataset/
|   |-- contexts.txt
|   `-- qa.json
```

`contexts.txt` should contain the source passages used for graph construction.

`qa.json` should contain question-answer pairs in this format:

```json
[
  {
    "query": "Which component builds the graph?",
    "answer": "Graph construction"
  }
]
```

## 3. Build Graph Artifacts

Run the graph-building subset of the pipeline:

```bash
python pipeline.py --dataset my_dataset --steps graph_build
```

This stage produces:

- `results/index/my_dataset_graph.json`
- `results/index/my_dataset_kv_store.json`
- `results/index/my_dataset_graph.gexf`
- `results/index/my_dataset_edge_index.faiss`
- `results/index/my_dataset_edge_payloads.npy`

You can also run the graph-only wrapper directly:

```bash
python index/build_graph.py --dataset my_dataset
```

## 4. Generate Answers

Short answers:

```bash
python pipeline.py --dataset my_dataset --steps answer_generation_short
```

Long answers:

```bash
python pipeline.py --dataset my_dataset --steps answer_generation_long
```

Both outputs together:

```bash
python pipeline.py --dataset my_dataset --steps answer_generation
```

Artifacts are written to `results/generated/`, and chunk-usage logs are written to `results/chunks/`.

## 5. Run F1 Evaluation

The F1 evaluator compares the short-answer output against `data/<dataset>/qa.json`.

```bash
python pipeline.py --dataset my_dataset --steps evaluation
```

The result is written to `results/evaluated/my_dataset_eval_f1.json`.

## 6. Pairwise LLM Evaluation

To compare TH-RAG against another system:

```bash
python evaluate/judge_Ultradomain.py \
  --answer-a results/generated/my_dataset_answers_long.json \
  --answer-b baseline_answers.json \
  --output results/evaluated/my_dataset_pairwise.json \
  --label-a TH-RAG \
  --label-b Baseline
```

Add `--plot results/evaluated/my_dataset_pairwise.png` to save a chart.

## 7. Useful Commands

List datasets:

```bash
python pipeline.py --list-datasets
```

Rebuild everything from scratch:

```bash
python pipeline.py --dataset my_dataset --force
```

Validate configuration:

```bash
python test_config.py
```

Run the test suite:

```bash
pytest
```

## 8. Troubleshooting

### Missing API key

If graph construction or evaluation fails immediately, confirm that `OPENAI_API_KEY` is set in `.env`.

### Missing dataset files

The pipeline requires `data/<dataset>/contexts.txt` for graph building and `data/<dataset>/qa.json` for answer generation and evaluation.

### Existing outputs are reused

The pipeline skips steps when the expected artifacts already exist. Use `--force` to rebuild them.

### FAISS or OpenAI errors

If the embedding stage fails, verify that the graph JSON and GEXF files were created successfully before rebuilding the index.
