# KGRAG - Knowledge Graph-based Retrieval Augmented Generation

A knowledge graph–based RAG (Retrieval-Augmented Generation) system.

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Tiabet/KGRAG.git
cd KGRAG

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
# or
set OPENAI_API_KEY=your-api-key-here  # Windows
```

### 2. Unified Run Interface

**Windows:**
```cmd
run_all.bat
```

**Linux/Mac:**
```bash
chmod +x run_all.sh
./run_all.sh
```

## 📁 Project Structure

```
KGRAG/
├── 📁 index/              # Graph construction
│   ├── build_graph.py
│   ├── graph_construction.py
│   ├── json_to_gexf.py
│   ├── edge_embedding.py
│   ├── build_index.sh     # For Linux/Mac
│   └── build_index.bat    # For Windows
│
├── 📁 generate/           # Answer generation
│   ├── graph_based_rag_short.py
│   ├── graph_based_rag_long.py
│   ├── answer_generation_short.py
│   ├── answer_generation_long.py
│   ├── generate_answers.sh   # For Linux/Mac
│   └── generate_answers.bat  # For Windows
│
├── 📁 evaluate/           # Answer evaluation
│   ├── judge_F1.py
│   ├── judge_Ultradomain.py
│   ├── evaluate_answers.sh   # For Linux/Mac
│   └── evaluate_answers.bat  # For Windows
│
├── 📁 prompt/             # Prompt templates
├── 📁 hotpotQA/           # Example dataset
├── 📁 UltraDomain/        # Example dataset
├── 📁 MultihopRAG/        # Example dataset
│
├── Retriever.py           # Common retriever
├── subtopic_choice.py     # Subtopic selection
├── topic_choice.py        # Topic selection
├── requirements.txt       # Dependencies
├── SETUP_GUIDE.md         # Detailed guide
├── run_all.sh             # Unified runner (Linux/Mac)
└── run_all.bat            # Unified runner (Windows)
```

## 🔧 Usage

### 1. Configuration ⚙️

```bash
# Create a .env file (copy from template)
cp .env.example .env

# Or generate a sample via the test script
python test_config.py --create-env

# Set your API key in .env
# OPENAI_API_KEY=your_actual_api_key_here
```

**Key settings:**
- `OPENAI_API_KEY`: OpenAI API key (required)
- `DEFAULT_MODEL`: default model (default: gpt-4o-mini)
- `TOP_K1`, `TOP_K2`: RAG retrieval parameters (default: 50, 10)
- `TOPIC_CHOICE_MIN/MAX`: number of topics to select (default: 5–10)
- `SUBTOPIC_CHOICE_MIN/MAX`: number of subtopics to select (default: 10–25)
- `MAX_TOKENS`, `OVERLAP`: text chunking (default: 3000, 300)
- `TEMPERATURE`: generation temperature (default: 0.5)

```bash
# Validate configuration
python test_config.py
```

### 2. Build the Graph Index 🏗️

First build a knowledge graph from text data and create a FAISS index.

**Required input:** `[dataset]/contexts.txt`

**Artifacts produced:**
- `graph_v1.json` — extracted triples
- `graph_v1.gexf` — graph file
- `edge_index_v1.faiss` — FAISS vector index
- `edge_payloads_v1.npy` — metadata

### 3. Generate Answers 🤖

Use the constructed graph to generate answers to questions.

**Required input:**
- Indexed dataset
- `[dataset]/qa.json` — question file

**Output:** results saved under `Result/Generated/`

**Modes:**
- Short answers (faster)
- Long answers (more detailed)
- Interactive mode (real-time Q&A)

### 3. Evaluate Answers 📊

Compare generated answers against the gold standard to measure performance.

**Metrics:**
- **F1 Score** — precision, recall, F1
- **UltraDomain evaluation** — LLM-based quality assessment

**Output:** results saved under `Result/Evaluation/`

## 💡 Highlights

- **Modular design:** clear separation by functionality
- **Cross-platform:** Windows, Linux, and Mac supported
- **Interactive interface:** easy-to-use menu system
- **Parallel processing:** multithreading for speed
- **Flexible configuration:** rich options and skip controls
- **Detailed logging:** progress and error tracking

## 📊 Performance

- **Speed:** fast graph construction with multithreading
- **Scalability:** supports large datasets
- **Accuracy:** high-quality triple extraction and retrieval

## 🛠️ Developer Guide

### How to Use

**1. GUI Tool (Windows)**
```bash
# Run the GUI tool on Windows
run_pipeline.bat
```

**2. Command-Line Interface**
```bash
# Run the entire pipeline
python pipeline.py --dataset your_dataset

# Run specific steps only
python pipeline.py --dataset your_dataset --steps graph_construction,edge_embedding

# List available datasets
python pipeline.py --list-datasets

# Force re-run (overwrite existing results)
python pipeline.py --dataset your_dataset --force
```

**3. Individual Modules (for debugging)**
```bash
# Graph construction
python index/graph_construction.py your_dataset

# Answer generation
python generate/answer_generation_short.py your_dataset

# Evaluation
python evaluate/judge_F1.py your_dataset
```

### Adding a New Dataset

1. Create `data/[dataset_name]/`
2. Save text data to `data/[dataset_name]/contexts.txt`
3. (Optional) Save a question list to `data/[dataset_name]/questions.txt`
4. Run the pipeline

### Tuning Configuration

Adjust hyperparameters in `.env`:
```env
# RAG retrieval performance
TOP_K1=100         # Retrieve more edges (default: 50)
TOP_K2=20          # Select more chunks (default: 10)

# Topic selection ranges
TOPIC_CHOICE_MAX=15      # More diverse topics (default: 10)
SUBTOPIC_CHOICE_MAX=30   # More diverse subtopics (default: 25)

# Model parameters
TEMPERATURE=0.3          # More conservative answers (default: 0.5)
MAX_TOKENS=5000          # Longer context (default: 3000)
```

## 📁 Project Layout

```
KGRAG/
├── 📄 pipeline.py          # Unified pipeline runner
├── 📄 config.py            # Configuration management
├── 📄 test_config.py       # Configuration test tool
├── 🖥️ run_pipeline.bat     # Windows GUI tool
├── 📁 index/               # Graph construction modules
├── 📁 generate/            # Answer generation modules
├── 📁 evaluate/            # Evaluation modules
├── 📁 prompt/              # Prompt templates
├── 📁 data/                # Datasets
└── 📁 results/             # Execution outputs
```

## 📝 License

Apache License 2.0

## 🤝 Contributing

Bug reports, feature requests, and pull requests are welcome!

---

For more details, see [SETUP_GUIDE.md](SETUP_GUIDE.md).
