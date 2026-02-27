# Analysis of Local GraphRAG Models

Benchmarks three local LLMs across three retrieval architectures — Baseline, Vector RAG, and GraphRAG — on financial QA datasets.

---

## Models

| Model | Size |
|---|---|
| `llama3.1:8b` | 8B |
| `gemma3:12b` | 12B |
| `qwen3:8b` | 8B |

## Datasets

All from [G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench):

- **FinQA** — Financial document QA
- **ConvFinQA** — Multi-turn financial QA
- **TAT-DQA** — Table and text financial QA

## Architectures

- **Baseline** — Model answers from training data only, no retrieval
- **Vector RAG** — FAISS vector index with sentence-transformer embeddings; retrieves top-5 chunks per question
- **GraphRAG** — Neo4j knowledge graph with extracted financial entities; traverses graph to find structured facts

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/download) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Install

```bash
git clone https://github.com/Jack-Korbitz/Analysis-of-local-GraphRAG-models.git
cd Analysis-of-local-GraphRAG-models

python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

If you get a PowerShell execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Environment Variables

```bash
cp .env.example .env
```

`.env` should contain:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
OLLAMA_BASE_URL=http://localhost:11434
```

### Pull Ollama Models

```bash
ollama pull llama3.1:8b
ollama pull gemma3:12b
ollama pull qwen3:8b
```

### Start Neo4j

```bash
docker-compose up -d
```

Wait ~30 seconds. Verify at http://localhost:7474 or `docker ps`.

---

## Running the Project

Run these scripts **in order**.

### Step 1 — Verify Ollama

Confirms Ollama is running and models are available.

```bash
python scripts/test_ollama.py
```

### Step 2 — Download Datasets

Downloads FinQA, ConvFinQA, and TAT-DQA locally to `data/benchmarks/`.

```bash
python scripts/download_datasets.py
```

### Step 3 — Build Indexes

Builds the FAISS vector indexes (RAG) and populates the Neo4j knowledge graph (GraphRAG). Must be run before benchmarks.

```bash
python scripts/build_improved_indexes.py
```

This indexes up to 2000 documents per dataset.

### Step 4 — Run Benchmarks

Runs all three approaches (Baseline, RAG, GraphRAG) in parallel across all models and datasets. 15 questions per dataset per model.

```bash
# Ensure Docker is running first:
# docker-compose up -d

python scripts/run_parallel_benchmarks.py
```

Results saved to:
- `results/metrics/baseline_fast.json`
- `results/metrics/rag_fast.json`
- `results/metrics/graphrag_fast.json`

### Step 5 — Compare Results

Loads all result files and prints a summary comparison across models and approaches.

```bash
python scripts/compare_all_runs.py
```

---

## Project Structure

```
├── scripts/
│   ├── test_ollama.py              # Step 1 - verify Ollama connection
│   ├── download_datasets.py        # Step 2 - download benchmark datasets
│   ├── build_improved_indexes.py   # Step 3 - build RAG + GraphRAG indexes
│   ├── run_parallel_benchmarks.py  # Step 4 - run all benchmarks
│   └── compare_all_runs.py         # Step 5 - compare results
├── src/
│   ├── models/
│   │   └── ollama_client.py        # Ollama API wrapper
│   ├── rag/
│   │   ├── embeddings.py           # Sentence-transformer embeddings
│   │   ├── vector_store.py         # FAISS vector store
│   │   └── retriever.py            # RAG retriever
│   ├── graphrag/
│   │   ├── neo4j_client.py         # Neo4j connection
│   │   ├── graph_builder.py        # Entity extraction + graph population
│   │   └── graph_retriever.py      # Graph traversal for context
│   └── utils/
│       └── chunking.py             # Document chunking
├── data/
│   └── benchmarks/                 # Downloaded datasets (not in Git)
├── results/
│   └── metrics/                    # Benchmark output JSON files
├── config/
│   └── models.yaml
├── docker-compose.yml              # Neo4j container
├── requirements.txt
└── .env.example
```

---

## Graph Schema (Neo4j)

```
(Company)-[:HAS_METRIC]->(Metric)-[:FOR_YEAR]->(Year)
(Company)-[:HAS_DOCUMENT]->(Document)
(Document)-[:ABOUT]->(Company)
(Document)-[:FROM_YEAR]->(Year)
```

Retrieval priority:
1. Company + Year + Metric type
2. Company + Year
3. Company only
4. Metric type only
5. Document text fallback

---

## Troubleshooting

**Ollama not running**
```bash
ollama serve
ollama list
```

**Neo4j connection refused**
```bash
docker ps                  # check container is running
docker-compose down
docker-compose up -d       # restart
```

**Unicode error on Windows**
```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/compare_all_runs.py
```
