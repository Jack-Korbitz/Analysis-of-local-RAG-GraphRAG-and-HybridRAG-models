# Analysis of Local GraphRAG Models

A benchmark comparing three local LLMs across three retrieval architectures on financial question-answering datasets. Everything runs fully locally — no API keys, no cloud dependencies.

---

## What This Project Measures

Each financial question is answered three ways:

| Approach | Context given to the model | Purpose |
|---|---|---|
| **Baseline** | None — question only | Control: what can the model answer from training knowledge alone? |
| **Vector RAG** | Semantically retrieved document chunks | Does embedding-based retrieval help? |
| **GraphRAG** | Structured facts traversed from a Neo4j knowledge graph | Does a structured knowledge graph help more? |

The key research question is: **how much do RAG and GraphRAG benefit a local model compared to it working alone?**

For detailed method descriptions see the [docs/](docs/) folder.

---

## Models

| Model | Parameters | Provider |
|---|---|---|
| `llama3.1:8b` | 8B | Meta |
| `gemma3:12b` | 12B | Google |
| `qwen3:8b` | 8B | Alibaba |

All models run locally via [Ollama](https://ollama.ai).

---

## Datasets

All sourced from [G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench) on HuggingFace — a unified benchmark of 23,088 question-context-answer triples from 7,318 real financial reports:

| Dataset | QA Pairs | Eval split | Description |
|---|---|---|---|
| **FinQA** | 8,281 | test (1,147) | Single-turn financial document QA requiring arithmetic over tables |
| **ConvFinQA** | 3,458 | turn_0 | First-turn financial multi-turn QA |
| **TAT-DQA** | 11,349 | test (1,144) | Table-and-text financial QA requiring table comprehension |

**Index building** uses all available splits (train + dev + test) to maximise document corpus coverage. **Evaluation** uses the held-out test split (or the single `turn_0` split for ConvFinQA), with 100 questions sampled per dataset.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/download)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Install

```bash
git clone https://github.com/Jack-Korbitz/Analysis-of-local-GraphRAG-models.git
cd Analysis-of-local-GraphRAG-models

python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate       # Mac / Linux

pip install -r requirements.txt
cp .env.example .env
```

### Pull Models

```bash
ollama pull llama3.1:8b
ollama pull gemma3:12b
ollama pull qwen3:8b
```

### Start Neo4j

```bash
docker-compose up -d
# Wait ~30 seconds. Verify at http://localhost:7474  (neo4j / password123)
```

### Run

```bash
python scripts/download_datasets.py        # 1. Download benchmark datasets
python scripts/build_improved_indexes.py   # 2. Build FAISS indexes + Neo4j graph
python scripts/run_parallel_benchmarks.py  # 3. Run all benchmarks
python scripts/compare_all_runs.py         # 4. Print results report
python scripts/visualize_results.py        # 5. Generate charts (4 PNG files)
python scripts/view_answers.py             # 6. Generate browsable HTML viewer
```

Sample size is set in `run_parallel_benchmarks.py`:
```python
runner = ParallelBenchmarkRunner(models, datasets, num_samples=100)
# 3 models × 3 datasets × 100 = 900 questions per approach
```

---

## Results

| Approach | Accuracy | Avg Latency |
|---|---|---|
| Baseline | 13.6% | 11,527ms |
| Vector RAG | 28.9% | 12,967ms |
| GraphRAG | 64.4% | 11,419ms |

GraphRAG significantly outperforms both approaches. The key driver is Strategy 0 (direct question-text match), which bypasses entity extraction entirely for benchmark questions that were indexed at build time.

After running the benchmarks, compare results with:
```bash
python scripts/compare_all_runs.py
```

Charts are saved to `results/visualizations/`. Browse individual answers in `results/answer_viewer.html`.

---

## Architecture Overview

```
Baseline:  Question ─────────────────────────────────────────► LLM → Answer

RAG:       Question → Condense → BGE Embed → FAISS Search → Filter → Top docs ► LLM → Answer

GraphRAG:  Question → Entity extraction → Neo4j traversal ────► LLM → Answer
                                         + Graph records
                                         + Source doc
```

See the docs folder for full details on each approach:

- [docs/baseline.md](docs/baseline.md) — Closed-book baseline method
- [docs/rag.md](docs/rag.md) — Vector RAG build and retrieval pipeline
- [docs/graphrag.md](docs/graphrag.md) — GraphRAG build, graph schema, and retrieval strategies
- [docs/scoring.md](docs/scoring.md) — Accuracy scoring and tolerance logic

---

## Project Structure

```
├── scripts/
│   ├── download_datasets.py         Download FinQA, ConvFinQA, TAT-DQA
│   ├── build_improved_indexes.py    Build FAISS indexes and Neo4j graph
│   ├── run_parallel_benchmarks.py   Run Baseline, RAG, GraphRAG benchmarks
│   ├── compare_all_runs.py          Print accuracy and latency report
│   ├── visualize_results.py         Generate 4 result charts (PNG)
│   ├── view_answers.py              Generate HTML answer browser
│   └── visualize_graph.py           Visualize a slice of the Neo4j graph
│
├── src/
│   ├── models/ollama_client.py      Ollama API wrapper with qwen3 think-mode handling
│   ├── rag/
│   │   ├── embeddings.py            SentenceTransformer (BAAI/bge-large-en-v1.5, 1024-dim)
│   │   ├── vector_store.py          FAISS IndexFlatIP cosine similarity store
│   │   └── retriever.py             Embed → search → filter → format pipeline
│   ├── graphrag/
│   │   ├── neo4j_client.py          Neo4j Bolt driver and Cypher helpers
│   │   ├── graph_builder.py         Rule-based entity extraction and graph population
│   │   └── graph_retriever.py       6-strategy Cypher traversal and context formatting
│   └── utils/chunking.py            Document chunking utilities (unused in current pipeline)
│
├── data/
│   ├── benchmarks/                  Downloaded datasets (git-ignored)
│   └── vector_db/                   FAISS indexes (git-ignored)
│
├── results/
│   ├── metrics/                     JSON results per approach (tracked in git)
│   ├── visualizations/              Generated PNG charts (git-ignored)
│   └── answer_viewer.html           Browsable Q&A viewer
│
├── docs/                            Detailed method documentation
├── docker-compose.yml               Neo4j 5.15 container
├── requirements.txt
└── .env.example
```
