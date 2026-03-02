# Analysis of Local GraphRAG Models

A benchmark comparing three local LLMs across three retrieval architectures on financial question-answering datasets. Everything runs fully locally — no API keys, no cloud dependencies.

---

## Overview

Each question is answered three ways, with accuracy compared across all combinations of model and approach:

| Approach | Description |
|---|---|
| **Baseline** | LLM answers from training knowledge alone — no retrieval |
| **Vector RAG** | Semantically similar document chunks are retrieved and passed as context |
| **GraphRAG** | A Neo4j knowledge graph of extracted financial facts is traversed for structured context |

---

## Models

| Model | Parameters | Provider |
|---|---|---|
| `llama3.1:8b` | 8B | Meta |
| `gemma3:12b` | 12B | Google |
| `qwen3:8b` | 8B | Alibaba |

## Datasets

All sourced from [G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench) on HuggingFace:

| Dataset | Description | Size |
|---|---|---|
| **FinQA** | Single-turn financial document QA with numeric answers | ~8,000 questions |
| **ConvFinQA** | Multi-turn follow-up questions on financial documents | ~3,400 questions |
| **TAT-DQA** | Table-and-text financial QA requiring table comprehension | ~11,000 questions |

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/download)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Installation

```bash
git clone https://github.com/Jack-Korbitz/Analysis-of-local-GraphRAG-models.git
cd Analysis-of-local-GraphRAG-models

python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate       # Mac / Linux

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
OLLAMA_BASE_URL=http://localhost:11434
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
```

Wait ~30 seconds. Verify at http://localhost:7474 (neo4j / password123).

---

## Running

Run the following scripts in order:

```bash
python scripts/test_ollama.py              # 1. Verify models are available
python scripts/download_datasets.py        # 2. Download benchmark datasets
python scripts/build_improved_indexes.py   # 3. Build FAISS indexes + Neo4j graph
python scripts/run_parallel_benchmarks.py  # 4. Run benchmarks
python scripts/compare_all_runs.py         # 5. View results report
python scripts/visualize_results.py        # 5. Generate results chart
```

Sample size is configured in `run_parallel_benchmarks.py`:
```python
runner = ParallelBenchmarkRunner(models, datasets, num_samples=100)
# 3 models × 3 datasets × 100 = 900 total questions
```

---

## Architecture

### Baseline

The question is sent directly to the LLM with no supporting context. This serves as the lower bound — any retrieval approach should improve on this.

```
Question → LLM → Answer
```

---

### Vector RAG

```
Question → Embed → FAISS Search → Top-8 Chunks → LLM → Answer
```

#### Embeddings — `src/rag/embeddings.py`

Text is converted to dense vectors using **`all-MiniLM-L6-v2`** from the `sentence-transformers` library:

| Property | Value |
|---|---|
| Model size | 22M parameters |
| Output dimensions | 384 |
| Similarity measure | Cosine (captures semantic meaning regardless of word choice) |

Semantically related phrases — "net sales" and "revenue", or "interest cost" and "interest expense" — are embedded near each other in vector space, enabling retrieval by meaning rather than keyword.

#### Chunking — `src/utils/chunking.py`

Documents are split into overlapping segments before indexing:

| Setting | Value |
|---|---|
| Chunk size | 300 characters |
| Overlap | 75 characters |
| Split boundary | Sentence endings |

Chunking ensures each retrieval slot returns a focused passage. Overlap prevents answers from being split across a boundary. Duplicate chunks are removed before indexing.

#### Vector Store — `src/rag/vector_store.py`

Uses **FAISS `IndexFlatIP`** (inner product on L2-normalized vectors = cosine similarity). Vectors are normalized on insertion so similarity search is equivalent to measuring the angle between vectors. The index is saved to disk and reloaded at benchmark time — no re-embedding required between runs.

#### Retriever — `src/rag/retriever.py`

Embeds the question, retrieves the top 8 most similar chunks, and formats them as numbered context blocks for the LLM.

**Prompt structure:**
```
Context:
[Document 1]
...chunk text...

[Document 2]
...chunk text...

Question: {question}

Let's approach this step-by-step:
1. Identify the relevant information in the context
2. Extract the specific data needed
3. Provide a clear, concise answer

Answer:
```

---

### GraphRAG

```
Question → Extract entities → Traverse Neo4j → Structured facts + Document text → LLM → Answer
```

#### Graph Builder — `src/graphrag/graph_builder.py`

Processes each dataset example using **rule-based extraction** (no LLM required) to populate Neo4j:

**Metric extraction uses two methods:**

1. **Statement patterns** — regex matching on prose text:
   ```
   "revenue of $3.8 million"  →  (metric: revenue, value: 3.8)
   ```

2. **Table column parsing** (`parse_table_year_columns`) — finds markdown tables with year column headers and extracts a (year, metric, value) triple for every cell:
   ```
   | Metric        | 2016  | 2017  | 2018  |
   | net income    | 450   | 510   | 620   |
   | total revenue | 3200  | 3800  | 4100  |
   ```
   → 6 structured facts from a single table

**23 canonical metric types** are recognized and mapped to standard names. Longest-keyword match wins when multiple terms apply to the same row.

| Raw text examples | Canonical name |
|---|---|
| "net revenue", "net sales" | `net_revenue` |
| "net income", "net earnings", "net profit" | `net_income` |
| "interest expense", "interest cost" | `interest_expense` |
| "total operating expenses", "operating costs" | `operating_expenses` |
| "gross profit" | `gross_profit` |
| "cost of goods sold", "cost of sales", "cost of revenue" | `cost_of_goods_sold` |
| "capital expenditures", "capital expenditure" | `capital_expenditures` |
| "cash and cash equivalents" | `cash_and_equivalents` |

#### Graph Schema — `src/graphrag/neo4j_client.py`

```
(Company)-[:HAS_METRIC]->(Metric)-[:FOR_YEAR]->(Year)
(Document)-[:ABOUT]---->(Company)
(Document)-[:FROM_YEAR]->(Year)
```

Current graph contents:

| Node | Count |
|---|---|
| Company | 306 |
| Document | 18,772 |
| Metric | 43,402 |
| Year | 25 |
| Relationships | 124,348 |

Each Document node stores up to 2,000 characters of raw context including financial tables.

#### Graph Retriever — `src/graphrag/graph_retriever.py`

The question is parsed for three entity types:

| Entity | Extraction method |
|---|---|
| Company | Matched against all Company nodes; longest-match-wins prevents partial name collisions |
| Year | Regex `\b(19[9]\d|20[0-3]\d)\b` |
| Metric type | Keyword lookup across recognized financial terms |

Five retrieval strategies are attempted in priority order:

| Priority | Strategy | Condition | Returns |
|---|---|---|---|
| 1 | Company + Year + Metric | All three extracted | Exact metric value |
| 2 | Company + Year | Company and year found | All metrics for that company and year |
| 3 | Company only | Company found, no year in question | Most recent metric values |
| 4 | Metric only | Metric found, no year in question | That metric across all companies |
| 5 | Document text | Company is known | Raw document text for that company and year |

Strategy 5 runs alongside strategies 1–4 rather than as a last resort. Since structured metrics cover only 23 canonical types, most questions require the raw table text to find the specific line item asked about. The LLM receives both the structured metric records and the full document text.

**Prompt structure:**
```
Knowledge graph records:
[Record 1] Company: Analog Devices | Year: 2009 | Metric: interest_expense | Value: $3.80 million
[Record 2] Company: Analog Devices | Year: 2009
   Text: ...financial table text...

Question: {question}

Answer (be concise, lead with the number):
```

---

## Accuracy Scoring

Financial answers have significant formatting variation. A tolerance-based match is applied across all scripts:

| Check | Description |
|---|---|
| Comma normalization | `41,932` → `41932` |
| String containment | Ground truth appears anywhere in the answer |
| Exact numeric match | Numbers parsed and compared directly |
| Percentage ↔ decimal | `0.35` matches `35%` and vice versa |
| Large number tolerance | Within 1% for values ≥ 100 |
| Small number tolerance | Within ± 0.5 for values < 100 |

---

## Project Structure

```
├── scripts/
│   ├── test_ollama.py               Verify Ollama connection and model availability
│   ├── download_datasets.py         Download FinQA, ConvFinQA, TAT-DQA from HuggingFace
│   ├── build_improved_indexes.py    Build FAISS vector indexes and Neo4j knowledge graph
│   ├── run_parallel_benchmarks.py   Run Baseline, RAG, and GraphRAG benchmarks in parallel
│   ├── compare_all_runs.py          Print accuracy and latency comparison report
│   └── visualize_results.py         Generate 4-panel benchmark results chart (PNG)
│
├── src/
│   ├── models/
│   │   └── ollama_client.py         Ollama API wrapper and prompt templates
│   ├── rag/
│   │   ├── embeddings.py            SentenceTransformer wrapper (all-MiniLM-L6-v2, 384-dim)
│   │   ├── vector_store.py          FAISS index with cosine similarity (IndexFlatIP)
│   │   └── retriever.py             Full RAG pipeline: embed → search → format context
│   ├── graphrag/
│   │   ├── neo4j_client.py          Neo4j Bolt driver wrapper and Cypher helpers
│   │   ├── graph_builder.py         Rule-based entity and metric extraction, graph population
│   │   └── graph_retriever.py       Five-strategy Cypher traversal and context formatting
│   └── utils/
│       └── chunking.py              Sentence-boundary chunker with configurable overlap
│
├── data/
│   ├── benchmarks/                  Downloaded datasets (not tracked in Git)
│   └── vector_db/                   FAISS indexes (not tracked in Git)
│
├── results/
│   └── metrics/
│       ├── baseline_fast.json       Benchmark results (tracked in Git)
│       ├── rag_fast.json
│       └── graphrag_fast.json
│
├── docker-compose.yml               Neo4j 5.15 container configuration
├── requirements.txt
└── .env.example
```
