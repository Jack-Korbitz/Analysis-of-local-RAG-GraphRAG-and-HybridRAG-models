# Analysis of Local GraphRAG Models

A benchmark comparison of local LLMs using three retrieval architectures:
**Baseline**, **Vector RAG**, and **GraphRAG** using Neo4j.

---

## Project Overview

This project compares how well local LLMs answer financial questions when given
different types of context retrieval. We test three local Ollama models against
standardized financial benchmarks using three retrieval strategies.

### Models Tested
- `gemma3:27b` - Google Gemma 3 (27B parameters)
- `gpt-oss:20b` - GPT OSS (20B parameters)
- `qwen3:30b` - Qwen 3 MoE (30B parameters)

### Retrieval Architectures
1. **Baseline** - No retrieval, models answer from training data only
2. **Vector RAG** - FAISS vector store with sentence-transformer embeddings
3. **GraphRAG** - Neo4j knowledge graph with entity/relationship extraction

### Datasets
- [G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench) - FinQA, ConvFinQA, TAT-DQA
- [galileo-ai/ragbench](https://huggingface.co/datasets/galileo-ai/ragbench) - hotpotqa, finqa, pubmedqa, msmarco

---

## Results Summary

### Latency Comparison (milliseconds)

| Model       | Baseline | Vector RAG | GraphRAG |
|-------------|----------|------------|----------|
| gemma3:27b  | 10,015ms | 16,381ms   | 14,078ms |
| gpt-oss:20b | 7,245ms  | 6,919ms    | **6,045ms** |
| qwen3:30b   | 8,023ms  | 8,335ms    | 8,069ms  |

### Key Findings

- GraphRAG correctly answered financial entity questions that Baseline and Vector RAG got wrong
- gpt-oss:20b is the fastest model across all three architectures
- Vector RAG struggles with long financial documents due to imprecise chunk retrieval
- GraphRAG excels at structured lookups (Company + Year + Metric)
- Baseline models hallucinate specific financial figures not in training data

### Answer Quality Example

Question: "What was Analog Devices reported interest expense for fiscal year 2009?"
Ground Truth: **$3.8 million**

| Approach   | Answer         | Correct? |
|------------|----------------|----------|
| Baseline   | $68 million    | No       |
| Vector RAG | $2.4 million   | No       |
| GraphRAG   | $3.80 million  | Yes      |

---

## Project Structure

```
Analysis-of-local-GraphRAG-models/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docker-compose.yml
├── config/
│   └── models.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/          # Downloaded locally, not in Git
├── src/
│   ├── models/
│   │   └── ollama_client.py
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── graphrag/
│   │   ├── neo4j_client.py
│   │   ├── graph_builder.py
│   │   └── graph_retriever.py
│   └── utils/
│       └── chunking.py
├── scripts/
│   ├── test_ollama.py
│   ├── download_datasets.py
│   ├── explore_datasets.py
│   ├── run_baseline_benchmark.py
│   ├── run_rag_benchmark.py
│   ├── run_graphrag_benchmark.py
│   ├── compare_baseline_rag.py
│   └── final_comparison.py
├── notebooks/
└── results/
    ├── metrics/
    │   ├── baseline_results.json
    │   ├── rag_results.json
    │   ├── rag_chunked_results.json
    │   └── graphrag_results.json
    └── visualizations/
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/download) installed and running
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

### 1. Clone the Repository

```bash
git clone https://github.com/Jack-Korbitz/Analysis-of-local-GraphRAG-models.git
cd Analysis-of-local-GraphRAG-models
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\Activate.ps1

# Activate on Mac/Linux
source venv/bin/activate
```

If you get an execution policy error on Windows:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit .env with your credentials
```

Your `.env` file should contain:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
OLLAMA_BASE_URL=http://localhost:11434

# Optional - for online LLM comparison
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### 5. Pull Ollama Models

```bash
ollama pull gemma3:27b
ollama pull gpt-oss:20b
ollama pull qwen3:30b
```

### 6. Start Neo4j with Docker

```bash
docker-compose up -d
```

Wait 30 seconds, then verify at http://localhost:7474

### 7. Download Datasets

```bash
python scripts/download_datasets.py
```

This downloads the following benchmark configurations:
- t2-ragbench: FinQA, ConvFinQA, TAT-DQA
- galileo-ragbench: hotpotqa, finqa, pubmedqa, msmarco

Note: Datasets are stored locally and excluded from Git.

---

## Running the Benchmarks

Run each benchmark in order to build up the full comparison.

### Step 1: Verify Ollama Connection

```bash
python scripts/test_ollama.py
```

Expected output:
```
Ollama is running!
Available models:
  - gemma3:27b (27.4B, Q4_K_M)
  - gpt-oss:20b (20.9B, MXFP4)
  - qwen3:30b (30.5B, Q4_K_M)
```

### Step 2: Run Baseline Benchmark

Tests models with no retrieval augmentation.

```bash
python scripts/run_baseline_benchmark.py
```

Results saved to: `results/metrics/baseline_results.json`

### Step 3: Run Vector RAG Benchmark

Builds a FAISS vector index from financial documents and tests retrieval-augmented answering.

```bash
python scripts/run_rag_benchmark.py
```

Results saved to: `results/metrics/rag_chunked_results.json`

### Step 4: Build Knowledge Graph and Run GraphRAG Benchmark

Extracts entities from financial documents into Neo4j and tests graph-based retrieval.

```bash
python scripts/run_graphrag_benchmark.py
```

Results saved to: `results/metrics/graphrag_results.json`

### Step 5: Run Final Comparison

```bash
$env:PYTHONIOENCODING="utf-8"
python scripts/final_comparison.py
```

To save the report to a file:
```bash
python scripts/final_comparison.py > results/final_comparison_report.txt
```

---

## Architecture Details

### Baseline

Models answer questions directly from their training data with no external context.

```
Question --> LLM --> Answer
```

### Vector RAG

Documents are chunked, embedded, and stored in a FAISS vector index.
At query time, the most similar chunks are retrieved and provided as context.

```
Question --> Embedding Model --> FAISS Search --> Top K Chunks --> LLM --> Answer
```

Components:
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Vector store: FAISS with cosine similarity
- Chunk size: 400 characters with 100 character overlap
- Retrieval: Top 3 most similar chunks

### GraphRAG

Financial entities are extracted from documents and stored as a knowledge graph in Neo4j.
At query time, entities are identified in the question and the graph is traversed to find relevant facts.

```
Question --> Entity Extraction --> Graph Traversal --> Structured Facts --> LLM --> Answer
```

Graph schema:
```
(Company)-[:HAS_METRIC]->(Metric)-[:FOR_YEAR]->(Year)
(Company)-[:HAS_DOCUMENT]->(Document)
(Document)-[:ABOUT]->(Company)
(Document)-[:FROM_YEAR]->(Year)
```

Retrieval strategies (in priority order):
1. Company + Year + Metric type lookup
2. Company + Year lookup (all metrics)
3. Company only lookup (all years)
4. Metric type only (all companies)
5. Document text search fallback

---

## Neo4j Browser

After starting Docker, you can explore the knowledge graph visually at:

```
http://localhost:7474
```

Useful Cypher queries:

```cypher
// View all companies
MATCH (c:Company) RETURN c.name LIMIT 25

// View metrics for a specific company
MATCH (c:Company {name: "Analog Devices"})-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year)
RETURN c.name, m.name, m.value, y.value

// View graph schema
CALL db.schema.visualization()

// Count all nodes by type
MATCH (n)
RETURN labels(n)[0] as type, count(n) as count
ORDER BY count DESC
```

---

## Configuration

### Model Configuration

Edit `config/models.yaml` to change which models are tested:

```yaml
ollama_models:
  - name: "gemma3:27b"
    type: "ollama"
    temperature: 0.7

  - name: "gpt-oss:20b"
    type: "ollama"
    temperature: 0.7

  - name: "qwen3:30b"
    type: "ollama"
    temperature: 0.7
```

### Benchmark Configuration

Key parameters you can adjust in the benchmark scripts:

| Parameter     | Default | Description                          |
|---------------|---------|--------------------------------------|
| num_samples   | 5       | Number of questions to test          |
| max_examples  | 200     | Documents to index for RAG/GraphRAG  |
| top_k         | 3       | Number of chunks/records to retrieve |
| chunk_size    | 400     | Characters per chunk (Vector RAG)    |
| chunk_overlap | 100     | Overlap between chunks               |

---

## Troubleshooting

### Ollama not found
```bash
# Make sure Ollama is running
ollama serve

# Check available models
ollama list
```

### Model requires too much memory
```
model requires more system memory than is available
```
Try a smaller model or more compressed quantization:
```bash
ollama pull gemma3:9b
```

### Neo4j connection refused
```bash
# Check Docker is running
docker ps

# Restart Neo4j
docker-compose down
docker-compose up -d
```

### Git push rejected (large files)
Datasets should not be committed to Git. Remove them from tracking:
```bash
git rm -r --cached data/benchmarks/
git add data/benchmarks/.gitkeep
git commit -m "Remove large dataset files from Git tracking"
```

### Windows PowerShell execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Unicode encoding error on Windows
```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/final_comparison.py
```

---

## Dependencies

Key packages used in this project:

| Package               | Purpose                          |
|-----------------------|----------------------------------|
| ollama                | Local LLM client                 |
| sentence-transformers | Document and query embeddings    |
| faiss-cpu             | Vector similarity search         |
| neo4j                 | Graph database client            |
| datasets              | HuggingFace dataset loading      |
| python-dotenv         | Environment variable management  |
| tqdm                  | Progress bars                    |
| langchain             | LLM framework utilities          |

Full dependency list in `requirements.txt`.

---

## Status

- [x] Repository setup
- [x] Ollama integration and model testing
- [x] Dataset download and exploration
- [x] Baseline benchmark
- [x] Vector RAG implementation (FAISS)
- [x] GraphRAG implementation (Neo4j)
- [x] Final comparison report
- [ ] Online LLM comparison (OpenAI, Anthropic, Google)
- [ ] Expanded dataset coverage
- [ ] Visualization dashboard
- [ ] LLM-based entity extraction for GraphRAG