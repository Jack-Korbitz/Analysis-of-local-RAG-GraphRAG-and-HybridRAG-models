# Pipeline Walkthrough

This project does not train any models. All LLMs run as-is through Ollama, and the embedding model (`all-MiniLM-L6-v2`) is downloaded pre-trained from HuggingFace. The work that is analogous to "training" is **index building** — converting raw documents into the searchable structures (FAISS index, Neo4j graph) that each retrieval approach depends on. This doc walks through the full pipeline from raw data to benchmark results.

---

## Stage 0 — Environment Setup

Before any data is processed, three services must be running:

| Service | Purpose | How |
|---|---|---|
| Ollama | Serves all three LLMs locally | `ollama pull llama3.1:8b` etc. |
| Neo4j | Stores and queries the knowledge graph | `docker-compose up -d` |
| Python venv | Isolates Python dependencies | `pip install -r requirements.txt` |

`scripts/test_ollama.py` pings Ollama and confirms each model responds before anything else runs.

---

## Stage 1 — Data Download

**Script:** `scripts/download_datasets.py`

Downloads three financial QA datasets from HuggingFace ([G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench)) and saves them locally with `save_to_disk`:

| Dataset | Field layout | ~Size |
|---|---|---|
| FinQA | `pre_text`, `table`, `post_text`, `program_answer` | 8,000 Q |
| ConvFinQA | Same as FinQA + multi-turn `question` chain | 3,400 Q |
| TAT-DQA | `context` only (no separate table field) | 11,000 Q |

Each example also carries `company_name` and `report_year`, which the graph builder uses directly.

---

## Stage 2 — Index Building

**Script:** `scripts/build_improved_indexes.py`

This is the most compute-intensive stage. It runs two builders in sequence.

### 2a. FAISS Vector Index (for Vector RAG)

```
Raw documents
    → DocumentChunker (chunk_size=600, overlap=100, sentence boundaries)
    → Deduplicate (drop chunks < 50 chars or exact duplicates)
    → EmbeddingModel.embed_batch() — all-MiniLM-L6-v2, batch=32
    → FAISS IndexFlatIP (L2-normalized → cosine similarity)
    → saved to data/vector_db/rag_<dataset>_v2/
```

One index is built per dataset. Up to 2,000 documents are chunked per dataset. The resulting indexes hold tens of thousands of 384-dimensional vectors. Because the index is saved to disk, embeddings only need to be computed once — the benchmark loads from disk at query time.

**Key parameters:**

| Parameter | Value | Effect |
|---|---|---|
| `chunk_size` | 600 chars | Balances context richness vs. retrieval precision |
| `chunk_overlap` | 100 chars | Prevents answers split across chunk boundaries |
| `max_examples` | 2,000 | Controls how many documents are indexed per dataset |
| `embed_batch_size` | 32 | Throughput on CPU |

### 2b. Neo4j Knowledge Graph (for GraphRAG)

```
Raw documents (full dataset, no cap)
    → GraphBuilder.build_from_dataset(use_llm=False)
        → create_company() node per unique company_name
        → create_document() node (stores up to 2,000 chars of raw context)
        → extract_entities_rule_based() — regex + keyword matching
        → parse_table_year_columns() — multi-year table parsing
        → create_metric() nodes linked to Company and Year
    → Neo4j holds completed graph
```

**Entity extraction (rule-based, no LLM):**

The `GraphBuilder` uses three extraction passes on each document:

1. **`_extract_from_tables`** — scans markdown table rows for cells matching metric keywords, reads the numeric value from that row.

2. **`_extract_from_sentences`** — 28 regex patterns covering constructs like `"revenue of $3.8 million"`, `"net income increased to $510M"`.

3. **`_extract_from_statements`** — simplified `LABEL $VALUE` patterns for structured financial statement layouts.

4. **`parse_table_year_columns`** — finds tables with year headers (`| 2016 | 2017 | 2018 |`) and emits one `(year, metric, value)` triple per cell. This is what creates the majority of the multi-year metric coverage.

**Metric normalization:**

All values are normalized to millions. Billion-denominated values are multiplied by 1,000; thousand-denominated values are divided by 1,000. Values between 1900–2100 are skipped (treated as years, not financial figures).

**Graph schema after build:**

```
(Company)-[:HAS_METRIC]->(Metric {name, value, unit})-[:FOR_YEAR]->(Year {value})
(Company)-[:HAS_DOCUMENT]->(Document {text, year, question})
(Document)-[:MENTIONS]->(Metric)
(Document)-[:FROM_YEAR]->(Year)
```

Typical graph size after full dataset ingestion:

| Node type | Count |
|---|---|
| Company | 306 |
| Document | 18,772 |
| Metric | 43,402 |
| Year | 25 |
| Relationships | 124,348 |

---

## Stage 3 — Benchmark Execution

**Script:** `scripts/run_parallel_benchmarks.py`

Runs all three approaches sequentially (Baseline → RAG → GraphRAG), with up to `max_workers=3` threads handling different model/dataset combinations in parallel within each approach.

**Sample size:** 100 questions per dataset per model = 900 total questions per approach.

### 3a. Baseline

No retrieval. The document context embedded in each dataset example is passed directly alongside the question. This is not "zero-shot from training data" — it's a direct-context pass where the model reads the annual report excerpt and answers from it. It serves as the lower bound for the retrieval approaches.

### 3b. Vector RAG

```
Question
    → embed with all-MiniLM-L6-v2
    → FAISS search, top_k=10 candidates
    → filter: cosine score >= 0.4, year must match question years
    → keep top 3 passing chunks
    → combine with source document context
    → LLM generates answer
```

The retrieved passages are placed first, source document last. This exploits the model's recency bias — anchoring the final answer on the authoritative source document.

### 3c. GraphRAG

```
Question
    → entity extraction (company, year, metric type via regex/keyword)
    → 5-strategy Cypher waterfall (stop at first successful retrieval)
    → structured records + raw document text
    → LLM generates answer
```

The graph context is capped at 1,500 characters to prevent context overflow in 8B-parameter models. The source document is appended after.

**5-strategy waterfall (priority order):**

| # | Condition | Returns |
|---|---|---|
| 1 | Company + year + metric all parsed | Exact metric value node |
| 2 | Company + year, metric not in 23 types | All metrics for that company/year |
| 3 | Company only, no year | Most recent metric values |
| 4 | Metric only, no company | That metric across all companies |
| 5 | Company known | Raw document text for that company/year |

Strategy 5 runs alongside 1–4, not as a fallback. Structured metric records and raw document text are combined into a single context.

---

## Stage 4 — Accuracy Scoring

Each model response is scored against the ground truth answer (`program_answer` or `original_answer`) using a tolerance-based match function in `_is_correct()`:

| Check | Logic |
|---|---|
| Comma normalization | `"41,932"` → `41932` before comparison |
| String containment | Ground truth appears anywhere in the response |
| Numeric extraction | Pulls the leading/final/explicit number from the response |
| Relative tolerance | Within 1% for values >= 100 |
| Absolute tolerance | Within ±0.5 for values < 100 |
| Percent ↔ decimal | `0.35` matches `35%` |
| Billion/million | `"3.8 billion"` → `3800` before comparison |

The number extractor (`_extract_answer_number`) tries in this priority: explicit `ANSWER:` tag → last `= <value>` in shown calculations → first number matching a set of fallback patterns.

---

## Stage 5 — Results

**Scripts:** `scripts/compare_all_runs.py`, `scripts/visualize_results.py`

Results are written to `results/metrics/` as JSON after each approach completes:

```
baseline_fast.json   →  { "llama3.1:8b_finqa": [{question, ground_truth, answer, correct, latency_ms}, ...], ... }
rag_fast.json
graphrag_fast.json
```

`compare_all_runs.py` prints accuracy and average latency per model/dataset combination.

`visualize_results.py` generates a 4-panel PNG chart (requires `matplotlib` — run with the conda Python: `python scripts/visualize_results.py`).

---

## Full Execution Order

```bash
python scripts/test_ollama.py              # 1. verify Ollama + models
python scripts/download_datasets.py        # 2. download FinQA / ConvFinQA / TAT-DQA
python scripts/build_improved_indexes.py   # 3. build FAISS indexes + Neo4j graph  ← the "training" step
python scripts/run_parallel_benchmarks.py  # 4. run all 900 questions × 3 approaches
python scripts/compare_all_runs.py         # 5. print accuracy table
python scripts/visualize_results.py        # 5. generate results chart (PNG)
```