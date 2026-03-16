# Vector RAG Method

## Overview

Vector RAG (Retrieval-Augmented Generation) retrieves the most semantically similar document chunks to the question and passes them as context to the model alongside the original source document.

```
Question → Embed → FAISS Search → Filter → Top chunks + Source doc → LLM → Answer
```

Unlike the baseline, the model has access to relevant passages pulled from a pre-built index of all benchmark documents. Unlike GraphRAG, retrieval is based on embedding similarity rather than structured graph traversal.

---

## Build Stage

Indexes are built by running:
```bash
python scripts/build_improved_indexes.py
```

One FAISS index is built per dataset (finqa, convfinqa, tatqa). Indexes are saved to `data/vector_db/` and reloaded at benchmark time.

### Step 1 — Context Extraction

All available splits (train, dev, test) are concatenated before indexing to maximise corpus coverage. For ConvFinQA, which only has a single `turn_0` split, that split is used in full. Each dataset example is read and its document context is assembled:

| Dataset | Splits indexed | Fields used |
|---|---|---|
| FinQA | train + dev + test (8,281 docs) | `pre_text` + `table` + `post_text` |
| ConvFinQA | turn_0 (3,458 docs) | `pre_text` + `table` + `post_text` |
| TAT-DQA | train + dev + test (11,349 docs) | `context` |

### Step 2 — Table-Aware Chunking (`src/utils/chunking.py`)

Documents are split into overlapping segments before indexing. A table-aware chunker is used so markdown table rows are never split mid-row — entire table blocks are kept intact as single chunks.

| Setting | Value |
|---|---|
| Chunk size | 600 characters |
| Overlap | 100 characters |
| Split boundary | Sentence endings for prose; table block boundaries for tables |

Duplicate chunks are removed before indexing. Only chunks with more than 50 characters are kept.

### Step 3 — Embedding (`src/rag/embeddings.py`)

Text chunks are converted to dense vectors using `all-MiniLM-L6-v2` from the `sentence-transformers` library:

| Property | Value |
|---|---|
| Model size | 22M parameters |
| Output dimensions | 384 |
| Similarity measure | Cosine |

Semantically related phrases — "net sales" and "revenue", or "interest cost" and "interest expense" — are embedded close together in vector space, enabling retrieval by meaning rather than exact keyword match.

### Step 4 — FAISS Indexing (`src/rag/vector_store.py`)

Embeddings are stored in a **FAISS `IndexFlatIP`** index (inner product on L2-normalized vectors, which equals cosine similarity). The index is saved to disk as a pickle file alongside the text and metadata arrays.

Each chunk stores metadata including company name, year, source example ID, and the original question it came from.

---

## How It Works at Benchmark Time

### Retrieval Pipeline (`src/rag/retriever.py`)

1. **Embed the question** using the same `all-MiniLM-L6-v2` model
2. **Search FAISS** for the top 10 most similar chunks
3. **Filter by similarity score** — chunks with cosine similarity below 0.4 are discarded
4. **Filter by year** — if the question mentions a specific year (e.g. "2018"), chunks from a different year are excluded
5. **Keep up to 3 quality chunks** after filtering

### Context Construction

Retrieved chunks are placed first, followed by the source document from the dataset. This ordering exploits the model's recency bias — placing the authoritative source document last anchors the final answer on the ground-truth text.

```
[Retrieved passages]
[Document 1]
...chunk text...

[Document 2]
...chunk text...

[Source document]
...pre_text + table + post_text...
```

If no chunks pass the similarity/year filter, only the source document is passed.

### Prompt Structure

**FinQA / TAT-DQA (chain-of-thought):**
```
Context:
{retrieved chunks + source document}

Question: {question}

Show your step-by-step calculation, then write ANSWER: <number>
```

**ConvFinQA (number-only):**
```
Context:
{retrieved chunks + source document}

Question: {question}

ANSWER:
```

### Token Limits

| Dataset | Max tokens |
|---|---|
| FinQA | 1000 |
| TAT-DQA | 1000 |
| ConvFinQA | 600 |

RAG gets more tokens than baseline to accommodate the retrieved context.

---

## Output Format

Results are saved to `results/metrics/rag.json` with the same structure as baseline:

```json
{
  "llama3.1:8b_finqa": [
    {
      "question": "What were the total operating expenses for American Airlines in 2018?",
      "ground_truth": "41932.2",
      "answer": "Based on the retrieved context, total operating expenses were $41,932 million.\n\nANSWER: 41932",
      "correct": true,
      "latency_ms": 15243.12
    }
  ]
}
```

### Why RAG Sometimes Underperforms Baseline

When all three approaches (baseline, RAG, GraphRAG) receive the source document, RAG adding retrieved chunks does not always help — the source document already contains the answer, and extra retrieved text can introduce conflicting numbers. With a closed-book baseline, RAG's lift over baseline is much more pronounced.
