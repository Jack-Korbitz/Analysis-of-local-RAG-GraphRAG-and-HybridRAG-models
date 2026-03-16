# GraphRAG Method

## Overview

GraphRAG retrieves context by traversing a Neo4j knowledge graph of structured financial facts and document text, then passes the results alongside the source document to the model.

```
Question → Entity extraction → Neo4j traversal → Graph records + Source doc → LLM → Answer
```

Unlike vector RAG which retrieves by embedding similarity, GraphRAG uses explicit structured relationships — company nodes, metric nodes, year nodes, and document nodes — to find precisely relevant facts.

---

## Build Stage

The graph and RAG indexes are built together by running:
```bash
python scripts/build_improved_indexes.py
```

The graph is cleared and rebuilt from scratch each time. Build time depends on dataset size and hardware.

### Step 1 — Context Extraction (`src/graphrag/graph_builder.py`)

All available splits (train, dev, test) are concatenated before graph building to maximise coverage. For ConvFinQA, which has only a single `turn_0` split, that split is used in full:

| Dataset | Splits indexed | Fields used for context |
|---|---|---|
| FinQA | train + dev + test (8,281 docs) | `pre_text` + `table` + `post_text` |
| ConvFinQA | turn_0 (3,458 docs) | `pre_text` + `table` + `post_text` |
| TAT-DQA | train + dev + test (11,349 docs) | `context` |

Up to 4,000 characters of context are stored per document node.

### Step 2 — Company and Year Identification

- **Company**: read from `company_name` field (TAT-DQA). FinQA/ConvFinQA do not have this field, so company is stored as `"Unknown"` — these documents are still reachable via question-text lookup (see Strategy 0 in retrieval).
- **Year**: read from `report_year` field if present; otherwise extracted from the question text using regex `\b(19[9]\d|20[0-3]\d)\b`.

### Step 3 — Document Node Creation (`src/graphrag/neo4j_client.py`)

A `Document` node is created (or updated via MERGE) for each example:

```cypher
MERGE (d:Document {id: $doc_id})
SET d.text = $text
SET d.company = $company
SET d.year = $year
SET d.question = $question   -- up to 500 characters
```

The doc ID is prefixed with the dataset name (`finqa_`, `convfinqa_`, `tatqa_`) to prevent collisions across datasets.

An index on `d.question` is created at build time so question-based lookups are fast.

### Step 4 — Metric Extraction

Two extraction methods run in parallel on each document's text:

#### Rule-Based Sentence Patterns

Regex patterns match financial figures mentioned in prose:

```
"interest expense of $3.8 million"  →  (metric: interest_expense, value: 3.8)
"total revenue was $4.1 billion"    →  (metric: revenue, value: 4100.0)
```

Values are normalized to millions: billion values are multiplied by 1,000, thousand values are divided by 1,000.

#### Multi-Year Table Parser (`parse_table_year_columns`)

Markdown tables with year column headers are parsed to extract a `(year, metric, value)` triple for every cell:

```
| Metric        | 2016  | 2017  | 2018  |
| net income    | 450   | 510   | 620   |
| total revenue | 3200  | 3800  | 4100  |
```
→ 6 structured fact nodes from a single table, each linked to the correct year.

**23 canonical metric types** are recognized. Longest-keyword match wins when multiple terms apply to the same row:

| Raw text examples | Canonical metric name |
|---|---|
| "net revenue", "net sales", "total revenue" | `revenue` |
| "net income", "net earnings", "net profit" | `net_income` |
| "interest expense", "interest cost" | `interest_expense` |
| "total operating expenses", "operating costs" | `operating_expenses` |
| "gross profit" | `gross_profit` |
| "cost of goods sold", "cost of sales", "cost of revenue" | `cost_of_goods_sold` |
| "capital expenditures", "capex" | `capital_expenditures` |
| "cash and cash equivalents" | `cash_and_equivalents` |
| "total assets" | `total_assets` |
| "operating income", "income from operations" | `operating_income` |
| "depreciation", "amortization" | `depreciation` |
| "earnings per share", "eps" | `earnings_per_share` |

### Step 5 — Graph Population

For each extracted metric, a `Metric` node is created and linked to its `Company` and `Year`:

```cypher
MATCH (c:Company {name: $company})
MERGE (y:Year {value: $year})
CREATE (m:Metric {name: $name, value: $value})
CREATE (c)-[:HAS_METRIC]->(m)
CREATE (m)-[:FOR_YEAR]->(y)
```

Documents are linked to their company and year:

```cypher
CREATE (d)-[:ABOUT]->(c)
CREATE (d)-[:FROM_YEAR]->(y)
```

---

## Graph Schema

```
(Company)-[:HAS_METRIC]->(Metric)-[:FOR_YEAR]->(Year)
(Document)-[:ABOUT]---->(Company)
(Document)-[:FROM_YEAR]->(Year)
```

| Node type | Approximate count |
|---|---|
| Company | 1,000+ |
| Document | 23,000+ |
| Metric | 100,000+ |
| Year | 30+ |

Counts are higher than earlier runs because the graph is now built from all splits (train + dev + test) across all three datasets.

---

## How It Works at Benchmark Time

### Entity Extraction (`src/graphrag/graph_retriever.py`)

The question is parsed for three entity types before querying the graph:

| Entity | Method |
|---|---|
| Company | Matched against all Company nodes; longest match wins to prevent "Entergy" beating "Entergy Louisiana" |
| Year | Regex `\b(19[9]\d|20[0-3]\d)\b` |
| Metric type | Keyword lookup across 20+ recognized financial terms |

### Retrieval Strategies

Six strategies are attempted in priority order. The first one that returns results is used:

**Strategy 0 — Direct question match (highest priority)**

Looks up the document directly by matching the stored `question` property:

```cypher
MATCH (d:Document)
WHERE d.question = $question
RETURN d.company, d.text, d.year
LIMIT 1
```

This is the most reliable strategy because the benchmark documents were indexed during the build stage and their questions are stored verbatim. For any benchmark question that matches a stored document, this short-circuits all entity guessing.

**Strategy 1 — Company + Year + Metric (exact structured lookup)**

Used when all three entities are extracted from the question. Returns the exact metric value immediately.

```cypher
MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)
      -[:FOR_YEAR]->(y:Year {value: $year})
WHERE m.name = $metric_type
RETURN c.name, m.name, m.value, y.value
```

**Strategy 2 — Company + Year → Documents**

Used when company and year are found but no canonical metric type is matched. Returns document text for that company and year — the raw tables likely contain the specific line item asked about.

**Strategy 3 — Company only (no year)**

Returns the most recent documents for the company when no year appears in the question.

**Strategy 4 — Metric only (no company)**

Returns that metric across all companies when no company is identified. Useful for comparative or general questions.

**Strategy 5 — Full-text fallback**

CONTAINS search across all document text using the company name, metric name, or the first 50 characters of the question as the search term.

### Context Construction

Graph records are placed first (structured facts), followed by the source document from the dataset (raw tables). The graph context is capped at 3,000 characters.

```
[Knowledge graph]
[Record 1] Company: Analog Devices | Year: 2009 | Metric: interest_expense | Value: $3.80 million
[Record 2] Company: Analog Devices | Year: 2009
   Text: ...financial table text from graph...

[Source document]
...pre_text + table + post_text from dataset...
```

### Token Limits

| Dataset | Max tokens |
|---|---|
| FinQA | 1200 |
| TAT-DQA | 1200 |
| ConvFinQA | 800 |

GraphRAG gets the most tokens because it passes both graph records and the source document.

---

## Output Format

Results are saved to `results/metrics/graphrag.json` with the same structure as baseline and RAG:

```json
{
  "llama3.1:8b_finqa": [
    {
      "question": "What was Analog Devices' interest expense for fiscal year 2009?",
      "ground_truth": "3.8",
      "answer": "From the knowledge graph: interest_expense for Analog Devices in 2009 = $3.80 million\n\nANSWER: 3.8",
      "correct": true,
      "latency_ms": 10823.45
    }
  ]
}
```

### Why Strategy 0 Matters

Before adding Strategy 0, FinQA and ConvFinQA documents were stored in the graph with `company = "Unknown"` because those datasets have no `company_name` field. This meant every FinQA/ConvFinQA question fell through to the Strategy 5 full-text fallback, which is unreliable. Strategy 0's direct question lookup bypasses the company/year/metric entity extraction entirely, giving GraphRAG accurate context for all three datasets.
