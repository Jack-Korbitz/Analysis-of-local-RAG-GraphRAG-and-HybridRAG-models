# RAG vs GraphRAG — Deep Dive

## Vector RAG

### Index time (done once, saved to disk)

Every document from the corpus gets chunked and run through `all-MiniLM-L6-v2`, a small 22M parameter model that converts any text into a fixed **384-number vector**. These vectors encode semantic meaning — sentences about similar topics end up as similar vectors in 384-dimensional space.

```
"Intel revenue was $53.3B in 2012"
        ↓  SentenceTransformer
[0.12, -0.34, 0.07, ..., 0.91]   ← 384 numbers
```

The vectors are stored in a **FAISS `IndexFlatIP`** — a flat array where every vector is compared directly. `IP` = Inner Product, which equals cosine similarity once vectors are L2-normalized (done in `src/rag/vector_store.py`).

### Query time

1. The question is embedded with the **same model** into a 384-number vector
2. FAISS computes dot products between the question vector and every stored document vector
3. Top 8 highest scores (most cosine-similar) are returned

```
"What was Intel's revenue in 2012?"
        ↓  embed
[0.11, -0.31, 0.09, ...]
        ↓  FAISS dot product against all stored vectors
Top 8 chunks by similarity score
```

### Why it can hurt performance here

The similarity search finds text that **sounds like** the question. "Intel revenue" is semantically close to any chunk mentioning Intel or revenue — including Intel's revenue in 2009, 2010, 2011. Those chunks get retrieved alongside the correct 2012 one and introduce competing numbers. The model then has to pick the right one, which it sometimes gets wrong.

---

## GraphRAG

### Index time (done once, loaded into Neo4j)

Annual reports are parsed and loaded into a property graph with a fixed schema:

```
(Company) -[:HAS_METRIC]→ (Metric {name, value}) -[:FOR_YEAR]→ (Year {value})
(Document {text, year}) -[:ABOUT]→ (Company)
```

So "Intel's operating income in 2012 was $14.6B" becomes a node, not floating text.

### Query time — 5-strategy waterfall

The `graph_retriever.py` doesn't do similarity search. It does **entity extraction + structured lookup**:

**Step 1 — Parse the question with regex:**
- Year: `re.search(r'\b(19[9]\d|20[0-3]\d)\b', query)` → `2012`
- Metric: keyword dict match → `revenue`
- Company: scan all Company nodes, find longest substring match in question → `"Intel"`

**Step 2 — Try strategies in order, stop at first hit:**

| Strategy | Cypher query | Used when |
|---|---|---|
| **1** | `MATCH (Company)-[:HAS_METRIC]->(Metric)-[:FOR_YEAR]->(Year)` | Company + year + metric all found |
| **2** | `MATCH (Document)-[:ABOUT]->(Company) WHERE d.year = year` | Company + year, but metric not in the 23 known types |
| **3** | `MATCH (Document)-[:ABOUT]->(Company) ORDER BY year DESC` | Company found, no year in question |
| **4** | `MATCH (Company)-[:HAS_METRIC {name: metric}]` | Metric found, no company |
| **5** | `WHERE toLower(d.text) CONTAINS search_term` | Full-text fallback |

**Strategy 1 is the ideal case** — it returns a single pre-extracted structured record like:

```
Company: Intel | Year: 2012 | Metric: revenue | Value: $53300.00 million
```

No ambiguity, no competing numbers, no table reading required.

### Why it beats baseline

When strategy 1 fires, the model gets a clean fact rather than a raw financial table. It skips the hard part — table parsing — entirely. This is why GraphRAG at 67.8% beats baseline at 56.1%.

### Where it fails

- Strategy 1 only covers **23 metric types** hardcoded in `graph_retriever.py`. Any question about something outside that list (e.g. "stock-based compensation for restricted stock units") falls through to strategy 2 or lower
- Company matching is substring-based — if the question says "Entergy" when the graph has "Entergy Louisiana, LLC", it might match the wrong node
- Year regex only catches 4-digit years — "fiscal 2009" would still work, but more ambiguous phrasings could be missed

---

## Side-by-side Summary

```
Question: "What was Intel's revenue in 2012?"

RAG:                              GraphRAG:
  Embed question → vector           regex: year=2012, metric=revenue
  Search 384-dim space              substring: company="Intel"
  Return 8 similar chunks           Cypher: MATCH (Intel)-[:HAS_METRIC]→
  (some are 2009, 2011 Intel)         (Metric{name:revenue})-[:FOR_YEAR]->(2012)
  Model must pick right number      Return: $53300.00 million
  → fuzzy, error-prone              → exact, no ambiguity
```

| | RAG | GraphRAG |
|---|---|---|
| Storage | FAISS vector index | Neo4j graph database |
| Search method | Cosine similarity on embeddings | Cypher query (company + year + metric) |
| What it returns | Text passages that *seem* related | Structured records that *are* the answer |
| Fails when | Wrong chunks match semantically | Entity extraction fails (wrong company/year parsed) |
| Beats baseline when | — (currently doesn't) | Graph has the exact metric pre-extracted |

The reason GraphRAG wins here is that FinQA questions are highly structured ("what was X's Y in year Z?") — exactly the kind of query a graph database is designed for. RAG's fuzzy matching works better for open-ended questions where you don't know exactly what you're looking for.