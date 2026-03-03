"""
Generate results/answer_viewer.html — a self-contained UI for browsing
every benchmark Q&A with green/red correct-answer highlighting.
"""

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

METRICS_DIR = Path("results/metrics")
OUTPUT_FILE = Path("results/answer_viewer.html")

APPROACH_LABELS = {
    "baseline_fast": "Baseline",
    "rag_fast":      "RAG",
    "graphrag_fast": "GraphRAG",
}


def check_accuracy(answer, ground_truth):
    answer_str = str(answer).lower()
    gt_str = re.sub(r"\.0$", "", str(ground_truth).lower())
    answer_norm = re.sub(r"(\d),(\d)", r"\1\2", answer_str)
    gt_norm     = re.sub(r"(\d),(\d)", r"\1\2", gt_str)

    if gt_norm in answer_norm:
        return True

    answer_numbers = re.findall(r"-?\d+\.?\d*", answer_norm)
    gt_numbers     = re.findall(r"-?\d+\.?\d*", gt_norm)

    if not gt_numbers or not answer_numbers:
        return False

    try:
        gt_val = float(gt_numbers[0])
        for ans_num in answer_numbers:
            ans_val = float(ans_num)
            if gt_val == ans_val:
                return True
            if 0 < abs(gt_val) < 1 and abs(ans_val - gt_val * 100) < 0.5:
                return True
            if 0 < abs(ans_val) < 1 and abs(ans_val * 100 - gt_val) < 0.5:
                return True
            if abs(gt_val) >= 100 and abs(gt_val - ans_val) / abs(gt_val) < 0.01:
                return True
            if abs(gt_val) < 100 and abs(ans_val - gt_val) < 0.5:
                return True
    except ValueError:
        pass

    return False


def load_all_items():
    items = []
    for fname, approach_label in APPROACH_LABELS.items():
        path = METRICS_DIR / f"{fname}.json"
        if not path.exists():
            print(f"  Missing: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key, questions in data.items():
            model, dataset = key.rsplit("_", 1)
            for q in questions:
                items.append({
                    "question":     q["question"],
                    "ground_truth": q["ground_truth"],
                    "answer":       q["answer"],
                    "correct":      check_accuracy(q["answer"], q["ground_truth"]),
                    "latency_ms":   round(q.get("latency_ms", 0)),
                    "model":        model,
                    "dataset":      dataset.upper(),
                    "approach":     approach_label,
                })
    return items


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Answer Viewer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
  }

  /* ── Header ─────────────────────────────────────── */
  header {
    background: #1a1d27;
    border-bottom: 1px solid #2a2d3a;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  header h1 {
    font-size: 17px;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
  }
  .stats {
    font-size: 13px;
    color: #888;
    white-space: nowrap;
  }
  .stats strong { color: #ccc; }

  /* ── Filter bar ──────────────────────────────────── */
  .filters {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    flex: 1;
    align-items: center;
  }
  .filters select, .filters input[type=search] {
    background: #252836;
    border: 1px solid #3a3d4e;
    color: #ddd;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
    outline: none;
    cursor: pointer;
  }
  .filters select:focus, .filters input[type=search]:focus {
    border-color: #5b8dee;
  }
  .filters input[type=search] { min-width: 220px; }

  .btn-reset {
    background: #2e3148;
    border: 1px solid #3a3d4e;
    color: #aaa;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 13px;
    cursor: pointer;
  }
  .btn-reset:hover { background: #383c58; color: #fff; }

  /* ── Card list ───────────────────────────────────── */
  #cards {
    padding: 20px 24px;
    display: grid;
    gap: 14px;
    max-width: 1200px;
    margin: 0 auto;
  }

  .card {
    background: #1a1d27;
    border-radius: 10px;
    border-left: 5px solid #444;
    padding: 16px 18px;
    display: grid;
    gap: 10px;
  }
  .card.correct   { border-left-color: #22c55e; }
  .card.incorrect { border-left-color: #ef4444; }

  /* ── Card sections ───────────────────────────────── */
  .card-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }

  .badge {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 20px;
    letter-spacing: 0.3px;
  }
  .badge-approach-Baseline { background: #2e3148; color: #9da8d0; }
  .badge-approach-RAG      { background: #1b2d4a; color: #5b9bd5; }
  .badge-approach-GraphRAG { background: #1a2e1e; color: #4caf6e; }
  .badge-model  { background: #2a2030; color: #c084fc; }
  .badge-dataset { background: #2a2820; color: #fbbf24; }

  .correct-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    margin-left: auto;
  }
  .correct-badge.yes { background: #14532d; color: #4ade80; }
  .correct-badge.no  { background: #450a0a; color: #f87171; }

  .latency {
    font-size: 11px;
    color: #666;
    white-space: nowrap;
  }

  .question-text {
    font-size: 14px;
    color: #c8d0e0;
    line-height: 1.55;
  }

  .answer-block {
    background: #12141e;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 13px;
    color: #d0d8f0;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
  }
  .answer-block.collapsed { max-height: 80px; }

  .answer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }
  .label {
    font-size: 11px;
    font-weight: 700;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .expand-btn {
    font-size: 11px;
    color: #5b8dee;
    cursor: pointer;
    background: none;
    border: none;
    padding: 0;
  }
  .expand-btn:hover { color: #7aa5f5; }

  .ground-truth {
    font-size: 12px;
    color: #888;
  }
  .ground-truth span { color: #bbb; font-weight: 500; }

  /* ── Empty state ────────────────────────────────── */
  .empty {
    text-align: center;
    padding: 60px 20px;
    color: #555;
    font-size: 15px;
  }
</style>
</head>
<body>

<header>
  <h1>Benchmark Answer Viewer</h1>
  <div class="filters">
    <input type="search" id="q-search" placeholder="Search questions…" oninput="render()">
    <select id="f-approach" onchange="render()">
      <option value="">All Approaches</option>
      <option>Baseline</option>
      <option>RAG</option>
      <option>GraphRAG</option>
    </select>
    <select id="f-model" onchange="render()">
      <option value="">All Models</option>
    </select>
    <select id="f-dataset" onchange="render()">
      <option value="">All Datasets</option>
    </select>
    <select id="f-correct" onchange="render()">
      <option value="">All Results</option>
      <option value="true">Correct only</option>
      <option value="false">Incorrect only</option>
    </select>
    <button class="btn-reset" onclick="resetFilters()">Reset</button>
  </div>
  <div class="stats" id="stats"></div>
</header>

<div id="cards"></div>

<script>
const DATA = __DATA__;

// Populate dynamic filter options
const models   = [...new Set(DATA.map(d => d.model))].sort();
const datasets = [...new Set(DATA.map(d => d.dataset))].sort();

const modelSel   = document.getElementById('f-model');
const datasetSel = document.getElementById('f-dataset');
models.forEach(m => {
  const o = document.createElement('option'); o.value = m; o.textContent = m;
  modelSel.appendChild(o);
});
datasets.forEach(d => {
  const o = document.createElement('option'); o.value = d; o.textContent = d;
  datasetSel.appendChild(o);
});

function getFilters() {
  return {
    search:   document.getElementById('q-search').value.toLowerCase(),
    approach: document.getElementById('f-approach').value,
    model:    document.getElementById('f-model').value,
    dataset:  document.getElementById('f-dataset').value,
    correct:  document.getElementById('f-correct').value,
  };
}

function resetFilters() {
  document.getElementById('q-search').value = '';
  document.getElementById('f-approach').value = '';
  document.getElementById('f-model').value = '';
  document.getElementById('f-dataset').value = '';
  document.getElementById('f-correct').value = '';
  render();
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function render() {
  const f = getFilters();
  const filtered = DATA.filter(d => {
    if (f.approach && d.approach !== f.approach) return false;
    if (f.model    && d.model    !== f.model)    return false;
    if (f.dataset  && d.dataset  !== f.dataset)  return false;
    if (f.correct === 'true'  && !d.correct)     return false;
    if (f.correct === 'false' &&  d.correct)     return false;
    if (f.search && !d.question.toLowerCase().includes(f.search) &&
                    !d.answer.toLowerCase().includes(f.search))  return false;
    return true;
  });

  const correctCount = filtered.filter(d => d.correct).length;
  const stats = document.getElementById('stats');
  stats.innerHTML = `Showing <strong>${filtered.length}</strong> / ${DATA.length} &nbsp;·&nbsp; ` +
    `<strong style="color:#4ade80">${correctCount}</strong> correct &nbsp;·&nbsp; ` +
    `<strong style="color:#f87171">${filtered.length - correctCount}</strong> incorrect`;

  const container = document.getElementById('cards');

  if (filtered.length === 0) {
    container.innerHTML = '<div class="empty">No results match the current filters.</div>';
    return;
  }

  container.innerHTML = filtered.map((d, i) => {
    const cardId  = `card-${i}`;
    const answLen = d.answer.length;
    const long    = answLen > 300;

    return `
<div class="card ${d.correct ? 'correct' : 'incorrect'}" id="${cardId}">
  <div class="card-meta">
    <span class="badge badge-approach-${escHtml(d.approach)}">${escHtml(d.approach)}</span>
    <span class="badge badge-model">${escHtml(d.model)}</span>
    <span class="badge badge-dataset">${escHtml(d.dataset)}</span>
    <span class="latency">${(d.latency_ms/1000).toFixed(1)}s</span>
    <span class="correct-badge ${d.correct ? 'yes' : 'no'}">${d.correct ? '✓ Correct' : '✗ Wrong'}</span>
  </div>

  <div class="question-text">${escHtml(d.question)}</div>

  <div>
    <div class="answer-header">
      <span class="label">Model Answer</span>
      ${long ? `<button class="expand-btn" onclick="toggleAnswer('ans-${i}', this)">expand</button>` : ''}
    </div>
    <div class="answer-block${long ? ' collapsed' : ''}" id="ans-${i}">${escHtml(d.answer)}</div>
  </div>

  <div class="ground-truth">Ground truth: <span>${escHtml(d.ground_truth)}</span></div>
</div>`;
  }).join('');
}

function toggleAnswer(id, btn) {
  const el = document.getElementById(id);
  if (el.classList.contains('collapsed')) {
    el.classList.remove('collapsed');
    btn.textContent = 'collapse';
  } else {
    el.classList.add('collapsed');
    btn.textContent = 'expand';
  }
}

render();
</script>
</body>
</html>
"""


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("Loading results…")
    items = load_all_items()
    print(f"  {len(items)} items loaded")

    json_data = json.dumps(items, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA__", json_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    correct = sum(1 for i in items if i["correct"])
    print(f"  {correct}/{len(items)} correct ({100*correct/len(items):.1f}%)")
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
