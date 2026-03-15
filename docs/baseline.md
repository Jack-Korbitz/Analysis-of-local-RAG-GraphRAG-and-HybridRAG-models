# Baseline Method

## Overview

The baseline is a **closed-book** evaluation. The model receives only the question — no document context, no retrieved chunks, no graph records. It must answer purely from knowledge acquired during pre-training.

This serves as the control condition. Any accuracy above this number reflects the benefit of retrieval.

---

## Build Stage

The baseline has no build stage. There is nothing to index or construct beforehand. Questions are sent directly to the model at benchmark time.

---

## How It Works

### Prompt Construction

The prompt varies by dataset because some tasks require multi-step arithmetic (FinQA, TAT-DQA) while others ask for a single number (ConvFinQA).

**FinQA / TAT-DQA — chain-of-thought prompt:**
```
Financial question: {question}

Show your step-by-step calculation, then write ANSWER: <number>
```

**ConvFinQA — number-only prompt:**
```
Financial question: {question}

Answer (number only):
```

### System Prompt

All datasets share the same financial analyst system prompt:

**FinQA / TAT-DQA:**
```
You are a financial analyst. You will be given a question.

Approach:
1. Identify the exact figures needed from the text or table.
2. Show your calculation step by step.
3. On the final line write: ANSWER: <number>

Rules:
- ANSWER must be a single number (e.g. ANSWER: 3.8 or ANSWER: -2.5).
- Do not include units or currency symbols in the ANSWER line.
- For yes/no questions output ANSWER: 1 for yes and ANSWER: 0 for no.
- Parentheses in financial tables mean negative (e.g. (832) = -832).
- Never say you cannot answer; the data is always in the provided context.
```

**ConvFinQA:**
```
You are a financial analyst. The answer is a specific number or percentage
that can be read or calculated directly from the provided text and table.

Rules:
- Output ONLY the final numeric answer (e.g. 3.8 or -2.5% or 0.532).
- Do not include units, currency symbols, or explanatory text.
- If a calculation is needed, do it silently and output only the result.
- Never say you cannot answer; the answer is always in the provided context.
```

### Token Limits

| Dataset | Max tokens |
|---|---|
| FinQA | 800 |
| TAT-DQA | 800 |
| ConvFinQA | 400 |

ConvFinQA gets fewer tokens because answers are expected to be a single number with no reasoning chain.

### qwen3:8b Special Handling

qwen3 uses internal chain-of-thought ("thinking") tokens that consume the token budget before producing any visible output. Two mechanisms suppress this:

1. `think=False` passed as a direct parameter to `ollama.chat()` (supported in Ollama ≥ 0.6)
2. `/no_think` prepended to the system prompt as a fallback for older Ollama versions
3. Any leaked `<think>...</think>` blocks are stripped from the response with regex

---

## Output Format

Each completed question produces one record in `results/metrics/baseline_fast.json`:

```json
{
  "question": "What was Analog Devices' interest expense for fiscal year 2009?",
  "ground_truth": "3.8",
  "answer": "I don't have access to Analog Devices' 2009 annual report...",
  "correct": false,
  "latency_ms": 4821.34
}
```

Results are grouped by `{model}_{dataset}` key:
```json
{
  "llama3.1:8b_finqa": [ ... ],
  "gemma3:12b_finqa":  [ ... ],
  "qwen3:8b_finqa":    [ ... ],
  "llama3.1:8b_convfinqa": [ ... ],
  ...
}
```

### Expected Accuracy

Because financial QA questions ask for exact figures from specific filings (e.g. a company's interest expense in a particular fiscal year), models are unlikely to have memorized the exact values. Expected closed-book accuracy is **10–20%**, primarily reflecting questions where the answer can be reasoned from general financial knowledge or the model happens to recall the figure.
