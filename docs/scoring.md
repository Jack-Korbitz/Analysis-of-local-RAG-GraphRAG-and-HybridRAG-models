# Accuracy Scoring

## Overview

Financial QA answers have significant formatting variation — a model may write `$41,932 million`, `41932.2`, `41.9 billion`, or `≈41,932` for the same ground truth value. A single exact-string match would miss most correct answers.

A tolerance-based scorer is applied consistently across all scripts: `compare_all_runs.py`, `view_answers.py`, and `visualize_results.py`.

---

## Scoring Logic

Checks are applied in order. The answer is marked correct as soon as any check passes.

### 1. Pre-processing

Both the answer and ground truth are normalized before any comparison:

- Lowercase both strings
- Strip trailing `.0` from the ground truth only (e.g. `"2530454.0"` → `"2530454"`)
  - Uses `re.sub(r'\.0$', ...)` — NOT `.replace('.0', '')` which would corrupt `-0.032` into `-032`
- Strip commas from numbers in both strings (`41,932` → `41932`)

### 2. Word-Boundary String Match

```python
re.search(r'(?<!\d)' + re.escape(gt_norm) + r'(?!\d)', answer_norm)
```

The ground truth value must appear in the answer without adjacent digits. This prevents `531` from matching inside `-531` or `2531`. Correct if the normalized ground truth string is found as a standalone token.

### 3. Numeric Extraction

All numbers are extracted from both strings using `re.findall(r'-?\d+\.?\d*', ...)`. If either string yields no numbers, the answer is marked incorrect.

### 4. Boolean Yes/No (GT = 0 or 1)

Some questions have a binary ground truth (`1` = yes, `0` = no). Natural language responses are mapped to numeric:

| Ground truth | Matches answer containing |
|---|---|
| `1` (yes) | `yes`, `true`, `did`, `exceeded`, `greater`, `more`, `higher` |
| `0` (no) | `no`, `false`, `did not`, `not exceed`, `less`, `lower`, `neither` |

### 5. Exact Numeric Match

`gt_val == ans_val` after parsing both as floats.

### 6. Percentage ↔ Decimal Conversion

Models sometimes report a percentage as a decimal or vice versa:

| Ground truth | Model output | Match condition |
|---|---|---|
| `0.35` (decimal) | `35` (percent) | `abs(ans - gt × 100) < 0.5` |
| `35` (percent) | `0.35` (decimal) | `abs(ans × 100 - gt) < 0.5` |

### 7. Large Number Tolerance (≥ 100)

Within 1% relative tolerance:
```python
abs(gt_val) >= 100 and abs(gt_val - ans_val) / abs(gt_val) < 0.01
```

Handles rounding differences in large financial figures (e.g. `41932` vs `41931.8`).

### 8. Mid-Range Absolute Tolerance (1 – 99)

Within ± 0.5 absolute:
```python
1 <= abs(gt_val) < 100 and abs(ans_val - gt_val) < 0.5
```

### 9. Small Decimal Relative Tolerance (< 1)

Within 1% relative tolerance for values less than 1:
```python
abs(gt_val) < 1 and abs(gt_val - ans_val) / abs(gt_val) < 0.01
```

This is intentionally tighter than ± 0.5 absolute — applying ± 0.5 to small decimals would mean `0.9` incorrectly matches `0.9765625`.

### 10. Unit Scale Mismatch

Models sometimes report in millions when the ground truth is in raw units (or vice versa). For ground truth values ≥ 1,000, both 1,000× and 1,000,000× scaled versions of the model's answer are tested within 1% tolerance:

```python
abs(gt_val) >= 1000:
    for scale in [1_000, 1_000_000]:
        if abs(ans_val * scale - gt_val) / abs(gt_val) < 0.01:
            return True
```

---

## Inline Scoring vs Rescoring

The `correct` field stored in the JSON result files is computed at benchmark run time using `_is_correct()` in `run_parallel_benchmarks.py`. This function additionally uses `_extract_answer_number()` which preprocesses the answer (strips `$`, removes commas, converts `"3.8 billion"` to `3800.0`) before numeric comparison.

The `check_accuracy()` function in `compare_all_runs.py`, `view_answers.py`, and `visualize_results.py` performs the same logical checks but operates on raw strings without the number-extraction preprocessing. In practice the results agree closely, but there can be small differences for answers that mention billion/million multipliers without an explicit number tag.

---

## Examples

| Ground truth | Model answer | Result | Reason |
|---|---|---|---|
| `3.8` | `"interest expense was $3.8 million"` | Correct | String containment |
| `41932.2` | `"ANSWER: 41932"` | Correct | Large number within 1% |
| `0.532` | `"53.2%"` | Correct | Percentage ↔ decimal |
| `-0.032` | `"-3.2%"` | Correct | Percentage ↔ decimal |
| `1` | `"Yes, it exceeded..."` | Correct | Boolean yes/no |
| `531` | `"-531"` | Incorrect | Word-boundary check blocks partial match |
| `3.8` | `"approximately 4"` | Incorrect | Outside ± 0.5 for range 1–99 |
