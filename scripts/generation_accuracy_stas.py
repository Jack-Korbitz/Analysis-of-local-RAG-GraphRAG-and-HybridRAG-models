"""Verify all statistics reported in the generated paper against raw JSON data."""

import json
import re
from pathlib import Path

METRICS_DIR = Path("results/metrics")
APPROACHES = ['baseline', 'oracle', 'rag', 'graphrag', 'hybrid']
LABELS = {
    'baseline': 'Baseline', 'oracle': 'Oracle', 'rag': 'RAG',
    'graphrag': 'GraphRAG', 'hybrid': 'HybridRAG',
}


def check_acc(answer, gt):
    a = re.sub(r'(\d),(\d)', r'\1\2', str(answer).lower())
    g = re.sub(r'(\d),(\d)', r'\1\2', re.sub(r'\.0$', '', str(gt).lower()))
    if g and re.search(r'(?<!\d)' + re.escape(g) + r'(?!\d)', a):
        return True
    ans_nums = re.findall(r'-?\d+\.?\d*', a)
    gt_nums = re.findall(r'-?\d+\.?\d*', g)
    if not gt_nums or not ans_nums:
        return False
    try:
        gv = float(gt_nums[0])
        for an in ans_nums:
            av = float(an)
            if gv == av: return True
            if 0 < abs(gv) < 1 and abs(av - gv * 100) < 0.5: return True
            if 0 < abs(av) < 1 and abs(av * 100 - gv) < 0.5: return True
            if abs(gv) >= 100 and abs(gv - av) / abs(gv) < 0.01: return True
            if 1 <= abs(gv) < 100 and abs(av - gv) < 0.5: return True
            if 0 < abs(gv) < 1 and abs(gv - av) / abs(gv) < 0.01: return True
            if abs(gv) >= 1000:
                for scale in [1_000, 1_000_000]:
                    if abs(av * scale - gv) / abs(gv) < 0.01: return True
    except ValueError:
        pass
    return False


def strict_em(answer, gt):
    a = re.sub(r'(\d),(\d)', r'\1\2', str(answer).strip().lower())
    g = re.sub(r'(\d),(\d)', r'\1\2', re.sub(r'\.0$', '', str(gt).strip().lower()))
    return g != '' and g in a


def load_results():
    results = {}
    for approach in APPROACHES:
        p = METRICS_DIR / f"{approach}.json"
        if p.exists():
            results[approach] = json.loads(p.read_text(encoding='utf-8'))
    return results


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    results = load_results()
    print(f"Loaded: {[LABELS[a] for a in results]}")

    # ── 6.1 Overall Accuracy by Approach ──
    section("6.1 Overall Accuracy by Approach")
    print(f"{'Approach':<12} {'Accuracy':>10} {'Avg Latency':>12} {'Questions':>10}")
    print('-' * 48)
    for approach in APPROACHES:
        if approach not in results:
            continue
        all_q = [q for qs in results[approach].values() for q in qs]
        n = len(all_q)
        correct = sum(check_acc(q['answer'], q['ground_truth']) for q in all_q)
        pct = correct / n * 100 if n else 0
        avg_lat = sum(q['latency_ms'] for q in all_q) / n / 1000 if n else 0
        print(f"{LABELS[approach]:<12} {pct:>9.1f}% {avg_lat:>10.1f}s {n:>10}")

    # ── 6.2 Per-Model Accuracy ──
    section("6.2 Per-Model Accuracy")
    header = f"{'Model':<14}" + "".join(f"{LABELS[a]:>12}" for a in APPROACHES if a in results)
    print(header)
    print('-' * len(header))
    for model in ['llama3.1:8b', 'gemma3:12b', 'qwen3:8b']:
        row = f"{model:<14}"
        for approach in APPROACHES:
            if approach not in results:
                continue
            qs = [q for k, v in results[approach].items() if k.rsplit('_', 1)[0] == model for q in v]
            if not qs:
                row += f"{'n/a':>12}"
            else:
                correct = sum(check_acc(q['answer'], q['ground_truth']) for q in qs)
                row += f"{correct / len(qs) * 100:>11.1f}%"
        print(row)

    # ── 6.3 Per-Dataset Accuracy ──
    section("6.3 Per-Dataset Accuracy")
    header = f"{'Dataset':<12}" + "".join(f"{LABELS[a]:>12}" for a in APPROACHES if a in results)
    print(header)
    print('-' * len(header))
    for ds in ['finqa', 'convfinqa', 'tatqa']:
        row = f"{ds.upper():<12}"
        for approach in APPROACHES:
            if approach not in results:
                continue
            qs = [q for k, v in results[approach].items() if k.endswith(f'_{ds}') for q in v]
            if not qs:
                row += f"{'n/a':>12}"
            else:
                correct = sum(check_acc(q['answer'], q['ground_truth']) for q in qs)
                row += f"{correct / len(qs) * 100:>11.1f}%"
        print(row)

    # ── 6.4 Retrieval Quality ──
    section("6.4 Retrieval Quality")
    for approach in ['rag', 'graphrag', 'hybrid']:
        if approach not in results:
            continue
        rm_qs = [q.get('retrieval_metrics', {}) for v in results[approach].values()
                 for q in v if q.get('retrieval_metrics')]
        avg_ret = sum(r.get('retrieved_count', 0) for r in rm_qs) / len(rm_qs) if rm_qs else 0
        sims = [r['avg_similarity'] for r in rm_qs if r.get('avg_similarity')]
        avg_sim = sum(sims) / len(sims) if sims else 0
        aic = sum(1 for r in rm_qs if r.get('answer_in_context')) / len(rm_qs) * 100 if rm_qs else 0
        print(f"{LABELS[approach]:<12}  avg_retrieved={avg_ret:.1f}  avg_sim={avg_sim:.3f}  answer_in_context={aic:.1f}%")

    # ── 6.5 Strict EM vs Relaxed ──
    section("6.5 Strict Exact Match")
    print(f"{'Approach':<12} {'Relaxed':>10} {'Strict EM':>10} {'Gap':>8}")
    print('-' * 44)
    for approach in APPROACHES:
        if approach not in results:
            continue
        all_q = [q for qs in results[approach].values() for q in qs]
        n = len(all_q) or 1
        relaxed = sum(check_acc(q['answer'], q['ground_truth']) for q in all_q) / n * 100
        strict = sum(strict_em(q['answer'], q['ground_truth']) for q in all_q) / n * 100
        gap = relaxed - strict
        print(f"{LABELS[approach]:<12} {relaxed:>9.1f}% {strict:>9.1f}% {'+' + f'{gap:.1f}':>7}pp")

    # ── 6.6 Generator Conditional Accuracy ──
    section("6.6 Generator Conditional Accuracy")
    print(f"{'Approach':<12} {'Overall':>10} {'In-Context':>18} {'Not-in-Context':>18}")
    print('-' * 62)
    for approach in ['oracle', 'rag', 'graphrag', 'hybrid']:
        if approach not in results:
            continue
        all_q = [q for qs in results[approach].values() for q in qs]
        n = len(all_q) or 1
        ov = sum(check_acc(q['answer'], q['ground_truth']) for q in all_q) / n * 100
        aic_qs = [q for q in all_q if q.get('retrieval_metrics', {}).get('answer_in_context')]
        non_qs = [q for q in all_q if not q.get('retrieval_metrics', {}).get('answer_in_context')]
        ci = sum(check_acc(q['answer'], q['ground_truth']) for q in aic_qs) / len(aic_qs) * 100 if aic_qs else 0
        co = sum(check_acc(q['answer'], q['ground_truth']) for q in non_qs) / len(non_qs) * 100 if non_qs else 0
        print(f"{LABELS[approach]:<12} {ov:>9.1f}% {ci:>9.1f}% (n={len(aic_qs):<3}) {co:>9.1f}% (n={len(non_qs):<3})")

    # ── 6.7 GraphRAG Stratified Accuracy ──
    section("6.7 GraphRAG Stratified Accuracy")
    if 'graphrag' in results:
        strat_acc = {}
        for qs in results['graphrag'].values():
            for q in qs:
                s = q.get('retrieval_metrics', {}).get('retrieval_strategy', 'unknown')
                if s not in strat_acc:
                    strat_acc[s] = {'correct': 0, 'total': 0, 'aic': 0}
                strat_acc[s]['total'] += 1
                if check_acc(q['answer'], q['ground_truth']):
                    strat_acc[s]['correct'] += 1
                if q.get('retrieval_metrics', {}).get('answer_in_context'):
                    strat_acc[s]['aic'] += 1

        print(f"{'Strategy':<28} {'Questions':>10} {'Accuracy':>10} {'AiC':>8}")
        print('-' * 60)
        for s in sorted(strat_acc, key=lambda x: -strat_acc[x]['total']):
            d = strat_acc[s]
            acc = d['correct'] / d['total'] * 100 if d['total'] else 0
            aic = d['aic'] / d['total'] * 100 if d['total'] else 0
            print(f"{s:<28} {d['total']:>10} {acc:>9.1f}% {aic:>7.1f}%")
    else:
        print("No GraphRAG data found.")

    # ── GraphRAG Strategy Distribution ──
    section("GraphRAG Strategy Distribution")
    if 'graphrag' in results:
        strats = {}
        for qs in results['graphrag'].values():
            for q in qs:
                s = q.get('retrieval_metrics', {}).get('retrieval_strategy', 'unknown')
                strats[s] = strats.get(s, 0) + 1
        total = sum(strats.values())
        for s in sorted(strats, key=lambda x: -strats[x]):
            print(f"  {s:<28} {strats[s]:>4}  ({strats[s] / total * 100:.0f}%)")


if __name__ == '__main__':
    main()
