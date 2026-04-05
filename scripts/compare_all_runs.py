import sys
import re
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


def load_all_results():
    """Load all result files from different benchmark runs"""
    output_dir = Path("results/metrics")
    
    result_files = {
        'Current Models': {
            'baseline': output_dir / "baseline.json",
            'oracle': output_dir / "oracle.json",
            'rag': output_dir / "rag.json",
            'graphrag': output_dir / "graphrag.json",
            'hybrid': output_dir / "hybrid.json"
        }
    }
    
    all_results = {}
    
    for run_name, files in result_files.items():
        run_results = {}
        for approach, path in files.items():
            if path.exists():
                with open(path, encoding='utf-8') as f:
                    run_results[approach] = json.load(f)
        
        if run_results:
            all_results[run_name] = run_results
    
    return all_results


def aggregate_by_model(results_dict):
    aggregated = {}
    for key, questions in results_dict.items():
        model = key.rsplit('_', 1)[0]
        if model not in aggregated:
            aggregated[model] = []
        aggregated[model].extend(questions)
    return aggregated


def check_accuracy(answer, ground_truth, question=''):
    answer_str = str(answer).lower()
    # Strip only trailing ".0" (e.g. "22929.0" → "22929").
    # Do NOT use .replace('.0','') — that corrupts values like "-0.032..." → "-032..."
    gt_str = re.sub(r'\.0$', '', str(ground_truth).lower())

    # Normalize: strip commas from numbers (41,932 → 41932)
    answer_norm = re.sub(r'(\d),(\d)', r'\1\2', answer_str)
    gt_norm = re.sub(r'(\d),(\d)', r'\1\2', gt_str)

    # Whole-number string match (word-boundary safe — avoids "531" matching "-531" or "1" matching "2013")
    if gt_norm and re.search(r'(?<!\d)' + re.escape(gt_norm) + r'(?!\d)', answer_norm):
        return True

    # Extract all numbers from both strings
    answer_numbers = re.findall(r'-?\d+\.?\d*', answer_norm)
    gt_numbers = re.findall(r'-?\d+\.?\d*', gt_norm)

    if not gt_numbers or not answer_numbers:
        return False

    try:
        gt_val = float(gt_numbers[0])

        # Boolean yes/no: GT is 0 or 1 — map natural language to numeric
        if gt_val in (0.0, 1.0):
            if gt_val == 1.0 and re.search(r'\b(yes|true|did|exceeded?|greater|more|higher)\b', answer_str):
                return True
            if gt_val == 0.0 and re.search(r'\b(no|false|did not|not exceed|less|lower|neither)\b', answer_str):
                return True

        for ans_num in answer_numbers:
            ans_val = float(ans_num)

            # Exact numeric match
            if gt_val == ans_val:
                return True

            # Percentage ↔ decimal (gt=0.35 matches ans=35, or gt=35 matches ans=0.35)
            if 0 < abs(gt_val) < 1 and abs(ans_val - gt_val * 100) < 0.5:
                return True
            if 0 < abs(ans_val) < 1 and abs(ans_val * 100 - gt_val) < 0.5:
                return True

            # Relative tolerance for large numbers (within 1%)
            if abs(gt_val) >= 100 and abs(gt_val - ans_val) / abs(gt_val) < 0.01:
                return True

            # Absolute tolerance for mid-range numbers 1–99 (within 0.5)
            if 1 <= abs(gt_val) < 100 and abs(ans_val - gt_val) < 0.5:
                return True

            # Relative tolerance for small decimals <1 (within 1%)
            # ±0.5 is too loose here — e.g. 0.9 would match GT=0.9765625
            if abs(gt_val) < 1 and gt_val != 0 and abs(gt_val - ans_val) / abs(gt_val) < 0.01:
                return True

            # Unit scale: model strips "million"/"thousand" suffix — ans may be 1000x/1M× smaller
            if abs(gt_val) >= 1000:
                for scale in [1_000, 1_000_000]:
                    if abs(ans_val * scale - gt_val) / abs(gt_val) < 0.01:
                        return True

    except ValueError:
        pass

    return False


def strict_exact_match(answer, ground_truth):
    """Strict EM: normalize commas and case only, no numeric tolerance."""
    ans_norm = re.sub(r'(\d),(\d)', r'\1\2', str(answer).strip().lower())
    gt_norm = re.sub(r'\.0$', '', re.sub(r'(\d),(\d)', r'\1\2', str(ground_truth).strip().lower()))
    if not gt_norm:
        return False
    return bool(re.search(r'(?<!\d)' + re.escape(gt_norm) + r'(?!\d)', ans_norm))


def calculate_accuracy_metrics(results_dict):
    accuracy = {}
    for model, questions in results_dict.items():
        correct = 0
        total = len(questions)
        
        for q in questions:
            if check_accuracy(q['answer'], q['ground_truth'], q.get('question', '')):
                correct += 1
        
        accuracy[model] = {
            'correct': correct,
            'total': total,
            'accuracy': (correct / total * 100) if total > 0 else 0
        }
    
    return accuracy


def aggregate_retrieval_metrics(results_dict):
    """Aggregate per-question retrieval_metrics from RAG or GraphRAG results."""
    counts, sims, in_ctx, strategies = [], [], [], {}
    for questions in results_dict.values():
        for q in questions:
            rm = q.get('retrieval_metrics')
            if not rm:
                continue
            counts.append(rm.get('retrieved_count', 0))
            if 'avg_similarity' in rm and rm['avg_similarity'] is not None:
                sims.append(rm['avg_similarity'])
            if rm.get('answer_in_context') is not None:
                in_ctx.append(rm['answer_in_context'])
            strat = rm.get('retrieval_strategy')
            if strat:
                strategies[strat] = strategies.get(strat, 0) + 1
    return {
        'avg_retrieved': round(sum(counts) / len(counts), 2) if counts else 0,
        'avg_similarity': round(sum(sims) / len(sims), 3) if sims else None,
        'answer_in_context_pct': round(sum(in_ctx) / len(in_ctx) * 100, 1) if in_ctx else None,
        'strategy_dist': strategies,
        'n': len(counts)
    }


_ENTITY_STRATEGIES = frozenset({'company_year_metric', 'document_search', 'company_only', 'metric_only', 'metric_fuzzy'})


def print_retrieval_metrics(all_results):
    print_header("RETRIEVAL QUALITY METRICS  (closed-book run)")

    for run_name, results in all_results.items():
        if 'rag' not in results and 'graphrag' not in results and 'hybrid' not in results:
            continue

        print(f"\n{run_name}")
        print("-" * 120)

        if 'rag' in results:
            m = aggregate_retrieval_metrics(results['rag'])
            if m['n'] > 0:
                sim_str = f"{m['avg_similarity']:.3f}" if m['avg_similarity'] is not None else "n/a"
                aic_str = f"{m['answer_in_context_pct']:.1f}%" if m['answer_in_context_pct'] is not None else "n/a"
                print(f"  Vector RAG  | chunks retrieved: {m['avg_retrieved']:.1f} avg  "
                      f"| avg cosine sim: {sim_str}  "
                      f"| answer-in-context: {aic_str}  (n={m['n']})")

        if 'graphrag' in results:
            m = aggregate_retrieval_metrics(results['graphrag'])
            if m['n'] > 0:
                aic_str = f"{m['answer_in_context_pct']:.1f}%" if m['answer_in_context_pct'] is not None else "n/a"
                print(f"  GraphRAG    | records retrieved: {m['avg_retrieved']:.1f} avg  "
                      f"| answer-in-context: {aic_str}  (n={m['n']})")
                if m['strategy_dist']:
                    total = sum(m['strategy_dist'].values())
                    dist_str = "  |  ".join(
                        f"{k}: {v/total*100:.0f}%" for k, v in sorted(m['strategy_dist'].items(), key=lambda x: -x[1])
                    )
                    print(f"              strategy mix -> {dist_str}")
                    entity_matched = sum(
                        v for k, v in m['strategy_dist'].items() if k in _ENTITY_STRATEGIES
                    )
                    cov_pct = entity_matched / total * 100 if total else 0
                    print(f"              entity coverage -> {cov_pct:.1f}%  ({entity_matched} entity-matched / {total} total)")

        if 'hybrid' in results:
            m = aggregate_retrieval_metrics(results['hybrid'])
            if m['n'] > 0:
                sim_str = f"{m['avg_similarity']:.3f}" if m['avg_similarity'] is not None else "n/a"
                aic_str = f"{m['answer_in_context_pct']:.1f}%" if m['answer_in_context_pct'] is not None else "n/a"
                print(f"  Hybrid      | items retrieved: {m['avg_retrieved']:.1f} avg  "
                      f"| avg cosine sim: {sim_str}  "
                      f"| answer-in-context: {aic_str}  (n={m['n']})")


def get_model_size(model_name):
    """Extract parameter count from model name"""
    import re
    match = re.search(r'(\d+)b', model_name.lower())
    if match:
        return int(match.group(1))
    return 0


def print_header(title):
    print("\n" + "="*120)
    print(f"{title:^120}")
    print("="*120)


def _acc(results_dict, approach):
    if approach not in results_dict:
        return None
    agg = aggregate_by_model(results_dict[approach])
    acc = calculate_accuracy_metrics(agg)
    total = sum(v['total'] for v in acc.values())
    return sum(v['correct'] for v in acc.values()) / total * 100 if total else None


def _lat(results_dict, approach):
    if approach not in results_dict:
        return None
    agg = aggregate_by_model(results_dict[approach])
    per_model = [sum(r['latency_ms'] for r in v) / len(v) for v in agg.values() if v]
    return sum(per_model) / len(per_model) if per_model else None


def print_all_runs_comparison(all_results):
    print_header("BENCHMARK RESULTS")

    approaches = ['baseline', 'oracle', 'rag', 'graphrag', 'hybrid']
    labels     = ['Baseline', 'Oracle', 'RAG', 'GraphRAG', 'Hybrid']

    print("\nACCURACY BY APPROACH:")
    header = f"{'Run Name':<30}" + "".join(f"{l:<16}" for l in labels) + "Best"
    print(header)
    print("-" * 140)

    for run_name, results in all_results.items():
        accs = {a: _acc(results, a) for a in approaches}
        row = f"{run_name:<30}"
        candidates = []
        for a, l in zip(approaches, labels):
            if a in results and accs[a] is not None:
                row += f"{accs[a]:>8.1f}%      "
                candidates.append((l, accs[a]))
            else:
                row += f"{'n/a':<16}"
        best = max(candidates, key=lambda x: x[1])[0] if candidates else "n/a"
        print(row + best)

    print("\n" + "=" * 140)
    print("LATENCY BY APPROACH:")
    header = f"{'Run Name':<30}" + "".join(f"{l+' ms':<16}" for l in labels) + "Fastest"
    print(header)
    print("-" * 140)

    for run_name, results in all_results.items():
        lats = {a: _lat(results, a) for a in approaches}
        row = f"{run_name:<30}"
        candidates = []
        for a, l in zip(approaches, labels):
            if a in results and lats[a] is not None:
                row += f"{lats[a]:>10.0f}ms    "
                candidates.append((l, lats[a]))
            else:
                row += f"{'n/a':<16}"
        fastest = min(candidates, key=lambda x: x[1])[0] if candidates else "n/a"
        print(row + fastest)


def print_detailed_model_comparison(all_results):
    print_header("DETAILED MODEL COMPARISON")

    approaches = ['baseline', 'oracle', 'rag', 'graphrag', 'hybrid']
    labels     = ['Baseline', 'Oracle', 'RAG', 'GraphRAG', 'Hybrid']

    for run_name, results in all_results.items():
        print(f"\n{run_name.upper()}")
        print("-" * 140)

        aggs = {a: aggregate_by_model(results[a]) for a in approaches if a in results}
        accs = {a: calculate_accuracy_metrics(agg) for a, agg in aggs.items()}

        all_models = sorted({m for agg in aggs.values() for m in agg})

        print(f"\n{'Model':<20} " + "  ".join(f"{l:<24}" for l, a in zip(labels, approaches) if a in aggs) + "  Best")
        print("-" * 120)

        for model in all_models:
            parts = []
            candidates = []
            for a, l in zip(approaches, labels):
                if a not in aggs or model not in aggs[a]:
                    parts.append(f"{'n/a':<24}")
                    continue
                pct = accs[a][model]['accuracy']
                lat = sum(r['latency_ms'] for r in aggs[a][model]) / len(aggs[a][model])
                parts.append(f"{pct:>5.1f}% ({lat:>6.0f}ms)    ")
                candidates.append((l, pct))
            best = max(candidates, key=lambda x: x[1])[0] if candidates else "n/a"
            print(f"{model:<20} " + " ".join(parts) + f"  {best}")


def print_strict_em_comparison(all_results):
    print("\n" + "="*120)
    print(f"{'STRICT EXACT MATCH vs RELAXED SCORING':^120}")
    print("="*120)
    print("Gap shows how much numeric tolerance inflates the score.\n")

    approaches = [('baseline','Baseline'), ('oracle','Oracle'),
                  ('rag','RAG'), ('graphrag','GraphRAG'), ('hybrid','Hybrid')]

    for run_name, results in all_results.items():
        print(f"\n{run_name.upper()}")
        print(f"{'Approach':<14}  {'Model':<22}  {'Relaxed':>9}  {'Strict EM':>10}  {'Gap':>7}")
        print("-"*70)
        for approach, label in approaches:
            if approach not in results:
                continue
            agg = {}
            for key, qs in results[approach].items():
                model = key.rsplit('_', 1)[0]
                if model not in agg:
                    agg[model] = []
                agg[model].extend(qs)
            for model in sorted(agg.keys()):
                qs = agg[model]
                if not qs:
                    continue
                relaxed = sum(check_accuracy(q['answer'], q['ground_truth']) for q in qs) / len(qs) * 100
                strict = sum(strict_exact_match(q['answer'], q['ground_truth']) for q in qs) / len(qs) * 100
                gap = relaxed - strict
                print(f"{label:<14}  {model:<22}  {relaxed:>8.1f}%  {strict:>9.1f}%  {gap:>+6.1f}%")


def compute_mcnemar(results_a, results_b):
    """McNemar's test (Yates-corrected) between two matched result lists."""
    try:
        from scipy.stats import chi2 as chi2_dist
    except ImportError:
        return None

    a_map = {r['question']: check_accuracy(r['answer'], r['ground_truth']) for r in results_a}
    b_map = {r['question']: check_accuracy(r['answer'], r['ground_truth']) for r in results_b}
    common = set(a_map) & set(b_map)
    if len(common) < 5:
        return None

    n01 = n10 = 0
    for q in common:
        a_c, b_c = a_map[q], b_map[q]
        if not a_c and b_c:   n01 += 1
        elif a_c and not b_c: n10 += 1

    b, c = n01, n10
    if b + c == 0:
        return {'chi2': 0.0, 'p_value': 1.0, 'n': len(common), 'b': b, 'c': c}

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = chi2_dist.sf(stat, df=1)
    return {'chi2': round(stat, 3), 'p_value': round(p_val, 4), 'n': len(common), 'b': b, 'c': c}


def print_mcnemar_table(all_results):
    print("\n" + "="*120)
    print(f"{'McNEMAR\'S TEST — STATISTICAL SIGNIFICANCE':^120}")
    print("="*120)
    print("Baseline vs RAG and Baseline vs GraphRAG. * p<0.05  ** p<0.01  *** p<0.001  ns=not significant\n")

    for run_name, results in all_results.items():
        if 'baseline' not in results:
            continue
        comparisons = [(a, l) for a, l in [('rag','RAG'),('graphrag','GraphRAG'),('hybrid','Hybrid')] if a in results]
        if not comparisons:
            continue

        print(f"\n{run_name.upper()}")
        print(f"{'Key':<30}", end="")
        for _, l in comparisons:
            print(f"  {'Baseline vs '+l:<35}", end="")
        print()
        print("-"*100)

        for key in sorted(results['baseline'].keys()):
            base_qs = results['baseline'].get(key, [])
            if not base_qs:
                continue
            print(f"{key:<30}", end="")
            for comp_approach, _ in comparisons:
                comp_qs = results.get(comp_approach, {}).get(key, [])
                if not comp_qs:
                    print(f"  {'n/a':<35}", end="")
                    continue
                r = compute_mcnemar(base_qs, comp_qs)
                if r is None:
                    print(f"  {'insufficient data':<35}", end="")
                    continue
                sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
                direction = 'comp>base' if r['b'] > r['c'] else 'base>comp' if r['c'] > r['b'] else 'tied'
                cell = f"χ²={r['chi2']:.2f} p={r['p_value']:.3f}{sig} [{direction}]"
                print(f"  {cell:<35}", end="")
            print()


def compute_generator_conditional_accuracy(results_dict):
    """Accuracy conditioned on answer_in_context == True."""
    per_model = {}
    for key, questions in results_dict.items():
        model = key.rsplit('_', 1)[0]
        if model not in per_model:
            per_model[model] = {'cond_correct': 0, 'in_ctx': 0, 'correct': 0, 'total': 0}
        for q in questions:
            per_model[model]['total'] += 1
            correct = check_accuracy(q['answer'], q['ground_truth'])
            in_ctx = q.get('retrieval_metrics', {}).get('answer_in_context', False)
            if correct:
                per_model[model]['correct'] += 1
            if in_ctx:
                per_model[model]['in_ctx'] += 1
                if correct:
                    per_model[model]['cond_correct'] += 1
    out = {}
    for model, d in per_model.items():
        out[model] = {
            'cond_acc': (d['cond_correct'] / d['in_ctx'] * 100) if d['in_ctx'] > 0 else None,
            'in_ctx': d['in_ctx'],
            'total': d['total'],
            'uncond_acc': (d['correct'] / d['total'] * 100) if d['total'] > 0 else None,
        }
    return out


def print_generator_conditional_accuracy(all_results):
    print("\n" + "="*120)
    print(f"{'GENERATOR CONDITIONAL ACCURACY  (correct | answer was in context)':^120}")
    print("="*120)
    print("If conditional acc >> unconditional acc: retrieval is the bottleneck.")
    print("If conditional acc ≈ unconditional acc: model struggles to extract even when context is right.\n")

    for run_name, results in all_results.items():
        print(f"\n{run_name}")
        print(f"  {'Model':<22}  {'Approach':<12}  {'In-Ctx/Total':>14}  {'In-Ctx %':>9}  {'Cond Acc':>9}  {'Uncond Acc':>11}")
        print("  " + "-"*85)
        for approach, label in [('rag','Vector RAG'), ('graphrag','GraphRAG'), ('hybrid','Hybrid'), ('oracle','Oracle')]:
            if approach not in results:
                continue
            per_model = compute_generator_conditional_accuracy(results[approach])
            for model in sorted(per_model.keys()):
                d = per_model[model]
                in_ctx_pct = d['in_ctx'] / d['total'] * 100 if d['total'] else 0
                cond_str = f"{d['cond_acc']:.1f}%" if d['cond_acc'] is not None else "n/a"
                uncond_str = f"{d['uncond_acc']:.1f}%" if d['uncond_acc'] is not None else "n/a"
                print(f"  {model:<22}  {label:<12}  {d['in_ctx']:>6}/{d['total']:<6}  "
                      f"{in_ctx_pct:>8.1f}%  {cond_str:>9}  {uncond_str:>11}")


def print_graphrag_stratified_accuracy(all_results):
    print("\n" + "="*120)
    print(f"{'GRAPHRAG: ACCURACY BY RETRIEVAL STRATEGY + ENTITY COVERAGE':^120}")
    print("="*120)

    for run_name, results in all_results.items():
        if 'graphrag' not in results:
            continue

        graphrag = results['graphrag']
        print(f"\n{run_name}")

        # Entity coverage
        total_matched = total_fallback = total_all = 0
        per_dataset_strat = {}

        for key, questions in graphrag.items():
            dataset = key.rsplit('_', 1)[1]
            for q in questions:
                strat = q.get('retrieval_metrics', {}).get('retrieval_strategy', 'none')
                correct = check_accuracy(q['answer'], q['ground_truth'])
                total_all += 1
                if strat in _ENTITY_STRATEGIES:
                    total_matched += 1
                else:
                    total_fallback += 1

                # Per dataset+strategy
                dk = (dataset, strat)
                if dk not in per_dataset_strat:
                    per_dataset_strat[dk] = {'correct': 0, 'total': 0}
                per_dataset_strat[dk]['total'] += 1
                if correct:
                    per_dataset_strat[dk]['correct'] += 1

        cov_pct = total_matched / total_all * 100 if total_all else 0
        print(f"\n  ENTITY COVERAGE: {cov_pct:.1f}%  ({total_matched} entity-matched / {total_all} total, {total_fallback} fallback)")

        print(f"\n  {'Dataset':<12}  {'Strategy':<28}  {'Correct':>8}  {'Total':>7}  {'Accuracy':>9}")
        print("  " + "-"*72)
        for (dataset, strat), d in sorted(per_dataset_strat.items()):
            acc = d['correct'] / d['total'] * 100 if d['total'] else 0
            marker = "  ← entity" if strat in _ENTITY_STRATEGIES else "  ← fallback"
            print(f"  {dataset:<12}  {strat:<28}  {d['correct']:>8}  {d['total']:>7}  {acc:>8.1f}%{marker}")


def print_summary_insights(all_results):
    print_header("SUMMARY & INSIGHTS")

    all_accuracies = []
    all_latencies = []
    approach_best = {'baseline': [], 'rag': [], 'graphrag': [], 'hybrid': []}

    for run_name, results in all_results.items():
        # Only iterate over approaches that are actually present
        present = [a for a in ['baseline', 'rag', 'graphrag', 'hybrid'] if a in results]
        if len(present) < 1:
            continue

        for approach in present:
            agg = aggregate_by_model(results[approach])
            acc = calculate_accuracy_metrics(agg)
            if not acc or sum(v['total'] for v in acc.values()) == 0:
                continue

            overall_acc = sum(v['correct'] for v in acc.values()) / sum(v['total'] for v in acc.values()) * 100
            overall_lat = sum(sum(r['latency_ms'] for r in v) / len(v) for v in agg.values() if v) / len(agg)

            approach_best[approach].append(overall_acc)
            all_accuracies.append((run_name, approach, overall_acc))
            all_latencies.append((run_name, approach, overall_lat))

    if not all_accuracies or not all_latencies:
        print("\nInsufficient data for summary (need at least one of baseline/rag/graphrag).")
        return

    best_accuracy = max(all_accuracies, key=lambda x: x[2])
    fastest_run = min(all_latencies, key=lambda x: x[2])

    best_approach_overall = max(approach_best.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
    avg_baseline = sum(approach_best['baseline']) / len(approach_best['baseline']) if approach_best['baseline'] else 0
    avg_rag = sum(approach_best['rag']) / len(approach_best['rag']) if approach_best['rag'] else 0
    avg_graphrag = sum(approach_best['graphrag']) / len(approach_best['graphrag']) if approach_best['graphrag'] else 0
    avg_hybrid = sum(approach_best['hybrid']) / len(approach_best['hybrid']) if approach_best['hybrid'] else 0

    # Get models used — prefer baseline, fall back to any available approach
    models = []
    model_sizes = []
    for run_name, results in all_results.items():
        source = results.get('baseline') or results.get('rag') or results.get('graphrag')
        if source:
            agg = aggregate_by_model(source)
            models = list(agg.keys())
            model_sizes = [get_model_size(m) for m in models]
            break

    retrieval_accs = {'RAG': avg_rag, 'GraphRAG': avg_graphrag, 'Hybrid': avg_hybrid, 'Baseline': avg_baseline}
    winner = max(retrieval_accs, key=retrieval_accs.get)

    print(f"""
KEY FINDINGS:

1. BEST PERFORMANCE:
   - Highest accuracy: {best_accuracy[0]} using {best_accuracy[1].upper()} ({best_accuracy[2]:.1f}%)
   - Fastest approach: {fastest_run[0]} using {fastest_run[1].upper()} ({fastest_run[2]:.0f}ms avg)
   - Best approach overall: {best_approach_overall[0].upper()} (avg {sum(best_approach_overall[1])/len(best_approach_overall[1]):.1f}% across runs)

2. ARCHITECTURE COMPARISON:
   - Baseline: {avg_baseline:.1f}% avg accuracy (no retrieval overhead)
   - Vector RAG: {avg_rag:.1f}% avg accuracy (semantic search)
   - GraphRAG: {avg_graphrag:.1f}% avg accuracy (structured knowledge)
   - Hybrid: {avg_hybrid:.1f}% avg accuracy (graph + vector fusion)
   - Winner: {winner}

3. MODELS TESTED:
   - {len(models)} models: {', '.join(models) if models else 'n/a'}
   - Parameter sizes: {f"{min(model_sizes)}-{max(model_sizes)}B" if model_sizes else 'n/a'}
   - Average latency: {sum(lat for _, _, lat in all_latencies) / len(all_latencies):.0f}ms
    """)


def main():
    print("="*120)
    print(f"{'BENCHMARK ANALYSIS REPORT':^120}")
    print("="*120)
    
    all_results = load_all_results()
    
    if not all_results:
        print("\nNo benchmark results found!")
        print("Run benchmarks first:")
        print("  python scripts/run_parallel_benchmarks.py")
        return
    
    print(f"\nFound {len(all_results)} benchmark run(s):")
    for run_name in all_results.keys():
        print(f"  - {run_name}")
    
    print_all_runs_comparison(all_results)
    print_detailed_model_comparison(all_results)
    print_retrieval_metrics(all_results)
    print_strict_em_comparison(all_results)
    print_mcnemar_table(all_results)
    print_generator_conditional_accuracy(all_results)
    print_graphrag_stratified_accuracy(all_results)
    print_summary_insights(all_results)
    
    print("\n" + "="*120)
    print(f"{'END OF ANALYSIS':^120}")
    print("="*120)


if __name__ == "__main__":
    main()