import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


def load_all_results():
    """Load all result files from different benchmark runs"""
    output_dir = Path("results/metrics")
    
    result_files = {
        'Current Models': {
            'baseline': output_dir / "baseline_fast.json",
            'rag': output_dir / "rag_fast.json",
            'graphrag': output_dir / "graphrag_fast.json"
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


def check_accuracy(answer, ground_truth):
    answer_lower = str(answer).lower()
    gt_str = str(ground_truth).lower().replace('.0', '')
    
    if gt_str in answer_lower:
        return True
    
    import re
    answer_numbers = re.findall(r'-?\d+\.?\d*', answer_lower)
    gt_numbers = re.findall(r'-?\d+\.?\d*', gt_str)
    
    if gt_numbers and answer_numbers:
        try:
            gt_val = float(gt_numbers[0])
            for ans_num in answer_numbers:
                ans_val = float(ans_num)
                if abs(gt_val) < 1:
                    if abs(ans_val - gt_val) / max(abs(gt_val), 0.001) < 0.05:
                        return True
                else:
                    if abs(ans_val - gt_val) < 0.1:
                        return True
        except ValueError:
            pass
    
    return False


def calculate_accuracy_metrics(results_dict):
    accuracy = {}
    for model, questions in results_dict.items():
        correct = 0
        total = len(questions)
        
        for q in questions:
            if check_accuracy(q['answer'], q['ground_truth']):
                correct += 1
        
        accuracy[model] = {
            'correct': correct,
            'total': total,
            'accuracy': (correct / total * 100) if total > 0 else 0
        }
    
    return accuracy


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


def print_all_runs_comparison(all_results):
    print_header("BENCHMARK RESULTS")
    
    print("\nACCURACY BY APPROACH:")
    print(f"{'Run Name':<30} {'Baseline Acc':<15} {'RAG Acc':<15} {'GraphRAG Acc':<15} {'Best Approach'}")
    print("-"*120)
    
    for run_name, results in all_results.items():
        if len(results) < 3:
            continue
        
        baseline = aggregate_by_model(results['baseline'])
        rag = aggregate_by_model(results['rag'])
        graphrag = aggregate_by_model(results['graphrag'])
        
        baseline_acc = calculate_accuracy_metrics(baseline)
        rag_acc = calculate_accuracy_metrics(rag)
        graphrag_acc = calculate_accuracy_metrics(graphrag)
        
        all_baseline = sum(v['correct'] for v in baseline_acc.values()) / sum(v['total'] for v in baseline_acc.values()) * 100
        all_rag = sum(v['correct'] for v in rag_acc.values()) / sum(v['total'] for v in rag_acc.values()) * 100
        all_graphrag = sum(v['correct'] for v in graphrag_acc.values()) / sum(v['total'] for v in graphrag_acc.values()) * 100
        
        best = max([('Baseline', all_baseline), ('RAG', all_rag), ('GraphRAG', all_graphrag)], key=lambda x: x[1])[0]
        
        print(f"{run_name:<30} {all_baseline:>8.1f}%      {all_rag:>8.1f}%      {all_graphrag:>8.1f}%      {best}")
    
    print("\n" + "="*120)
    print("LATENCY BY APPROACH:")
    print(f"{'Run Name':<30} {'Baseline ms':<15} {'RAG ms':<15} {'GraphRAG ms':<15} {'Fastest'}")
    print("-"*120)
    
    for run_name, results in all_results.items():
        if len(results) < 3:
            continue
        
        baseline = aggregate_by_model(results['baseline'])
        rag = aggregate_by_model(results['rag'])
        graphrag = aggregate_by_model(results['graphrag'])
        
        baseline_avg = sum(sum(r['latency_ms'] for r in v) / len(v) for v in baseline.values()) / len(baseline)
        rag_avg = sum(sum(r['latency_ms'] for r in v) / len(v) for v in rag.values()) / len(rag)
        graphrag_avg = sum(sum(r['latency_ms'] for r in v) / len(v) for v in graphrag.values()) / len(graphrag)
        
        fastest = min([('Baseline', baseline_avg), ('RAG', rag_avg), ('GraphRAG', graphrag_avg)], key=lambda x: x[1])[0]
        
        print(f"{run_name:<30} {baseline_avg:>10.0f}ms    {rag_avg:>10.0f}ms    {graphrag_avg:>10.0f}ms    {fastest}")


def print_detailed_model_comparison(all_results):
    print_header("DETAILED MODEL COMPARISON")
    
    for run_name, results in all_results.items():
        if len(results) < 3:
            continue
        
        print(f"\n{run_name.upper()}")
        print("-"*120)
        
        baseline = aggregate_by_model(results['baseline'])
        rag = aggregate_by_model(results['rag'])
        graphrag = aggregate_by_model(results['graphrag'])
        
        baseline_acc = calculate_accuracy_metrics(baseline)
        rag_acc = calculate_accuracy_metrics(rag)
        graphrag_acc = calculate_accuracy_metrics(graphrag)
        
        models = sorted(baseline.keys())
        
        print(f"\n{'Model':<20} {'Baseline':<25} {'Vector RAG':<25} {'GraphRAG':<25} {'Best'}")
        print("-"*120)
        
        for model in models:
            b_pct = baseline_acc[model]['accuracy']
            r_pct = rag_acc[model]['accuracy']
            g_pct = graphrag_acc[model]['accuracy']
            
            b_lat = sum(r['latency_ms'] for r in baseline[model]) / len(baseline[model])
            r_lat = sum(r['latency_ms'] for r in rag[model]) / len(rag[model])
            g_lat = sum(r['latency_ms'] for r in graphrag[model]) / len(graphrag[model])
            
            best = max([('Baseline', b_pct), ('RAG', r_pct), ('GraphRAG', g_pct)], key=lambda x: x[1])[0]
            
            print(
                f"{model:<20} "
                f"{b_pct:>5.1f}% ({b_lat:>6.0f}ms)     "
                f"{r_pct:>5.1f}% ({r_lat:>6.0f}ms)     "
                f"{g_pct:>5.1f}% ({g_lat:>6.0f}ms)     "
                f"{best}"
            )


def print_summary_insights(all_results):
    print_header("SUMMARY & INSIGHTS")
    
    all_accuracies = []
    all_latencies = []
    approach_best = {'baseline': [], 'rag': [], 'graphrag': []}
    
    for run_name, results in all_results.items():
        if len(results) < 3:
            continue
        
        for approach in ['baseline', 'rag', 'graphrag']:
            agg = aggregate_by_model(results[approach])
            acc = calculate_accuracy_metrics(agg)
            
            overall_acc = sum(v['correct'] for v in acc.values()) / sum(v['total'] for v in acc.values()) * 100
            overall_lat = sum(sum(r['latency_ms'] for r in v) / len(v) for v in agg.values()) / len(agg)
            
            approach_best[approach].append(overall_acc)
            all_accuracies.append((run_name, approach, overall_acc))
            all_latencies.append((run_name, approach, overall_lat))
    
    best_accuracy = max(all_accuracies, key=lambda x: x[2])
    fastest_run = min(all_latencies, key=lambda x: x[2])
    
    best_approach_overall = max(approach_best.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
    avg_baseline = sum(approach_best['baseline']) / len(approach_best['baseline']) if approach_best['baseline'] else 0
    avg_rag = sum(approach_best['rag']) / len(approach_best['rag']) if approach_best['rag'] else 0
    avg_graphrag = sum(approach_best['graphrag']) / len(approach_best['graphrag']) if approach_best['graphrag'] else 0
    
    # Get models used
    for run_name, results in all_results.items():
        baseline = aggregate_by_model(results['baseline'])
        models = list(baseline.keys())
        model_sizes = [get_model_size(m) for m in models]
    
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
   - Winner: {"RAG" if avg_rag > max(avg_baseline, avg_graphrag) else "GraphRAG" if avg_graphrag > avg_baseline else "Baseline"}

3. MODELS TESTED:
   - {len(models)} models: {', '.join(models)}
   - Parameter sizes: {min(model_sizes)}-{max(model_sizes)}B
   - Average latency: {sum(lat for _, _, lat in all_latencies) / len(all_latencies):.0f}ms

4. CURRENT STATUS:
   - Financial QA is challenging: Best accuracy is {best_accuracy[2]:.1f}%
   - {"GraphRAG underperforming - extraction patterns need work" if avg_graphrag < avg_baseline else "GraphRAG competitive with baseline"}
   - Speed/accuracy tradeoff: {"Baseline fastest" if avg_baseline > avg_rag else "RAG provides best accuracy"}

5. NEXT STEPS TO IMPROVE:
   - Implement Chain-of-Thought prompting (+15-20% expected)
   - Add error recovery/self-critique (+10-15% expected)
   - Improve GraphRAG extraction patterns (markdown tables)
   - Consider table-aware chunking for RAG
   - Target: {best_accuracy[2]:.1f}% → 40-50%+ accuracy
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
    print_summary_insights(all_results)
    
    print("\n" + "="*120)
    print(f"{'END OF ANALYSIS':^120}")
    print("="*120)


if __name__ == "__main__":
    main()