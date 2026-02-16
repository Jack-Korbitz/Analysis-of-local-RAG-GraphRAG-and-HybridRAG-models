#!/usr/bin/env python3
"""
Review benchmark results
"""

import json
from pathlib import Path


def review_results(results_file: str):
    """Display results in a readable format"""
    
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    print("=" * 80)
    print("BASELINE BENCHMARK RESULTS REVIEW")
    print("=" * 80)
    
    # Get question list (same for all models)
    first_model = list(all_results.keys())[0]
    num_questions = len(all_results[first_model])
    
    for q_idx in range(num_questions):
        print(f"\n{'='*80}")
        print(f"QUESTION {q_idx + 1}")
        print(f"{'='*80}")
        
        # Get question text from first model
        question = all_results[first_model][q_idx]['question']
        ground_truth = all_results[first_model][q_idx]['ground_truth']
        
        print(f"\nQuestion: {question}")
        print(f"\nGround Truth: {ground_truth}")
        
        print(f"\n{'Model Answers:':_<80}")
        
        for model_name, results in all_results.items():
            answer = results[q_idx]['model_answer']
            latency = results[q_idx]['latency_ms']
            
            print(f"\n{model_name} ({latency:.0f}ms):")
            print(f"   {answer[:200]}...")  # First 200 chars


def main():
    results_file = "results/metrics/baseline_results.json"
    
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("Run run_baseline_benchmark.py first!")
        return
    
    review_results(results_file)


if __name__ == "__main__":
    main()