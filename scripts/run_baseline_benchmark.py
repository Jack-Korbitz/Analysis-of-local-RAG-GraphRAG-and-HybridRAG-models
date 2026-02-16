#!/usr/bin/env python3
"""
Baseline Benchmark - Test models WITHOUT RAG
Tests direct question-answering capability
"""

import sys
sys.path.append('.')

from src.models.ollama_client import OllamaClient
from datasets import load_from_disk
import json
from tqdm import tqdm
from pathlib import Path


def run_baseline_test(model_name: str, dataset_path: str, num_samples: int = 5):
    """
    Run baseline test on a model
    
    Args:
        model_name: Name of Ollama model
        dataset_path: Path to dataset
        num_samples: Number of questions to test
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    client = OllamaClient(model_name, temperature=0.1)  # Low temp for consistency
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Take first N samples
    samples = data.select(range(min(num_samples, len(data))))
    
    results = []
    
    for i, example in enumerate(tqdm(samples, desc=f"{model_name}")):
        question = example['question']
        ground_truth = example.get('program_answer') or example.get('original_answer')
        
        # Generate answer (no context - baseline)
        result = client.generate(
            prompt=question,
            system_prompt="You are a financial analyst. Answer the question concisely and accurately.",
            max_tokens=200
        )
        
        results.append({
            'question_id': example.get('id', i),
            'question': question,
            'ground_truth': str(ground_truth),
            'model_answer': result['response'],
            'latency_ms': result['latency_ms'],
            'success': result['success']
        })
        
        # Print first result as example
        if i == 0:
            print(f"\nExample Question:")
            print(f"   Q: {question[:100]}...")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Model Answer: {result['response'][:100]}...")
            print(f"   {result['latency_ms']}ms")
    
    # Calculate stats
    avg_latency = sum(r['latency_ms'] for r in results) / len(results)
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    
    print(f"\nResults for {model_name}:")
    print(f"   Questions answered: {len(results)}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Avg latency: {avg_latency:.2f}ms")
    
    return results


def main():
    """Run baseline benchmark on all models"""
    
    print("="*60)
    print("BASELINE BENCHMARK (No RAG)")
    print("="*60)
    print("Testing direct question-answering without retrieval")
    
    # Models to test
    models = ["gemma3:27b", "gpt-oss:20b", "qwen3:30b"]
    
    # Dataset to use (FinQA - good for baseline)
    dataset_path = "data/benchmarks/t2-ragbench-FinQA"
    
    # Number of questions to test (start small)
    num_samples = 5
    
    all_results = {}
    
    for model in models:
        try:
            results = run_baseline_test(model, dataset_path, num_samples)
            all_results[model] = results
        except Exception as e:
            print(f"\nError testing {model}: {e}")
            continue
    
    # Save results
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print comparison
    print(f"\nMODEL COMPARISON:")
    print(f"{'Model':<20} {'Avg Latency (ms)':<20} {'Success Rate'}")
    print("-" * 60)
    
    for model, results in all_results.items():
        if results:
            avg_latency = sum(r['latency_ms'] for r in results) / len(results)
            success_rate = sum(r['success'] for r in results) / len(results) * 100
            print(f"{model:<20} {avg_latency:<20.2f} {success_rate:.1f}%")


if __name__ == "__main__":
    main()