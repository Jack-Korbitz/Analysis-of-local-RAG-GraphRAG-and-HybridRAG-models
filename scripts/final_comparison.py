#!/usr/bin/env python3
"""
Final Comparison - Baseline vs RAG vs GraphRAG
Full analysis of answer quality and latency
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


def load_results():
    """Load all result files"""
    output_dir = Path("results/metrics")

    files = {
        'baseline': output_dir / "baseline_results.json",
        'rag': output_dir / "rag_chunked_results.json",
        'graphrag': output_dir / "graphrag_results.json"
    }

    results = {}
    for name, path in files.items():
        if path.exists():
            with open(path, encoding='utf-8') as f:
                results[name] = json.load(f)
        else:
            print(f"Missing: {path}")

    return results


def clean_text(text: str) -> str:
    """Clean text for safe printing"""
    return text.encode('ascii', errors='replace').decode('ascii')


def print_latency_summary(results: dict):
    """Print latency comparison table"""
    models = list(results['baseline'].keys())

    print("\n" + "="*80)
    print("LATENCY COMPARISON (milliseconds)")
    print("="*80)
    print(f"{'Model':<22} {'Baseline':>12} {'Vector RAG':>12} {'GraphRAG':>12} {'Best'}")
    print("-"*80)

    for model in models:
        latencies = {}
        for approach, data in results.items():
            if model in data:
                avg = sum(r['latency_ms'] for r in data[model]) / len(data[model])
                latencies[approach] = avg

        best = min(latencies, key=latencies.get)

        print(
            f"{model:<22} "
            f"{latencies.get('baseline', 0):>10.0f}ms "
            f"{latencies.get('rag', 0):>10.0f}ms "
            f"{latencies.get('graphrag', 0):>10.0f}ms "
            f"  {best}"
        )


def print_answer_comparison(results: dict):
    """Print side by side answer comparison"""
    models = list(results['baseline'].keys())
    first_model = models[0]
    num_questions = len(results['baseline'][first_model])

    print("\n" + "="*80)
    print("ANSWER QUALITY COMPARISON")
    print("="*80)

    for q_idx in range(num_questions):
        question = results['baseline'][first_model][q_idx]['question']
        ground_truth = results['baseline'][first_model][q_idx]['ground_truth']

        print(f"\n{'='*80}")
        print(f"QUESTION {q_idx + 1}: {clean_text(question[:120])}")
        print(f"GROUND TRUTH: {clean_text(str(ground_truth))}")
        print(f"{'='*80}")

        for model in models:
            print(f"\n  Model: {model}")
            print(f"  {'-'*70}")

            for approach in ['baseline', 'rag', 'graphrag']:
                if model in results.get(approach, {}):
                    answer = results[approach][model][q_idx]['model_answer']
                    latency = results[approach][model][q_idx]['latency_ms']

                    # Truncate and clean answer
                    answer_short = answer[:150].replace('\n', ' ')
                    if len(answer) > 150:
                        answer_short += "..."

                    answer_clean = clean_text(answer_short)
                    label = approach.upper().ljust(10)
                    print(f"  {label} ({latency:.0f}ms): {answer_clean}")


def print_graph_context_examples(results: dict):
    """Show examples of what GraphRAG retrieved"""
    models = list(results['graphrag'].keys())
    first_model = models[0]
    num_questions = len(results['graphrag'][first_model])

    print("\n" + "="*80)
    print("GRAPHRAG CONTEXT EXAMPLES")
    print("="*80)

    for q_idx in range(min(3, num_questions)):
        question = results['graphrag'][first_model][q_idx]['question']
        context = results['graphrag'][first_model][q_idx].get('graph_context', 'N/A')
        ground_truth = results['graphrag'][first_model][q_idx]['ground_truth']

        print(f"\nQuestion {q_idx + 1}: {clean_text(question[:100])}...")
        print(f"Ground Truth: {clean_text(str(ground_truth))}")
        print(f"Graph Context: {clean_text(context[:300])}...")


def print_summary(results: dict):
    """Print high level summary"""
    models = list(results['baseline'].keys())

    print("\n" + "="*80)
    print("PROJECT SUMMARY")
    print("="*80)

    print("""
Architecture Comparison:

  BASELINE (No RAG)
    - Models answer from training data only
    - Fast but prone to hallucination
    - No access to specific document data

  VECTOR RAG
    - Embeds documents into vector store (FAISS)
    - Retrieves similar text chunks
    - Better when documents are indexed correctly
    - Slower due to larger context

  GRAPHRAG (Neo4j)
    - Extracts entities and relationships into graph
    - Precise structured retrieval (Company + Year + Metric)
    - Fastest retrieval for structured financial data
    - Correct answer when entity is in graph
    """)

    print("Key Findings:")
    print("  1. GraphRAG is fastest for structured financial queries")
    print("  2. gpt-oss:20b is the fastest model across all approaches")
    print("  3. GraphRAG correctly answered Question 1 (all three models)")
    print("  4. Vector RAG struggles with long financial documents")
    print("  5. GraphRAG excels at entity-specific lookups (Company + Year + Metric)")
    print("  6. Baseline models hallucinate when lacking specific training data")

    print("\nModels Tested:")
    for model in models:
        print(f"  - {model}")

    print("\nDatasets Used:")
    print("  - G4KMU/t2-ragbench (FinQA, ConvFinQA, TAT-DQA)")
    print("  - galileo-ai/ragbench (hotpotqa, finqa, pubmedqa, msmarco)")

    print("\nTechnologies Used:")
    print("  - Ollama (local LLM serving)")
    print("  - FAISS (vector store for RAG)")
    print("  - Neo4j (graph database for GraphRAG)")
    print("  - sentence-transformers (embeddings)")
    print("  - HuggingFace datasets (benchmarks)")


def print_question_scorecard(results: dict):
    """Print a simple scorecard for each question"""
    models = list(results['baseline'].keys())
    first_model = models[0]
    num_questions = len(results['baseline'][first_model])

    print("\n" + "="*80)
    print("ANSWER SCORECARD")
    print("(Did the model answer include the ground truth value?)")
    print("="*80)

    for q_idx in range(num_questions):
        question = results['baseline'][first_model][q_idx]['question']
        ground_truth = results['baseline'][first_model][q_idx]['ground_truth']

        print(f"\nQ{q_idx + 1}: {clean_text(question[:80])}...")
        print(f"     Ground Truth: {ground_truth}")
        print(f"     {'Model':<22} {'Baseline':<12} {'RAG':<12} {'GraphRAG':<12}")
        print(f"     {'-'*58}")

        for model in models:
            scores = {}
            for approach in ['baseline', 'rag', 'graphrag']:
                if model in results.get(approach, {}):
                    answer = results[approach][model][q_idx]['model_answer']
                    # Simple check: does answer contain ground truth value?
                    gt_str = str(ground_truth).replace('.0', '')
                    contains = gt_str in answer
                    scores[approach] = 'YES' if contains else 'no'
                else:
                    scores[approach] = 'N/A'

            print(
                f"     {model:<22} "
                f"{scores.get('baseline', 'N/A'):<12} "
                f"{scores.get('rag', 'N/A'):<12} "
                f"{scores.get('graphrag', 'N/A'):<12}"
            )


def main():
    """Run final comparison"""
    print("="*80)
    print("FINAL COMPARISON REPORT")
    print("Baseline vs Vector RAG vs GraphRAG")
    print("="*80)

    results = load_results()

    if not results:
        print("No results found. Run all benchmarks first.")
        return

    print_latency_summary(results)
    print_answer_comparison(results)
    print_graph_context_examples(results)
    print_question_scorecard(results)
    print_summary(results)

    # Save report to file
    report_path = Path("results/final_comparison_report.txt")
    print(f"\nTo save report to file run:")
    print(f"  python scripts/final_comparison.py > results/final_comparison_report.txt")


if __name__ == "__main__":
    main()