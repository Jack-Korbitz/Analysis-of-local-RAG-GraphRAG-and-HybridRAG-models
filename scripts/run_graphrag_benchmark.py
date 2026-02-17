#!/usr/bin/env python3
"""
GraphRAG Benchmark - Test models with Neo4j knowledge graph retrieval
"""

import sys
sys.path.append('.')

from src.models.ollama_client import OllamaClient
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.graph_builder import GraphBuilder
from src.graphrag.graph_retriever import GraphRetriever
from datasets import load_from_disk
import json
from tqdm import tqdm
from pathlib import Path


def build_graph_if_needed(neo4j: Neo4jClient, dataset_path: str, max_examples: int = 200):
    """
    Build knowledge graph from dataset if not already built

    Args:
        neo4j: Neo4j client
        dataset_path: Path to dataset
        max_examples: Maximum examples to process
    """
    # Check if graph already has data
    stats = neo4j.get_graph_stats()
    
    if stats.get('Company', 0) > 0:
        print(f"\nExisting graph found:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print("\nUsing existing graph. To rebuild, clear the database first.")
        return
    
    print("\nNo existing graph found. Building knowledge graph...")
    
    builder = GraphBuilder(neo4j, llm_model="gpt-oss:20b")
    builder.build_from_dataset(
        dataset_path=dataset_path,
        max_examples=max_examples,
        use_llm=False
    )


def run_graphrag_test(
    model_name: str,
    retriever: GraphRetriever,
    dataset_path: str,
    num_samples: int = 5
):
    """
    Run GraphRAG test on a model

    Args:
        model_name: Name of Ollama model
        retriever: Graph retriever instance
        dataset_path: Path to dataset
        num_samples: Number of questions to test
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} (WITH GraphRAG)")
    print(f"{'='*60}")

    # Load model
    client = OllamaClient(model_name, temperature=0.1)

    # Load dataset
    dataset = load_from_disk(dataset_path)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    samples = data.select(range(min(num_samples, len(data))))

    results = []

    for i, example in enumerate(tqdm(samples, desc=f"{model_name} (GraphRAG)")):
        question = example['question']
        ground_truth = example.get('program_answer') or example.get('original_answer')

        # Retrieve context from knowledge graph
        context = retriever.retrieve_context_string(question, top_k=5)

        # Generate answer with graph context
        result = client.generate_with_context(
            question=question,
            context=context,
            max_tokens=200
        )

        results.append({
            'question_id': example.get('id', i),
            'question': question,
            'ground_truth': str(ground_truth),
            'graph_context': context,
            'model_answer': result['response'],
            'latency_ms': result['latency_ms'],
            'success': result['success']
        })

        # Print first result as example
        if i == 0:
            print(f"\nExample Question:")
            print(f"   Q: {question[:100]}...")
            print(f"   Ground Truth: {ground_truth}")
            print(f"\nGraph Context:")
            print(f"   {context[:300]}...")
            print(f"\nModel Answer:")
            print(f"   {result['response'][:150]}...")
            print(f"   Latency: {result['latency_ms']}ms")

    # Calculate stats
    avg_latency = sum(r['latency_ms'] for r in results) / len(results)
    success_rate = sum(r['success'] for r in results) / len(results) * 100

    print(f"\nResults for {model_name} (GraphRAG):")
    print(f"   Questions answered: {len(results)}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Avg latency: {avg_latency:.2f}ms")

    return results


def main():
    """Run GraphRAG benchmark on all models"""

    print("="*60)
    print("GRAPHRAG BENCHMARK (Neo4j Knowledge Graph)")
    print("="*60)

    # Initialize Neo4j
    neo4j = Neo4jClient()

    # Dataset path
    dataset_path = "data/benchmarks/t2-ragbench-FinQA"

    # Build graph if needed
    build_graph_if_needed(neo4j, dataset_path, max_examples=200)

    # Initialize retriever
    retriever = GraphRetriever(neo4j)

    # Models to test
    models = ["gemma3:27b", "gpt-oss:20b", "qwen3:30b"]

    # Number of questions
    num_samples = 5

    all_results = {}

    for model in models:
        try:
            results = run_graphrag_test(
                model,
                retriever,
                dataset_path,
                num_samples=num_samples
            )
            all_results[model] = results
        except Exception as e:
            print(f"\nError testing {model}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "graphrag_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print comparison table
    print(f"\nMODEL COMPARISON (GraphRAG):")
    print(f"{'Model':<20} {'Avg Latency (ms)':<20} {'Success Rate'}")
    print("-" * 60)

    for model, results in all_results.items():
        if results:
            avg_latency = sum(r['latency_ms'] for r in results) / len(results)
            success_rate = sum(r['success'] for r in results) / len(results) * 100
            print(f"{model:<20} {avg_latency:<20.2f} {success_rate:.1f}%")

    # Compare all three approaches
    baseline_file = output_dir / "baseline_results.json"
    rag_file = output_dir / "rag_chunked_results.json"

    if baseline_file.exists() and rag_file.exists():
        print(f"\n{'='*60}")
        print("FULL COMPARISON: Baseline vs RAG vs GraphRAG")
        print(f"{'='*60}")

        with open(baseline_file) as f:
            baseline = json.load(f)
        with open(rag_file) as f:
            rag = json.load(f)

        print(f"\n{'Model':<20} {'Baseline':>12} {'RAG':>12} {'GraphRAG':>12}")
        print("-" * 60)

        for model in models:
            if all(model in r for r in [baseline, rag, all_results]):
                b_lat = sum(r['latency_ms'] for r in baseline[model]) / len(baseline[model])
                r_lat = sum(r['latency_ms'] for r in rag[model]) / len(rag[model])
                g_lat = sum(r['latency_ms'] for r in all_results[model]) / len(all_results[model])

                print(f"{model:<20} {b_lat:>10.0f}ms {r_lat:>10.0f}ms {g_lat:>10.0f}ms")

    neo4j.close()


if __name__ == "__main__":
    main()