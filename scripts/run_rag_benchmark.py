#!/usr/bin/env python3
"""
RAG Benchmark - Test models WITH retrieval augmentation
"""

import sys
sys.path.append('.')

from src.models.ollama_client import OllamaClient
from src.rag.retriever import RAGRetriever
from datasets import load_from_disk
import json
from tqdm import tqdm
from pathlib import Path


def index_dataset_contexts(retriever: RAGRetriever, dataset_path: str, max_docs: int = 100):
    """
    Index the context documents from the dataset
    
    Args:
        retriever: RAG retriever instance
        dataset_path: Path to dataset
        max_docs: Maximum number of documents to index
    """
    print(f"\n{'='*60}")
    print("Indexing Dataset Contexts")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Extract unique contexts (avoid duplicates)
    contexts_dict = {}
    metadatas_dict = {}
    
    for i, example in enumerate(tqdm(data, desc="Extracting contexts")):
        if len(contexts_dict) >= max_docs:
            break
        
        # Get context (handle different dataset formats)
        context = example.get('context', '')
        
        if not context or len(context) < 50:  # Skip very short contexts
            continue
        
        # Use context as key to avoid duplicates
        context_id = example.get('context_id', f"ctx_{i}")
        
        if context_id not in contexts_dict:
            contexts_dict[context_id] = context
            metadatas_dict[context_id] = {
                'context_id': context_id,
                'company': example.get('company_name', 'Unknown'),
                'year': example.get('report_year', 'Unknown'),
                'source': 'dataset'
            }
    
    # Convert to lists
    contexts = list(contexts_dict.values())
    metadatas = list(metadatas_dict.values())
    
    print(f"\n📊 Found {len(contexts)} unique contexts to index")
    
    # Index in batches
    batch_size = 50
    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        print(f"\n📚 Indexing batch {i//batch_size + 1}/{(len(contexts)-1)//batch_size + 1}...")
        retriever.index_documents(batch_contexts, batch_metadatas, batch_size=32)
    
    print(f"\n✅ Indexed {len(contexts)} contexts")
    
    # Save the index
    retriever.save("finqa_rag_index")
    print(f"💾 Saved index to disk")


def run_rag_test(
    model_name: str,
    retriever: RAGRetriever,
    dataset_path: str,
    num_samples: int = 5,
    top_k: int = 3
):
    """
    Run RAG-enhanced test on a model
    
    Args:
        model_name: Name of Ollama model
        retriever: RAG retriever instance
        dataset_path: Path to dataset
        num_samples: Number of questions to test
        top_k: Number of context documents to retrieve
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} (WITH RAG)")
    print(f"{'='*60}")
    
    # Load model
    client = OllamaClient(model_name, temperature=0.1)
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Take first N samples
    samples = data.select(range(min(num_samples, len(data))))
    
    results = []
    
    for i, example in enumerate(tqdm(samples, desc=f"{model_name} (RAG)")):
        question = example['question']
        ground_truth = example.get('program_answer') or example.get('original_answer')
        
        # Retrieve relevant context
        context = retriever.retrieve_context_string(question, top_k=top_k)
        
        # Generate answer WITH context
        result = client.generate_with_context(
            question=question,
            context=context,
            max_tokens=200
        )
        
        results.append({
            'question_id': example.get('id', i),
            'question': question,
            'ground_truth': str(ground_truth),
            'retrieved_context': context[:500] + "..." if len(context) > 500 else context,
            'model_answer': result['response'],
            'latency_ms': result['latency_ms'],
            'success': result['success']
        })
        
        # Print first result as example
        if i == 0:
            print(f"\n📝 Example Question:")
            print(f"   Q: {question[:100]}...")
            print(f"   Ground Truth: {ground_truth}")
            print(f"\n📄 Retrieved Context (first 200 chars):")
            print(f"   {context[:200]}...")
            print(f"\n🤖 Model Answer:")
            print(f"   {result['response'][:150]}...")
            print(f"   ⏱️  {result['latency_ms']}ms")
    
    # Calculate stats
    avg_latency = sum(r['latency_ms'] for r in results) / len(results)
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    
    print(f"\n📊 Results for {model_name} (RAG):")
    print(f"   Questions answered: {len(results)}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Avg latency: {avg_latency:.2f}ms")
    
    return results


def main():
    """Run RAG benchmark on all models"""
    
    print("="*60)
    print("🚀 RAG BENCHMARK (With Retrieval)")
    print("="*60)
    
    # Initialize RAG retriever
    retriever = RAGRetriever(collection_name="finqa_rag")
    
    # Dataset to use
    dataset_path = "data/benchmarks/t2-ragbench-FinQA"
    
    # Try to load existing index
    print("\n🔍 Checking for existing index...")
    if not retriever.load("finqa_rag_index"):
        print("\n📚 No existing index found. Building new index...")
        retriever.vector_store.clear()
        index_dataset_contexts(retriever, dataset_path, max_docs=100)
    else:
        print("✅ Loaded existing index")
        print(f"   {retriever.get_stats()}")
    
    # Models to test
    models = ["gemma3:27b", "gpt-oss:20b", "qwen3:30b"]
    
    # Number of questions to test
    num_samples = 5
    
    all_results = {}
    
    for model in models:
        try:
            results = run_rag_test(
                model,
                retriever,
                dataset_path,
                num_samples=num_samples,
                top_k=3  # Retrieve top 3 contexts
            )
            all_results[model] = results
        except Exception as e:
            print(f"\n❌ Error testing {model}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "rag_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print comparison
    print(f"\n📊 MODEL COMPARISON (RAG):")
    print(f"{'Model':<20} {'Avg Latency (ms)':<20} {'Success Rate'}")
    print("-" * 60)
    
    for model, results in all_results.items():
        if results:
            avg_latency = sum(r['latency_ms'] for r in results) / len(results)
            success_rate = sum(r['success'] for r in results) / len(results) * 100
            print(f"{model:<20} {avg_latency:<20.2f} {success_rate:.1f}%")
    
    # Load baseline for comparison
    baseline_file = output_dir / "baseline_results.json"
    if baseline_file.exists():
        print(f"\n{'='*60}")
        print("📊 BASELINE vs RAG COMPARISON")
        print(f"{'='*60}")
        
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        print(f"\n{'Model':<20} {'Baseline (ms)':<15} {'RAG (ms)':<15} {'Difference'}")
        print("-" * 65)
        
        for model in models:
            if model in baseline_results and model in all_results:
                baseline_latency = sum(r['latency_ms'] for r in baseline_results[model]) / len(baseline_results[model])
                rag_latency = sum(r['latency_ms'] for r in all_results[model]) / len(all_results[model])
                diff = rag_latency - baseline_latency
                
                print(f"{model:<20} {baseline_latency:<15.2f} {rag_latency:<15.2f} {diff:+.2f}ms")


if __name__ == "__main__":
    main()