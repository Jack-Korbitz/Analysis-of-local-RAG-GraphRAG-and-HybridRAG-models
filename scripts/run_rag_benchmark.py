#!/usr/bin/env python3
"""
RAG Benchmark - Test models WITH retrieval augmentation (IMPROVED WITH CHUNKING)
"""

import sys
sys.path.append('.')

from src.models.ollama_client import OllamaClient
from src.rag.retriever import RAGRetriever
from datasets import load_from_disk
import json
from tqdm import tqdm
from pathlib import Path


def index_dataset_contexts(retriever: RAGRetriever, dataset_path: str, max_examples: int = 200):
    """
    Index the context documents from the dataset WITH CHUNKING
    
    Args:
        retriever: RAG retriever instance
        dataset_path: Path to dataset
        max_examples: Maximum number of dataset examples to process
    """
    from src.utils.chunking import DocumentChunker
    
    print(f"\n{'='*60}")
    print("Indexing Dataset Contexts WITH CHUNKING")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Take subset of examples
    examples = data.select(range(min(max_examples, len(data))))
    
    print(f"\nProcessing {len(examples)} examples from dataset")
    
    # Initialize chunker (smaller chunks = better retrieval)
    chunker = DocumentChunker(
        chunk_size=400,      # Smaller chunks
        chunk_overlap=100     # Good overlap
    )
    
    # Chunk all contexts
    print(f"\nChunking contexts...")
    all_chunks = chunker.chunk_dataset_contexts(
        list(examples),
        context_field='context'
    )
    
    print(f"\nCreated {len(all_chunks)} chunks from {len(examples)} examples")
    print(f"   Avg chunks per example: {len(all_chunks) / len(examples):.1f}")
    
    # Deduplicate chunks by text (avoid indexing duplicates)
    unique_chunks = {}
    for chunk in all_chunks:
        text = chunk['text']
        if text not in unique_chunks:
            unique_chunks[text] = chunk
    
    chunks_list = list(unique_chunks.values())
    print(f"\nAfter deduplication: {len(chunks_list)} unique chunks")
    
    # Extract texts and metadatas
    texts = [chunk['text'] for chunk in chunks_list]
    metadatas = [chunk['metadata'] for chunk in chunks_list]
    
    # Index in batches
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        print(f"\nIndexing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        retriever.index_documents(batch_texts, batch_metadatas, batch_size=32)
    
    print(f"\nIndexed {len(texts)} chunks")
    
    # Save the index
    retriever.save("finqa_rag_chunked_index")
    print(f"Saved chunked index to disk")


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
            print(f"\nExample Question:")
            print(f"   Q: {question[:100]}...")
            print(f"   Ground Truth: {ground_truth}")
            print(f"\nRetrieved Context (first 200 chars):")
            print(f"   {context[:200]}...")
            print(f"\nModel Answer:")
            print(f"   {result['response'][:150]}...")
            print(f"   Latency: {result['latency_ms']}ms")
    
    # Calculate stats
    avg_latency = sum(r['latency_ms'] for r in results) / len(results)
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    
    print(f"\nResults for {model_name} (RAG):")
    print(f"   Questions answered: {len(results)}")
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   Avg latency: {avg_latency:.2f}ms")
    
    return results


def main():
    """Run RAG benchmark on all models"""
    
    print("="*60)
    print("RAG BENCHMARK (With Retrieval + CHUNKING)")
    print("="*60)
    
    # Initialize RAG retriever
    retriever = RAGRetriever(collection_name="finqa_rag_chunked")
    
    # Dataset to use
    dataset_path = "data/benchmarks/t2-ragbench-FinQA"
    
    # Try to load existing index
    print("\nChecking for existing chunked index...")
    if not retriever.load("finqa_rag_chunked_index"):
        print("\nNo existing chunked index found. Building new index...")
        retriever.vector_store.clear()
        index_dataset_contexts(retriever, dataset_path, max_examples=200)
    else:
        print("Loaded existing chunked index")
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
            print(f"\nError testing {model}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "rag_chunked_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print comparison
    print(f"\nMODEL COMPARISON (RAG with Chunking):")
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
        print("BASELINE vs RAG (Chunked) COMPARISON")
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