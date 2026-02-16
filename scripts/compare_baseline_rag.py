#!/usr/bin/env python3
"""
Compare Original RAG vs Chunked RAG results
"""

import json
from pathlib import Path


def compare_results():
    """Compare RAG vs Chunked RAG results"""
    
    baseline_file = Path("results/metrics/baseline_results.json")
    rag_file = Path("results/metrics/rag_results.json")
    chunked_file = Path("results/metrics/rag_chunked_results.json")
    
    if not all([f.exists() for f in [baseline_file, rag_file, chunked_file]]):
        print("Missing results files!")
        return
    
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    with open(rag_file) as f:
        rag = json.load(f)
    
    with open(chunked_file) as f:
        chunked = json.load(f)
    
    print("=" * 100)
    print("BASELINE vs RAG (No Chunking) vs RAG (Chunked) COMPARISON")
    print("=" * 100)
    
    # Get first model to determine number of questions
    first_model = list(baseline.keys())[0]
    num_questions = len(baseline[first_model])
    
    for q_idx in range(num_questions):
        print(f"\n{'='*100}")
        print(f"QUESTION {q_idx + 1}")
        print(f"{'='*100}")
        
        # Get question details
        question = baseline[first_model][q_idx]['question']
        ground_truth = baseline[first_model][q_idx]['ground_truth']
        
        print(f"\nQuestion: {question}")
        print(f"\nGround Truth: {ground_truth}")
        
        # Compare gemma3:27b only (to keep output readable)
        model_name = "gemma3:27b"
        
        print(f"\n{'-'*100}")
        print(f"Model: {model_name}")
        print(f"{'-'*100}")
        
        baseline_answer = baseline[model_name][q_idx]['model_answer']
        rag_answer = rag[model_name][q_idx]['model_answer']
        chunked_answer = chunked[model_name][q_idx]['model_answer']
        
        print(f"\nBASELINE (no context):")
        print(f"   {baseline_answer[:200]}...")
        
        print(f"\nRAG - No Chunking (wrong docs retrieved):")
        print(f"   {rag_answer[:200]}...")
        
        print(f"\nRAG - WITH Chunking (better retrieval):")
        print(f"   {chunked_answer[:200]}...")
        
        # Show retrieved context snippet
        if 'retrieved_context' in chunked[model_name][q_idx]:
            context = chunked[model_name][q_idx]['retrieved_context']
            print(f"\nRetrieved Context (first 200 chars):")
            print(f"   {context[:200]}...")


def main():
    compare_results()


if __name__ == "__main__":
    main()