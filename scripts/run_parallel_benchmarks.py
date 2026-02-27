import sys
sys.path.append('.')

from src.models.ollama_client import OllamaClient
from src.rag.retriever import RAGRetriever
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.graph_retriever import GraphRetriever
from datasets import load_from_disk
import json
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import datetime, timedelta

# Ensure to run this line of code before executing code & open docker desktop
# docker-compose up -d
# Check after 30 seconds to ensure its running with: 
# docker ps

class ParallelBenchmarkRunner:
    def __init__(self, models, datasets, num_samples=10, max_workers=3):
        self.models = models
        self.datasets = datasets
        self.num_samples = num_samples
        self.max_workers = max_workers
        self.results_lock = threading.Lock()
        self.results = {}
        self.timings = {}
    
    def _run_model_baseline(self, model, dataset_name, dataset_path, samples):
        results = []
        client = OllamaClient(model, temperature=0.1)
        
        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            result = client.generate(
                prompt=example['question'],
                max_tokens=200
            )
            
            if not result.get('success', False):
                continue
            
            results.append({
                'question': example['question'],
                'ground_truth': str(example.get('program_answer') or example.get('original_answer')),
                'answer': result['response'],
                'latency_ms': result['latency_ms']
            })
        
        return model, dataset_name, results
    
    def run_baseline(self):
        print("\n" + "="*60)
        print("BASELINE BENCHMARK (PARALLEL)")
        print("="*60)
        
        start_time = time.time()
        self.results['baseline'] = {}
        
        tasks = []
        for dataset_name, dataset_path in self.datasets.items():
            dataset = load_from_disk(dataset_path)
            split = list(dataset.keys())[0]
            samples = list(dataset[split].select(range(min(self.num_samples, len(dataset[split])))))
            
            for model in self.models:
                tasks.append((model, dataset_name, dataset_path, samples))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_model_baseline, *task): task 
                for task in tasks
            }
            
            for future in as_completed(futures):
                model, dataset_name, results = future.result()
                key = f"{model}_{dataset_name}"
                
                with self.results_lock:
                    self.results['baseline'][key] = results
        
        elapsed = time.time() - start_time
        self.timings['baseline'] = elapsed
        
        self._save('baseline')
        
        print(f"\nBaseline completed in: {timedelta(seconds=int(elapsed))}")
    
    def _run_model_rag(self, model, dataset_name, retriever, samples):
        results = []
        client = OllamaClient(model, temperature=0.1)
        
        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            context = retriever.retrieve_context_string(example['question'], top_k=5)
            result = client.generate_with_context(
                question=example['question'],
                context=context,
                max_tokens=20000
            )
            
            if not result.get('success', False):
                continue
            
            results.append({
                'question': example['question'],
                'ground_truth': str(example.get('program_answer') or example.get('original_answer')),
                'answer': result['response'],
                'latency_ms': result['latency_ms']
            })
        
        return model, dataset_name, results
    
    def run_rag(self):
        print("\n" + "="*60)
        print("VECTOR RAG BENCHMARK (IMPROVED V2 INDEXES)")
        print("="*60)
        
        start_time = time.time()
        self.results['rag'] = {}
        retrievers = {}
        
        for dataset_name, dataset_path in self.datasets.items():
            print(f"\nLoading improved RAG index for {dataset_name}...")
            retriever = RAGRetriever(collection_name=f"rag_{dataset_name}_v2")
            
            if not retriever.load(f"rag_{dataset_name}_v2"):
                print(f"ERROR: Improved index not found for {dataset_name}")
                print(f"Run: python scripts/build_improved_indexes.py")
                return
            
            stats = retriever.get_stats()
            print(f"  Loaded: {stats['document_count']} chunks")
            retrievers[dataset_name] = retriever
        
        tasks = []
        for dataset_name, dataset_path in self.datasets.items():
            dataset = load_from_disk(dataset_path)
            split = list(dataset.keys())[0]
            samples = list(dataset[split].select(range(min(self.num_samples, len(dataset[split])))))
            
            for model in self.models:
                tasks.append((model, dataset_name, retrievers[dataset_name], samples))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_model_rag, *task): task 
                for task in tasks
            }
            
            for future in as_completed(futures):
                model, dataset_name, results = future.result()
                key = f"{model}_{dataset_name}"
                
                with self.results_lock:
                    self.results['rag'][key] = results
        
        elapsed = time.time() - start_time
        self.timings['rag'] = elapsed
        
        self._save('rag')
        
        print(f"\nRAG completed in: {timedelta(seconds=int(elapsed))}")
    
    def _run_model_graphrag(self, model, dataset_name, retriever, samples):
        results = []
        client = OllamaClient(model, temperature=0.1)
        
        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            context = retriever.retrieve_context_string(example['question'], top_k=5)
            result = client.generate_with_context(
                question=example['question'],
                context=context,
                max_tokens=20000
            )
            
            if not result.get('success', False):
                continue
            
            results.append({
                'question': example['question'],
                'ground_truth': str(example.get('program_answer') or example.get('original_answer')),
                'answer': result['response'],
                'latency_ms': result['latency_ms']
            })
        
        return model, dataset_name, results
    
    def run_graphrag(self):
        print("\n" + "="*60)
        print("GRAPHRAG BENCHMARK")
        print("="*60)
        
        start_time = time.time()
        
        neo4j = Neo4jClient()
        stats = neo4j.get_graph_stats()
        
        print(f"\nGraph statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        if stats.get('Company', 0) < 200:
            print("\nWARNING: Graph seems small. Run: python scripts/build_improved_indexes.py")
        
        retriever = GraphRetriever(neo4j)
        self.results['graphrag'] = {}
        
        tasks = []
        for dataset_name, dataset_path in self.datasets.items():
            dataset = load_from_disk(dataset_path)
            split = list(dataset.keys())[0]
            samples = list(dataset[split].select(range(min(self.num_samples, len(dataset[split])))))
            
            for model in self.models:
                tasks.append((model, dataset_name, retriever, samples))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_model_graphrag, *task): task 
                for task in tasks
            }
            
            for future in as_completed(futures):
                model, dataset_name, results = future.result()
                key = f"{model}_{dataset_name}"
                
                with self.results_lock:
                    self.results['graphrag'][key] = results
        
        neo4j.close()
        
        elapsed = time.time() - start_time
        self.timings['graphrag'] = elapsed
        
        self._save('graphrag')
        
        print(f"\nGraphRAG completed in: {timedelta(seconds=int(elapsed))}")
    
    def _save(self, approach):
        output_dir = Path("results/metrics")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f"{approach}_fast.json", 'w') as f:
            json.dump(self.results[approach], f, indent=2)
        
        print(f"Saved: {approach}_fast.json")
    
    def print_summary(self):
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for approach in ['baseline', 'rag', 'graphrag']:
            if approach not in self.results:
                continue
            
            print(f"\n{approach.upper()}:")
            total_questions = 0
            total_time_ms = 0
            
            for key, results in self.results[approach].items():
                if results:
                    avg_lat = sum(r['latency_ms'] for r in results) / len(results)
                    total_time_ms += sum(r['latency_ms'] for r in results)
                    total_questions += len(results)
                    print(f"  {key}: {avg_lat:.0f}ms avg, {len(results)} questions")
            
            if total_questions > 0:
                wall_time = self.timings.get(approach, 0)
                print(f"\n  TOTAL: {total_questions} questions")
                print(f"  Inference time: {total_time_ms/1000:.1f}s")
                print(f"  Wall clock time: {timedelta(seconds=int(wall_time))}")


def main():
    overall_start = time.time()
    
    models = ["llama3.1:8b", "gemma3:12b", "qwen3:8b"]
    
    datasets = {
        "finqa": "data/benchmarks/t2-ragbench-FinQA",
        "convfinqa": "data/benchmarks/t2-ragbench-ConvFinQA",
        "tatqa": "data/benchmarks/t2-ragbench-TAT-DQA",
    }
    
    runner = ParallelBenchmarkRunner(
        models, 
        datasets, 
        num_samples=10,
        max_workers=3
    )
    
    print("BENCHMARK")
    print("="*60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNew faster models (8-12B parameters):")
    for model in models:
        print(f"  - {model}")
    print(f"\nBenchmark Configuration:")
    print(f"  Models: {len(models)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Questions per dataset: {runner.num_samples}")
    print(f"  Total questions: {len(models) * len(datasets) * runner.num_samples}")
    
    runner.run_baseline()
    runner.run_rag()
    runner.run_graphrag()
    runner.print_summary()
    
    total_elapsed = time.time() - overall_start
    
    print("\n" + "="*60)
    print("FAST BENCHMARK COMPLETE!")
    print("="*60)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {timedelta(seconds=int(total_elapsed))}")
    print(f"\nTiming breakdown:")
    print(f"  Baseline: {timedelta(seconds=int(runner.timings.get('baseline', 0)))}")
    print(f"  RAG: {timedelta(seconds=int(runner.timings.get('rag', 0)))}")
    print(f"  GraphRAG: {timedelta(seconds=int(runner.timings.get('graphrag', 0)))}")
    print(f"\nResults saved to results/metrics/")
    print("  - baseline_fast.json")
    print("  - rag_fast.json")
    print("  - graphrag_fast.json")


if __name__ == "__main__":
    main()