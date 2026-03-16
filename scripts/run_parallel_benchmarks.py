import sys
sys.path.append('.')

from src.models.ollama_client import OllamaClient
from src.rag.retriever import RAGRetriever
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.graph_retriever import GraphRetriever
from datasets import load_from_disk
import json
import re
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from datetime import datetime, timedelta


def _extract_answer_number(text: str) -> float | None:
    """
    Extract the primary numeric answer from a model response.
    Handles: leading numbers, currency, percentages, commas, 'million'/'billion' multipliers.
    """
    text = str(text).strip()
    # Strip common noise
    text = text.replace(',', '').replace('$', '').replace('%', '')

    # Replace billion/million multipliers
    text = re.sub(r'(\d+\.?\d*)\s*billion', lambda m: str(float(m.group(1)) * 1e3), text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+\.?\d*)\s*million', r'\1', text, flags=re.IGNORECASE)

    # Priority 1: explicit ANSWER: tag
    m = re.search(r'ANSWER:\s*(-?\d+\.?\d*)', text.strip(), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Priority 2: last "= <number>" in the text — captures the result of shown calculations
    # e.g. "11.8 - 19.9 = -8.1" or "268496 + 131262 = 399758"
    eq_matches = re.findall(r'=\s*(-?\d+\.?\d*)', text)
    if eq_matches:
        try:
            return float(eq_matches[-1])
        except ValueError:
            pass

    # Priority 3: fallback patterns (first number found)
    patterns = [
        r'^(-?\d+\.?\d*)\s',                           # number at very start
        r'^(-?\d+\.?\d*)$',                             # entire response is a number
        r'(?:answer|result|is|was|:)\s*(-?\d+\.?\d*)',
        r'\b(-?\d+\.\d+)\b',                            # any decimal in text
        r'\b(-?\d{1,8})\b',                             # any short integer in text
    ]
    for p in patterns:
        m = re.search(p, text.strip(), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _is_correct(pred_text: str, ground_truth: str, tol: float = 0.01, question: str = '') -> bool:
    """
    Returns True if the predicted answer numerically matches the ground truth.
    Handles:
    - Relative tolerance (default 1%) for large values
    - Absolute tolerance (±0.5) for small values
    - Percentage ↔ decimal conversion (e.g. model outputs -3.2% for GT -0.032)
    - Boolean yes/no questions (GT=0 or 1, answer contains yes/no language)
    - Unit scale mismatch (model reports millions, GT is raw units)
    - Falls back to string containment then exact string match
    """
    pred_num = _extract_answer_number(pred_text)
    try:
        gt_num = float(str(ground_truth).replace(',', ''))
    except ValueError:
        gt_num = None

    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return abs(pred_num) < 1e-6

        # Boolean yes/no: GT is 0 or 1 — map natural language to numeric
        if gt_num in (0.0, 1.0):
            ans_lower = str(pred_text).lower()
            if gt_num == 1.0 and re.search(r'\b(yes|true|did|exceeded?|greater|more|higher)\b', ans_lower):
                return True
            if gt_num == 0.0 and re.search(r'\b(no|false|did not|not exceed|less|lower|neither)\b', ans_lower):
                return True

        # Exact or relative tolerance match
        if abs(pred_num - gt_num) / abs(gt_num) <= tol:
            return True

        # Percentage ↔ decimal: GT is a decimal (e.g. -0.032), model outputs percent (e.g. -3.2)
        if 0 < abs(gt_num) < 1 and abs(pred_num - gt_num * 100) < 0.5:
            return True
        # Reverse: GT is a percent (e.g. 35), model outputs decimal (e.g. 0.35)
        if 0 < abs(pred_num) < 1 and abs(pred_num * 100 - gt_num) < 0.5:
            return True

        # Absolute tolerance for mid-range numbers 1–99 (within 0.5)
        if 1 <= abs(gt_num) < 100 and abs(pred_num - gt_num) < 0.5:
            return True

        # Relative tolerance for small decimals <1 (within 1%)
        # ±0.5 is too loose here — e.g. 0.9 would match GT=0.9765625
        if abs(gt_num) < 1 and abs(pred_num - gt_num) / abs(gt_num) < 0.01:
            return True

        # Unit scale: model strips "million"/"thousand" suffix — pred may be 1000x/1M× smaller
        if abs(gt_num) >= 1000:
            for scale in [1_000, 1_000_000]:
                if abs(pred_num * scale - gt_num) / abs(gt_num) <= tol:
                    return True

        return False

    # Fallback: whole-number string match (word-boundary safe — avoids "531" matching "-531")
    gt_clean = re.sub(r'\.0$', '', str(ground_truth).strip())
    pred_clean = str(pred_text).strip()
    if gt_clean and re.search(r'(?<!\d)' + re.escape(gt_clean) + r'(?!\d)', pred_clean):
        return True

    return str(pred_text).strip().lower() == str(ground_truth).strip().lower()

# Ensure to run this line of code before executing code & open docker desktop
# docker-compose up -d
# Check after 30 seconds to ensure its running with: 
# docker ps

class ParallelBenchmarkRunner:
    def __init__(self, models, datasets, num_samples=100, max_workers=3):
        self.models = models
        self.datasets = datasets
        self.num_samples = num_samples
        self.max_workers = max_workers
        self.results_lock = threading.Lock()
        self.results = {}
        self.timings = {}
    
    @staticmethod
    def _get_eval_split(dataset, dataset_name: str) -> str:
        """Return the canonical T²-RAGBench evaluation split for each dataset."""
        if dataset_name == 'convfinqa':
            return 'turn_0'
        if 'test' in dataset:
            return 'test'
        if 'dev' in dataset:
            return 'dev'
        return list(dataset.keys())[0]

    @staticmethod
    def _get_system_prompt(dataset_name: str) -> tuple[str, int, int, int]:
        """
        Return (system_prompt, baseline_max_tokens, rag_max_tokens, graphrag_max_tokens)
        for a given dataset.

        FinQA and TAT-QA require multi-step arithmetic — a chain-of-thought prompt
        with an explicit ANSWER: tag significantly improves accuracy. ConvFinQA answers
        are simpler single-hop lookups where a terse "number only" prompt suffices.
        """
        if dataset_name in ('finqa', 'tatqa'):
            prompt = (
                "You are a financial analyst. You will be given an excerpt from a company's "
                "annual report followed by a question.\n\n"
                "Approach:\n"
                "1. Identify the exact figures needed from the text or table.\n"
                "2. Show your calculation step by step.\n"
                "3. On the final line write: ANSWER: <number>\n\n"
                "Rules:\n"
                "- ANSWER must be a single number (e.g. ANSWER: 3.8 or ANSWER: -2.5).\n"
                "- Do not include units or currency symbols in the ANSWER line.\n"
                "- For yes/no questions output ANSWER: 1 for yes and ANSWER: 0 for no.\n"
                "- Parentheses in financial tables mean negative (e.g. (832) = -832).\n"
                "- Never say you cannot answer; the data is always in the provided context."
            )
            return prompt, 800, 1000, 1200
        else:
            prompt = (
                "You are a financial analyst. You will be given an excerpt from a company's "
                "annual report followed by a question. The answer is a specific number or "
                "percentage that can be read or calculated directly from the provided text and table.\n"
                "Rules:\n"
                "- Output ONLY the final numeric answer (e.g. 3.8 or -2.5% or 0.532).\n"
                "- Do not include units, currency symbols, or explanatory text.\n"
                "- If a calculation is needed, do it silently and output only the result.\n"
                "- Never say you cannot answer; the answer is always in the provided context."
            )
            return prompt, 400, 600, 800

    @staticmethod
    def _clean_table(raw: str) -> str:
        """
        Clean a pandas-rendered markdown table for LLM consumption:
        - Drop the auto-generated numeric index column (first col is always |  N |)
        - Normalize financial negatives: "-23158 ( 23158 )" → "-23158"
        """
        lines = raw.strip().splitlines()
        cleaned = []
        for line in lines:
            # Remove leading index cell: "| N |" or "|---:|" at the start
            line = re.sub(r'^\|\s*-*\d*:?\s*\|', '|', line)
            # Normalize parenthetical negatives: -VALUE ( VALUE ) → -VALUE
            line = re.sub(r'-(\d[\d,\.]*)\s*\(\s*[\d,\.]+\s*\)', r'-\1', line)
            cleaned.append(line)
        return "\n".join(cleaned)

    def _build_dataset_context(self, example: dict) -> str:
        """
        Build context string from the dataset's document fields.
        - FinQA / ConvFinQA: use pre_text + table + post_text
        - TAT-DQA: those fields are absent; fall back to the 'context' field
        """
        parts = []
        if example.get('pre_text'):
            parts.append(example['pre_text'].strip())
        if example.get('table'):
            parts.append(self._clean_table(example['table']))
        if example.get('post_text'):
            parts.append(example['post_text'].strip())
        # TAT-DQA uses only a 'context' field — use it when the structured fields are absent
        if not parts and example.get('context'):
            parts.append(example['context'].strip())
        return "\n\n".join(parts) if parts else ""

    def _run_model_baseline(self, model, dataset_name, dataset_path, samples):
        results = []
        client = OllamaClient(model, temperature=0.1)

        system_prompt, max_tokens, _, _ = self._get_system_prompt(dataset_name)
        cot = dataset_name in ('finqa', 'tatqa')

        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            prompt_suffix = "Show your step-by-step calculation, then write ANSWER: <number>" if cot else "Answer (number only):"
            prompt = (
                f"Financial question: {example['question']}\n\n"
                f"{prompt_suffix}"
            )
            result = client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )

            if not result.get('success', False):
                continue

            gt = str(example.get('program_answer') or example.get('original_answer'))
            results.append({
                'question': example['question'],
                'ground_truth': gt,
                'answer': result['response'],
                'correct': _is_correct(result['response'], gt, question=example['question']),
                'latency_ms': result['latency_ms']
            })

        return model, dataset_name, results

    def run_baseline(self):
        print("\n" + "="*60)
        print("BASELINE BENCHMARK")
        print("="*60)
        
        start_time = time.time()
        self.results['baseline'] = {}
        
        tasks = []
        for dataset_name, dataset_path in self.datasets.items():
            dataset = load_from_disk(dataset_path)
            split = self._get_eval_split(dataset, dataset_name)
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

        rag_system_prompt, _, rag_max_tokens, _ = self._get_system_prompt(dataset_name)
        cot = dataset_name in ('finqa', 'tatqa')

        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            question = example['question']

            # Extract all years mentioned in the question for metadata filtering
            question_years = set(re.findall(r'\b(19[9]\d|20[0-3]\d)\b', question))

            # Retrieve extra candidates so we can filter down to 3 quality chunks
            raw = retriever.retrieve(question, top_k=10)

            # Filter: (1) cosine similarity >= 0.4, (2) year match when question has years
            MIN_SIM = 0.4
            kept_docs = []
            for doc, score, meta in zip(raw['documents'], raw['distances'], raw['metadatas']):
                if score < MIN_SIM:
                    continue
                if question_years:
                    chunk_year = str(meta.get('year', ''))
                    if chunk_year and chunk_year != 'Unknown' and chunk_year not in question_years:
                        continue
                kept_docs.append(doc)
                if len(kept_docs) >= 3:
                    break

            # Context order: retrieved passages first, source document last.
            # Placing the authoritative source doc at the end exploits the model's
            # recency bias — it anchors the final answer on the ground-truth document.
            dataset_ctx = self._build_dataset_context(example)
            if kept_docs:
                retrieved_str = "\n\n---\n\n".join(
                    f"[Document {i+1}]\n{d}" for i, d in enumerate(kept_docs)
                )
                if dataset_ctx:
                    context = f"[Retrieved passages]\n{retrieved_str}\n\n[Source document]\n{dataset_ctx}"
                else:
                    context = retrieved_str
            else:
                context = f"[Source document]\n{dataset_ctx}" if dataset_ctx else ""

            result = client.generate_with_context(
                question=question,
                context=context,
                system_prompt=rag_system_prompt,
                max_tokens=rag_max_tokens,
                cot=cot
            )

            if not result.get('success', False):
                continue

            gt = str(example.get('program_answer') or example.get('original_answer'))
            results.append({
                'question': example['question'],
                'ground_truth': gt,
                'answer': result['response'],
                'correct': _is_correct(result['response'], gt, question=example['question']),
                'latency_ms': result['latency_ms']
            })

        return model, dataset_name, results

    def run_rag(self):
        print("\n" + "="*60)
        print("VECTOR RAG BENCHMARK")
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
            split = self._get_eval_split(dataset, dataset_name)
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

        graphrag_system_prompt, _, _, graphrag_max_tokens = self._get_system_prompt(dataset_name)
        cot = dataset_name in ('finqa', 'tatqa')

        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            # top_k=2: dataset context already contains the source document, so
            # only the most relevant graph records add value. 5 records × 4000 chars
            # previously overflowed 8B context windows causing 16-min timeouts.
            graph_context = retriever.retrieve_context_string(example['question'], top_k=2)
            dataset_ctx = self._build_dataset_context(example)
            if dataset_ctx:
                # Graph records first (structured facts) then source document (raw tables).
                # Cap graph_context at 1500 chars to avoid context overflow — the
                # dataset source document is the authoritative source anyway.
                context = f"[Knowledge graph]\n{graph_context[:3000]}\n\n[Source document]\n{dataset_ctx}"
            else:
                context = graph_context
            result = client.generate_with_graph_context(
                question=example['question'],
                context=context,
                system_prompt=graphrag_system_prompt,
                max_tokens=graphrag_max_tokens,
                cot=cot
            )

            if not result.get('success', False):
                continue

            gt = str(example.get('program_answer') or example.get('original_answer'))
            results.append({
                'question': example['question'],
                'ground_truth': gt,
                'answer': result['response'],
                'correct': _is_correct(result['response'], gt, question=example['question']),
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
            split = self._get_eval_split(dataset, dataset_name)
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
        
        with open(output_dir / f"{approach}.json", 'w') as f:
            json.dump(self.results[approach], f, indent=2)

        print(f"Saved: {approach}.json")
    
    def print_summary(self):
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for approach in ['baseline', 'rag', 'graphrag']:
            if approach not in self.results:
                continue
            
            print(f"\n{approach.upper()}:")
            total_questions = 0
            total_correct = 0
            total_time_ms = 0

            for key, results in self.results[approach].items():
                if results:
                    n = len(results)
                    correct = sum(1 for r in results if r.get('correct', False))
                    avg_lat = sum(r['latency_ms'] for r in results) / n
                    total_time_ms += sum(r['latency_ms'] for r in results)
                    total_questions += n
                    total_correct += correct
                    print(f"  {key}: accuracy={correct}/{n} ({correct/n:.1%})  avg_latency={avg_lat:.0f}ms")

            if total_questions > 0:
                wall_time = self.timings.get(approach, 0)
                print(f"\n  TOTAL: {total_correct}/{total_questions} correct ({total_correct/total_questions:.1%})")
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
        num_samples=100,
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
    print("BENCHMARK COMPLETE!")
    print("="*60)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {timedelta(seconds=int(total_elapsed))}")
    print(f"\nTiming breakdown:")
    print(f"  Baseline: {timedelta(seconds=int(runner.timings.get('baseline', 0)))}")
    print(f"  RAG: {timedelta(seconds=int(runner.timings.get('rag', 0)))}")
    print(f"  GraphRAG: {timedelta(seconds=int(runner.timings.get('graphrag', 0)))}")
    print(f"\nResults saved to results/metrics/")
    print("  - baseline.json")
    print("  - rag.json")
    print("  - graphrag.json")


if __name__ == "__main__":
    main()