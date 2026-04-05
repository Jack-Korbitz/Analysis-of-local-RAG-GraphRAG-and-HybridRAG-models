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
    def __init__(self, models, datasets, num_samples=200, max_workers=3):
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
    def _get_system_prompt(dataset_name: str) -> tuple[str, str, str, int, int, int, int]:
        """
        Return (prompt, graphrag_prompt, hybrid_prompt,
                baseline_max_tokens, rag_max_tokens, graphrag_max_tokens, hybrid_max_tokens)
        for a given dataset. Baseline and RAG share the same prompt; GraphRAG and Hybrid
        get their own prompts that acknowledge partial/structured context.

        FinQA and TAT-QA require multi-step arithmetic — a chain-of-thought prompt
        with an explicit ANSWER: tag significantly improves accuracy. ConvFinQA answers
        are simpler single-hop lookups where a terse "number only" prompt suffices.
        """
        if dataset_name in ('finqa', 'tatqa'):
            prompt = (
                "You are a financial analyst. You will be given financial data followed by a question.\n\n"
                "Approach:\n"
                "1. Identify the exact figures needed from the provided data.\n"
                "2. Show your calculation step by step.\n"
                "3. On the final line write: ANSWER: <number>\n\n"
                "Rules:\n"
                "- ANSWER must be a single number (e.g. ANSWER: 3.8 or ANSWER: -2.5).\n"
                "- Do not include units or currency symbols in the ANSWER line.\n"
                "- For yes/no questions output ANSWER: 1 for yes and ANSWER: 0 for no.\n"
                "- Parentheses in financial tables mean negative (e.g. (832) = -832).\n"
                "- Only use figures that are directly about the company and time period in the question.\n"
                "- You MUST always output an ANSWER line. If the exact figures are not present, "
                "calculate from the closest available data or output your best numeric estimate."
            )
            graphrag_prompt = (
                "You are a financial analyst. You will be given structured financial records "
                "retrieved from a knowledge graph, followed by a question.\n\n"
                "The records may contain metric values, document excerpts, or both. "
                "They may not always contain every figure needed — use what is available.\n\n"
                "Approach:\n"
                "1. Identify relevant figures from the provided records.\n"
                "2. Show your calculation step by step.\n"
                "3. On the final line write: ANSWER: <number>\n\n"
                "Rules:\n"
                "- ANSWER must be a single number (e.g. ANSWER: 3.8 or ANSWER: -2.5).\n"
                "- Do not include units or currency symbols in the ANSWER line.\n"
                "- For yes/no questions output ANSWER: 1 for yes and ANSWER: 0 for no.\n"
                "- Parentheses in financial tables mean negative (e.g. (832) = -832).\n"
                "- If the data contains the relevant figures, compute and output the answer.\n"
                "- If the data does NOT contain any relevant figures, output: ANSWER: N/A"
            )
            hybrid_prompt = (
                "You are a financial analyst. You will be given TWO sources of financial data:\n"
                "1. Structured records from a knowledge graph (metrics, values)\n"
                "2. Retrieved document passages from a vector search\n\n"
                "The structured records are more precise but may be incomplete. "
                "The document passages provide broader context but may contain irrelevant text.\n"
                "Cross-reference both sources — prefer structured values when available, "
                "fall back to document passages for details not in the records.\n\n"
                "Approach:\n"
                "1. Check structured records first for exact figures.\n"
                "2. Use document passages to fill gaps or verify.\n"
                "3. Show your calculation step by step.\n"
                "4. On the final line write: ANSWER: <number>\n\n"
                "Rules:\n"
                "- ANSWER must be a single number (e.g. ANSWER: 3.8 or ANSWER: -2.5).\n"
                "- Do not include units or currency symbols in the ANSWER line.\n"
                "- For yes/no questions output ANSWER: 1 for yes and ANSWER: 0 for no.\n"
                "- Parentheses in financial tables mean negative (e.g. (832) = -832).\n"
                "- You MUST always output an ANSWER line — use your best numeric estimate."
            )
            return prompt, graphrag_prompt, hybrid_prompt, 800, 1400, 1200, 1400
        else:
            prompt = (
                "You are a financial analyst. You will be given financial data followed by a question.\n"
                "Rules:\n"
                "- Output ONLY the final numeric answer (e.g. 3.8 or -2.5% or 0.532).\n"
                "- Do not include units, currency symbols, or explanatory text.\n"
                "- If a calculation is needed, do it silently and output only the result.\n"
                "- Only use figures directly about the company and year in the question.\n"
                "- You MUST always output a number. If the exact figures are not present, "
                "output your best numeric estimate."
            )
            graphrag_prompt = (
                "You are a financial analyst. You will be given structured financial records "
                "retrieved from a knowledge graph, followed by a question.\n"
                "Rules:\n"
                "- Output ONLY the final numeric answer (e.g. 3.8 or -2.5% or 0.532).\n"
                "- Do not include units, currency symbols, or explanatory text.\n"
                "- If a calculation is needed, do it silently and output only the result.\n"
                "- If the data does NOT contain any relevant figures, output: N/A"
            )
            hybrid_prompt = (
                "You are a financial analyst. You will be given TWO sources of data:\n"
                "1. Structured records from a knowledge graph\n"
                "2. Retrieved document passages from vector search\n"
                "Cross-reference both — prefer structured values when available.\n"
                "Rules:\n"
                "- Output ONLY the final numeric answer (e.g. 3.8 or -2.5% or 0.532).\n"
                "- Do not include units, currency symbols, or explanatory text.\n"
                "- If a calculation is needed, do it silently and output only the result.\n"
                "- You MUST always output a number — use your best estimate."
            )
            return prompt, graphrag_prompt, hybrid_prompt, 400, 900, 800, 1000


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

        system_prompt, _, _, max_tokens, _, _, _ = self._get_system_prompt(dataset_name)
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

    def _run_model_oracle(self, model, dataset_name, dataset_path, samples):
        results = []
        client = OllamaClient(model, temperature=0.1)
        system_prompt, _, _, _, max_tokens, _, _ = self._get_system_prompt(dataset_name)
        cot = dataset_name in ('finqa', 'tatqa')

        for example in tqdm(samples, desc=f"{model} - {dataset_name} [Oracle]",
                            position=hash(model) % 10):
            gt = str(example.get('program_answer') or example.get('original_answer'))
            context = self._build_dataset_context(example)
            gt_norm = gt.replace(',', '').strip()
            retrieval_metrics = {
                'retrieved_count': 1 if context else 0,
                'answer_in_context': gt_norm in context.replace(',', '') if context else False
            }
            if context:
                result = client.generate_with_context(
                    question=example['question'], context=context,
                    system_prompt=system_prompt, max_tokens=max_tokens, cot=cot)
            else:
                prompt_suffix = ("Show your step-by-step calculation, then write ANSWER: <number>"
                                 if cot else "Answer (number only):")
                result = client.generate(
                    prompt=f"Financial question: {example['question']}\n\n{prompt_suffix}",
                    system_prompt=system_prompt, max_tokens=max_tokens)
            if not result.get('success', False):
                continue
            results.append({
                'question': example['question'],
                'ground_truth': gt,
                'answer': result['response'],
                'correct': _is_correct(result['response'], gt, question=example['question']),
                'latency_ms': result['latency_ms'],
                'retrieval_metrics': retrieval_metrics
            })
        return model, dataset_name, results

    def run_oracle(self):
        print("\n" + "="*60)
        print("ORACLE (PERFECT CONTEXT) BENCHMARK")
        print("="*60)
        start_time = time.time()
        self.results['oracle'] = {}
        tasks = []
        for dataset_name, dataset_path in self.datasets.items():
            dataset = load_from_disk(dataset_path)
            split = self._get_eval_split(dataset, dataset_name)
            samples = list(dataset[split].select(range(min(self.num_samples, len(dataset[split])))))
            for model in self.models:
                tasks.append((model, dataset_name, dataset_path, samples))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._run_model_oracle, *task): task for task in tasks}
            for future in as_completed(futures):
                model, dataset_name, results = future.result()
                key = f"{model}_{dataset_name}"
                with self.results_lock:
                    self.results['oracle'][key] = results
        elapsed = time.time() - start_time
        self.timings['oracle'] = elapsed
        self._save('oracle')
        print(f"\nOracle completed in: {timedelta(seconds=int(elapsed))}")

    @staticmethod
    def _expand_query(query: str) -> str:
        """
        Prepend BGE retrieval instruction — aligns query embedding space with
        document embedding space for bge-large-en-v1.5.
        Synonym expansion was removed: it created bloated vectors that matched
        too broadly and caused cross-company contamination.
        """
        return 'Represent this sentence for searching relevant passages: ' + query

    @staticmethod
    def _condense_query(query: str) -> str:
        """
        Strip filler phrases before embedding so the query vector focuses on
        financial entities and metrics rather than question scaffolding.
        Falls back to original if condensed result is too short.
        """
        filler = re.compile(
            r'\b(what was|what were|what is|what are|how much|how many|'
            r'please tell me|can you tell me|i would like to know|'
            r'the total|the amount of|the value of|the number of)\b',
            re.IGNORECASE
        )
        condensed = filler.sub('', query)
        condensed = re.sub(r'\s+', ' ', condensed).strip()
        return condensed if len(condensed) > 15 else query

    def _run_model_rag(self, model, dataset_name, retriever, samples):
        results = []
        client = OllamaClient(model, temperature=0.1)

        rag_system_prompt, _, _, _, rag_max_tokens, _, _ = self._get_system_prompt(dataset_name)
        cot = dataset_name in ('finqa', 'tatqa')

        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            question = example['question']
            gt = str(example.get('program_answer') or example.get('original_answer'))

            # Extract all years mentioned in the question for metadata filtering
            question_years = set(re.findall(r'\b(19[9]\d|20[0-3]\d)\b', question))
            question_lower = question.lower()

            # Condense (strip filler) then prepend BGE instruction
            embed_query = self._expand_query(self._condense_query(question))

            raw = retriever.retrieve(embed_query, top_k=20)

            # Company pre-filter: detect which (if any) retrieved companies appear
            # in the question. If identifiable, reject chunks from other companies.
            # This is the #1 RAG failure mode: retrieving Entergy data for Intel questions.
            _sfx = re.compile(
                r',?\s*(inc\.?|corp\.?|llc|ltd\.?|incorporated|corporation|company|group|holdings)$',
                re.IGNORECASE
            )
            def _company_matches(company: str) -> bool:
                if not company or company == 'Unknown' or len(company) < 4:
                    return True   # no info → don't filter out
                c_norm = _sfx.sub('', company).strip().lower()
                return c_norm in question_lower or company.lower() in question_lower

            # Only enable company filter if at least one retrieved candidate IS from
            # the right company (i.e. we can identify the company from the question).
            company_filter_active = any(
                _company_matches(meta.get('company', ''))
                for _, _, meta in zip(raw['documents'], raw['distances'], raw['metadatas'])
                if meta.get('company', 'Unknown') != 'Unknown'
            )

            # Filter: (1) similarity >= 0.50, (2) company match, (3) soft year match
            # Year filter is soft: a doc whose year is within ±1 of any questioned year
            # is kept — annual reports span fiscal years and the exact year tag often
            # lags the content by one year (e.g. a 2015 filing discusses 2014 figures).
            MIN_SIM = 0.40
            kept_docs = []
            kept_scores = []
            question_years_int = {int(y) for y in question_years}
            for doc, score, meta in zip(raw['documents'], raw['distances'], raw['metadatas']):
                if score < MIN_SIM:
                    continue
                if company_filter_active and not _company_matches(meta.get('company', '')):
                    continue
                if question_years_int:
                    chunk_year_str = str(meta.get('year', ''))
                    if chunk_year_str and chunk_year_str != 'Unknown':
                        try:
                            chunk_year_int = int(chunk_year_str)
                            # Keep if within ±1 of any year in the question
                            if not any(abs(chunk_year_int - qy) <= 1 for qy in question_years_int):
                                continue
                        except ValueError:
                            pass  # unparseable year → don't filter out
                kept_docs.append(doc)
                kept_scores.append(score)
                if len(kept_docs) >= 8:
                    break

            # Question-keyword fallback: when embedding+filter yields nothing, search
            # stored question metadata for keyword overlap. Mirrors GraphRAG strategy 5a.
            # Works well for FinQA/TAT-DQA where short questions confuse cosine retrieval.
            if not kept_docs:
                stopwords = {'what', 'were', 'was', 'the', 'from', 'that', 'this', 'with',
                             'have', 'for', 'and', 'did', 'does', 'how', 'much', 'many',
                             'percent', 'percentage', 'change', 'total', 'which', 'year'}
                key_terms = [w for w in re.findall(r'\b[a-zA-Z]{5,}\b', question.lower())
                             if w not in stopwords][:5]

                # Pull all raw candidates (no sim floor) and match by question keywords
                raw_all = retriever.retrieve(embed_query, top_k=100)
                for doc, score, meta in zip(raw_all['documents'], raw_all['distances'], raw_all['metadatas']):
                    stored_q = meta.get('question', '').lower()
                    if not stored_q:
                        continue
                    # Require ≥ half of key_terms to appear in stored question
                    if key_terms and sum(1 for t in key_terms if t in stored_q) >= max(2, len(key_terms) // 2):
                        if company_filter_active and not _company_matches(meta.get('company', '')):
                            continue
                        kept_docs.append(doc)
                        kept_scores.append(score)
                        if len(kept_docs) >= 3:
                            break

            # Closed-book: retrieved passages only — no source document.
            # This isolates what FAISS retrieval contributes vs. provided context.
            context = "\n\n---\n\n".join(
                f"[Document {i+1}]\n{d}" for i, d in enumerate(kept_docs)
            ) if kept_docs else ""

            # answer_in_context: proxy for retrieval recall — did we surface the right content?
            gt_norm = gt.replace(',', '').strip()
            retrieval_metrics = {
                'retrieved_count': len(kept_docs),
                'avg_similarity': round(sum(kept_scores) / len(kept_scores), 4) if kept_scores else 0.0,
                'answer_in_context': gt_norm in context.replace(',', '') if context else False
            }

            if context:
                result = client.generate_with_context(
                    question=question,
                    context=context,
                    system_prompt=rag_system_prompt,
                    max_tokens=rag_max_tokens,
                    cot=cot
                )
            else:
                # No chunks passed filtering — fall back to parametric knowledge
                prompt_suffix = "Show your step-by-step calculation, then write ANSWER: <number>" if cot else "Answer (number only):"
                result = client.generate(
                    prompt=f"Financial question: {question}\n\n{prompt_suffix}",
                    system_prompt=rag_system_prompt,
                    max_tokens=rag_max_tokens
                )

            if not result.get('success', False):
                continue

            results.append({
                'question': example['question'],
                'ground_truth': gt,
                'answer': result['response'],
                'correct': _is_correct(result['response'], gt, question=example['question']),
                'latency_ms': result['latency_ms'],
                'retrieval_metrics': retrieval_metrics,
                'retrieved_chunks': kept_docs,
            })

        return model, dataset_name, results

    def run_rag(self):
        print("\n" + "="*60)
        print("VECTOR RAG BENCHMARK")
        print("="*60)
        
        start_time = time.time()
        self.results['rag'] = {}
        retrievers = {}
        
        QUERY_MODEL = "BAAI/bge-large-en-v1.5"
        for dataset_name, dataset_path in self.datasets.items():
            print(f"\nLoading RAG index for {dataset_name}...")

            # Verify index was built with the same embedding model — a mismatch
            # means all similarity scores are meaningless (different vector spaces).
            marker = Path(f"data/vector_db/rag_{dataset_name}_v4/_model.json")
            if marker.exists():
                stored = json.loads(marker.read_text()).get("embedding_model")
                if stored != QUERY_MODEL:
                    print(f"  ERROR: index built with '{stored}', querying with '{QUERY_MODEL}'")
                    print(f"  Run: python scripts/build_improved_indexes.py")
                    return
                print(f"  Model verified: {stored}")
            else:
                print(f"  WARNING: no model marker found — index may be stale. Rebuild recommended.")

            retriever = RAGRetriever(embedding_model_name=QUERY_MODEL,
                                     collection_name=f"rag_{dataset_name}_v4")

            if not retriever.load(f"rag_{dataset_name}_v4"):
                print(f"  ERROR: Index not found for {dataset_name}")
                print(f"  Run: python scripts/build_improved_indexes.py")
                return

            stats = retriever.get_stats()
            print(f"  Loaded: {stats['document_count']} documents")
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

        _, graphrag_system_prompt, _, _, _, graphrag_max_tokens, _ = self._get_system_prompt(dataset_name)
        cot = dataset_name in ('finqa', 'tatqa')

        for example in tqdm(samples, desc=f"{model} - {dataset_name}", position=hash(model) % 10):
            gt = str(example.get('program_answer') or example.get('original_answer'))

            # Call retrieve_by_entity directly so we can capture strategy metadata,
            # then format as context string separately.
            graph_records = retriever.retrieve_by_entity(example['question'], top_k=5)
            graph_context = retriever.format_context(graph_records, example['question'])

            # answer_in_context: proxy for retrieval recall
            gt_norm = gt.replace(',', '').strip()
            strategies = [r.get('strategy', 'unknown') for r in graph_records]
            retrieval_metrics = {
                'retrieved_count': len(graph_records),
                'retrieval_strategy': strategies[0] if strategies else 'none',
                'answer_in_context': gt_norm in graph_context.replace(',', '') if graph_records else False
            }

            # Closed-book: graph context only — no source document.
            # This is the true test of what the knowledge graph contributes.
            context = graph_context[:3000]

            result = client.generate_with_graph_context(
                question=example['question'],
                context=context,
                system_prompt=graphrag_system_prompt,
                max_tokens=graphrag_max_tokens,
                cot=cot
            )

            if not result.get('success', False):
                continue

            results.append({
                'question': example['question'],
                'ground_truth': gt,
                'answer': result['response'],
                'correct': _is_correct(result['response'], gt, question=example['question']),
                'latency_ms': result['latency_ms'],
                'retrieval_metrics': retrieval_metrics,
                'retrieved_chunks': [graph_context[:3000]],
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

    def _run_model_hybrid(self, model, dataset_name, rag_retriever, graph_retriever, samples):
        """Hybrid: combine graph records + FAISS chunks into one context."""
        results = []
        client = OllamaClient(model, temperature=0.1)

        _, _, hybrid_prompt, _, _, _, hybrid_max_tokens = self._get_system_prompt(dataset_name)
        cot = dataset_name in ('finqa', 'tatqa')

        for example in tqdm(samples, desc=f"{model} - {dataset_name} [Hybrid]",
                            position=hash(model) % 10):
            question = example['question']
            gt = str(example.get('program_answer') or example.get('original_answer'))
            gt_norm = gt.replace(',', '').strip()

            # --- Graph retrieval ---
            graph_records = graph_retriever.retrieve_by_entity(question, top_k=5)
            graph_context = graph_retriever.format_context(graph_records, question)
            graph_strategy = graph_records[0].get('strategy', 'none') if graph_records else 'none'

            # --- Vector retrieval (simplified from _run_model_rag) ---
            question_years = set(re.findall(r'\b(19[9]\d|20[0-3]\d)\b', question))
            question_lower = question.lower()
            embed_query = self._expand_query(self._condense_query(question))
            raw = rag_retriever.retrieve(embed_query, top_k=20)

            _sfx = re.compile(
                r',?\s*(inc\.?|corp\.?|llc|ltd\.?|incorporated|corporation|company|group|holdings)$',
                re.IGNORECASE
            )
            def _company_matches(company: str) -> bool:
                if not company or company == 'Unknown' or len(company) < 4:
                    return True
                c_norm = _sfx.sub('', company).strip().lower()
                return c_norm in question_lower or company.lower() in question_lower

            company_filter_active = any(
                _company_matches(meta.get('company', ''))
                for _, _, meta in zip(raw['documents'], raw['distances'], raw['metadatas'])
                if meta.get('company', 'Unknown') != 'Unknown'
            )

            MIN_SIM = 0.40
            kept_docs = []
            kept_scores = []
            question_years_int = {int(y) for y in question_years}
            for doc, score, meta in zip(raw['documents'], raw['distances'], raw['metadatas']):
                if score < MIN_SIM:
                    continue
                if company_filter_active and not _company_matches(meta.get('company', '')):
                    continue
                if question_years_int:
                    chunk_year_str = str(meta.get('year', ''))
                    if chunk_year_str and chunk_year_str != 'Unknown':
                        try:
                            chunk_year_int = int(chunk_year_str)
                            if not any(abs(chunk_year_int - qy) <= 1 for qy in question_years_int):
                                continue
                        except ValueError:
                            pass
                kept_docs.append(doc)
                kept_scores.append(score)
                if len(kept_docs) >= 5:
                    break

            # --- Combine contexts ---
            # Graph records first (structured, precise), then vector passages (broader).
            # Cap each section to prevent context overflow on 8B models.
            parts = []
            if graph_records:
                parts.append(f"[KNOWLEDGE GRAPH RECORDS]\n{graph_context[:2000]}")
            if kept_docs:
                rag_str = "\n\n---\n\n".join(
                    f"[Passage {i+1}]\n{d}" for i, d in enumerate(kept_docs)
                )
                parts.append(f"[RETRIEVED PASSAGES]\n{rag_str[:2000]}")

            combined_context = "\n\n".join(parts) if parts else ""

            # Retrieval metrics
            all_context = combined_context.replace(',', '')
            retrieval_metrics = {
                'retrieved_count': len(graph_records) + len(kept_docs),
                'graph_records': len(graph_records),
                'rag_chunks': len(kept_docs),
                'retrieval_strategy': graph_strategy,
                'avg_similarity': round(sum(kept_scores) / len(kept_scores), 4) if kept_scores else 0.0,
                'answer_in_context': gt_norm in all_context if combined_context else False
            }

            if combined_context:
                if cot:
                    prompt = f"""Context:
{combined_context}

Question: {question}

Show your step-by-step calculation, then write ANSWER: <number>"""
                else:
                    prompt = f"""Context:
{combined_context}

Question: {question}

ANSWER:"""
                result = client.generate(prompt, hybrid_prompt, hybrid_max_tokens)
            else:
                prompt_suffix = ("Show your step-by-step calculation, then write ANSWER: <number>"
                                 if cot else "Answer (number only):")
                result = client.generate(
                    prompt=f"Financial question: {question}\n\n{prompt_suffix}",
                    system_prompt=hybrid_prompt, max_tokens=hybrid_max_tokens)

            if not result.get('success', False):
                continue

            results.append({
                'question': example['question'],
                'ground_truth': gt,
                'answer': result['response'],
                'correct': _is_correct(result['response'], gt, question=example['question']),
                'latency_ms': result['latency_ms'],
                'retrieval_metrics': retrieval_metrics,
                'retrieved_chunks': [combined_context[:3000]],
            })

        return model, dataset_name, results

    def run_hybrid(self):
        """Run hybrid RAG+GraphRAG benchmark — combines both retrievers."""
        print("\n" + "="*60)
        print("HYBRID (RAG + GraphRAG) BENCHMARK")
        print("="*60)

        start_time = time.time()

        # Initialize graph retriever
        neo4j = Neo4jClient()
        graph_retriever = GraphRetriever(neo4j)

        # Initialize RAG retrievers (one per dataset)
        QUERY_MODEL = "BAAI/bge-large-en-v1.5"
        rag_retrievers = {}
        for dataset_name in self.datasets:
            retriever = RAGRetriever(embedding_model_name=QUERY_MODEL,
                                     collection_name=f"rag_{dataset_name}_v4")
            if not retriever.load(f"rag_{dataset_name}_v4"):
                print(f"  ERROR: RAG index not found for {dataset_name}")
                neo4j.close()
                return
            stats = retriever.get_stats()
            print(f"  {dataset_name}: {stats['document_count']} chunks loaded")
            rag_retrievers[dataset_name] = retriever

        self.results['hybrid'] = {}

        tasks = []
        for dataset_name, dataset_path in self.datasets.items():
            dataset = load_from_disk(dataset_path)
            split = self._get_eval_split(dataset, dataset_name)
            samples = list(dataset[split].select(range(min(self.num_samples, len(dataset[split])))))
            for model in self.models:
                tasks.append((model, dataset_name, rag_retrievers[dataset_name],
                              graph_retriever, samples))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_model_hybrid, *task): task
                for task in tasks
            }
            for future in as_completed(futures):
                model, dataset_name, results = future.result()
                key = f"{model}_{dataset_name}"
                with self.results_lock:
                    self.results['hybrid'][key] = results

        neo4j.close()

        elapsed = time.time() - start_time
        self.timings['hybrid'] = elapsed
        self._save('hybrid')
        print(f"\nHybrid completed in: {timedelta(seconds=int(elapsed))}")

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
        
        for approach in ['baseline', 'oracle', 'rag', 'graphrag', 'hybrid']:
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
        num_samples=200,
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
    
    # runner.run_baseline()
    # runner.run_oracle()
    # runner.run_rag()
    # runner.run_graphrag()
    runner.run_hybrid()
    runner.print_summary()
    
    total_elapsed = time.time() - overall_start
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {timedelta(seconds=int(total_elapsed))}")
    print(f"\nTiming breakdown:")
    print(f"  Baseline:    {timedelta(seconds=int(runner.timings.get('baseline', 0)))}")
    print(f"  Oracle:      {timedelta(seconds=int(runner.timings.get('oracle', 0)))}")
    print(f"  RAG:         {timedelta(seconds=int(runner.timings.get('rag', 0)))}")
    print(f"  GraphRAG:    {timedelta(seconds=int(runner.timings.get('graphrag', 0)))}")
    print(f"  Hybrid:      {timedelta(seconds=int(runner.timings.get('hybrid', 0)))}")
    print(f"\nResults saved to results/metrics/")
    print("  - baseline.json")
    print("  - oracle.json")
    print("  - rag.json")
    print("  - graphrag.json")
    print("  - hybrid.json")


if __name__ == "__main__":
    main()