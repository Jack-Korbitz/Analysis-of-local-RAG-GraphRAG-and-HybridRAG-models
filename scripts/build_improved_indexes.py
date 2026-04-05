import sys
import json
sys.path.append('.')

from src.rag.retriever import RAGRetriever
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.graph_builder import GraphBuilder
from src.utils.chunking import DocumentChunker
from datasets import load_from_disk, concatenate_datasets
from pathlib import Path
from tqdm import tqdm

RAG_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def _extract_full_doc(example: dict) -> str:
    """Return pre_text + table + post_text as one unit (or context for TAT-DQA)."""
    parts = []
    if example.get('pre_text'):
        parts.append(str(example['pre_text']).strip())
    if example.get('table'):
        parts.append(str(example['table']).strip())
    if example.get('post_text'):
        parts.append(str(example['post_text']).strip())
    if not parts and example.get('context'):
        parts.append(str(example['context']).strip())
    return "\n\n".join(parts)


def build_rag_indexes(datasets, max_examples=None):
    """
    v4: table-aware chunking — each example is split into focused chunks.
    Tables are never split mid-row; prose is chunked at ~800 chars with 150-char
    overlap. Each chunk inherits company/year/question metadata from its parent
    document so the benchmark runner's company+year filter still works.

    max_examples=None indexes the full dataset (recommended). Capping to a
    subset risks test questions referencing documents that aren't indexed,
    which silently kills retrieval recall.
    """
    print("\n" + "="*60)
    print("BUILDING RAG INDEXES (v4 — table-aware chunking)")
    print("="*60)

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)

    for dataset_name, dataset_path in datasets.items():
        print(f"\nDataset: {dataset_name}")

        retriever = RAGRetriever(embedding_model_name=RAG_EMBEDDING_MODEL,
                                 collection_name=f"rag_{dataset_name}_v4")
        retriever.vector_store.clear()

        dataset = load_from_disk(dataset_path)
        all_data = concatenate_datasets(list(dataset.values()))
        if max_examples is not None:
            examples = all_data.select(range(min(max_examples, len(all_data))))
        else:
            examples = all_data

        # Deduplicate source documents first, then chunk each unique doc.
        # Keeps question metadata on every chunk for the keyword-fallback retriever.
        seen_docs = {}
        for i, ex in enumerate(examples):
            doc = _extract_full_doc(ex)
            if not doc or len(doc.strip()) < 50:
                continue
            if doc not in seen_docs:
                seen_docs[doc] = {
                    'example_id': ex.get('id', f'ex_{i}'),
                    'company': ex.get('company_name', 'Unknown'),
                    'year': str(ex.get('report_year', 'Unknown')),
                    'question': ex.get('question', '')[:100]
                }

        texts, meta_list = [], []
        seen_chunks = set()
        for doc, meta in seen_docs.items():
            chunks = chunker.chunk_table_aware(doc, metadata=meta)
            for chunk in chunks:
                text = chunk['text'].strip()
                if len(text) < 30 or text in seen_chunks:
                    continue
                seen_chunks.add(text)
                texts.append(text)
                meta_list.append(chunk['metadata'])

        print(f"  {len(seen_docs)} unique docs → {len(texts)} chunks")

        for i in tqdm(range(0, len(texts), 100), desc="Batches"):
            retriever.index_documents(texts[i:i+100], meta_list[i:i+100], batch_size=32)

        retriever.save(f"rag_{dataset_name}_v4")
        # Write model marker so the benchmark runner can verify alignment
        marker = Path(f"data/vector_db/rag_{dataset_name}_v4/_model.json")
        marker.write_text(json.dumps({"embedding_model": RAG_EMBEDDING_MODEL}))
        stats = retriever.get_stats()
        print(f"  Saved: rag_{dataset_name}_v4 ({stats['document_count']} chunks) [{RAG_EMBEDDING_MODEL}]")


def build_graphrag(datasets, max_examples=99999):
    print("\n" + "="*60)
    print("BUILDING GRAPHRAG")
    print("="*60)

    neo4j = Neo4jClient()

    print("\nClearing existing graph...")
    neo4j.clear_database()

    builder = GraphBuilder(neo4j, llm_model="llama3.1:8b")

    for dataset_name, dataset_path in datasets.items():
        print(f"\nIndexing {dataset_name}...")
        builder.build_from_dataset(
            dataset_path,
            dataset_name=dataset_name,
            max_examples=max_examples,
            use_llm=False
        )

    stats = neo4j.get_graph_stats()
    print("\n" + "="*60)
    print("FINAL GRAPH STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    neo4j.close()


def main():
    datasets = {
        "finqa": "data/benchmarks/t2-ragbench-FinQA",
        "convfinqa": "data/benchmarks/t2-ragbench-ConvFinQA",
        "tatqa": "data/benchmarks/t2-ragbench-TAT-DQA",
    }

    print("="*60)
    print("INDEX BUILDER")
    print("="*60)
############################################################################################################################   
######################################################Build  Selection######################################################
############################################################################################################################
    build_rag_indexes(datasets)  
    build_graphrag(datasets, max_examples=999999)  
############################################################################################################################
############################################################################################################################
############################################################################################################################
    print("\n" + "="*60)
    print("INDEX BUILDING COMPLETE")
    print("="*60)
    print("\nNext: python scripts/run_parallel_benchmarks.py")


if __name__ == "__main__":
    main()