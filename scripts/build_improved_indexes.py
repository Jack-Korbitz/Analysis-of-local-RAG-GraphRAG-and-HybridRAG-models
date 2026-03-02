import sys
sys.path.append('.')

from src.rag.retriever import RAGRetriever
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.graph_builder import GraphBuilder
from src.utils.chunking import DocumentChunker
from datasets import load_from_disk
from tqdm import tqdm


def build_rag_indexes(datasets, max_examples=2000):
    print("\n" + "="*60)
    print("BUILDING RAG INDEXES")
    print("="*60)

    for dataset_name, dataset_path in datasets.items():
        print(f"\nDataset: {dataset_name}")

        retriever = RAGRetriever(collection_name=f"rag_{dataset_name}_v2")
        retriever.vector_store.clear()

        dataset = load_from_disk(dataset_path)
        split = list(dataset.keys())[0]
        examples = dataset[split].select(range(min(max_examples, len(dataset[split]))))

        print(f"Chunking {len(examples)} documents...")
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=75)
        chunks = chunker.chunk_dataset_contexts(list(examples))

        unique_chunks = {}
        for chunk in chunks:
            text = chunk['text']
            if text not in unique_chunks and len(text.strip()) > 50:
                unique_chunks[text] = chunk

        chunks_list = list(unique_chunks.values())
        print(f"Created {len(chunks_list)} unique chunks")

        texts = [c['text'] for c in chunks_list]
        metas = [c['metadata'] for c in chunks_list]

        print("Indexing chunks...")
        for i in tqdm(range(0, len(texts), 100), desc="Batches"):
            retriever.index_documents(texts[i:i+100], metas[i:i+100], batch_size=32)

        retriever.save(f"rag_{dataset_name}_v2")
        stats = retriever.get_stats()
        print(f"  Saved: rag_{dataset_name}_v2 ({stats['document_count']} chunks)")


def build_graphrag(datasets, max_examples=2000):
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
   

    build_rag_indexes(datasets, max_examples=2000)
    build_graphrag(datasets, max_examples=999999)  # use full dataset for graph building

    print("\n" + "="*60)
    print("INDEX BUILDING COMPLETE")
    print("="*60)
    print("\nNext: python scripts/run_parallel_benchmarks.py")


if __name__ == "__main__":
    main()
