import sys
sys.path.append('.')

from src.rag.retriever import RAGRetriever
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.graph_builder import GraphBuilder
from src.utils.chunking import DocumentChunker
from datasets import load_from_disk
from tqdm import tqdm


def build_improved_rag_indexes(datasets, max_examples=2000):
    """Build comprehensive RAG indexes with better chunking"""
    print("\n" + "="*60)
    print("BUILDING IMPROVED RAG INDEXES")
    print("="*60)
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\nDataset: {dataset_name}")
        
        retriever = RAGRetriever(collection_name=f"rag_{dataset_name}_v2")
        retriever.vector_store.clear()
        
        print("Loading dataset...")
        dataset = load_from_disk(dataset_path)
        split = list(dataset.keys())[0]
        examples = dataset[split].select(range(min(max_examples, len(dataset[split]))))
        
        print(f"Chunking {len(examples)} documents...")
        chunker = DocumentChunker(
            chunk_size=300,
            chunk_overlap=75
        )
        
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
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
            batch_texts = texts[i:i+batch_size]
            batch_metas = metas[i:i+batch_size]
            retriever.index_documents(batch_texts, batch_metas, batch_size=32)
        
        retriever.save(f"rag_{dataset_name}_v2")
        print(f"Saved improved index: rag_{dataset_name}_v2")
        
        stats = retriever.get_stats()
        print(f"  Total chunks: {stats['document_count']}")


def build_improved_graphrag(datasets, max_examples=2000):
    """Build comprehensive GraphRAG with more documents and improved extraction"""
    print("\n" + "="*60)
    print("BUILDING IMPROVED GRAPHRAG")
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


def test_retrieval_quality(datasets):
    """Test the quality of improved retrievals"""
    print("\n" + "="*60)
    print("TESTING RETRIEVAL QUALITY")
    print("="*60)
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\nDataset: {dataset_name}")
        
        dataset = load_from_disk(dataset_path)
        split = list(dataset.keys())[0]
        test_question = dataset[split][0]
        
        retriever = RAGRetriever(collection_name=f"rag_{dataset_name}_v2")
        retriever.load(f"rag_{dataset_name}_v2")
        
        context = retriever.retrieve_context_string(test_question['question'], top_k=5)
        
        print(f"  Question: {test_question['question'][:100]}...")
        print(f"  Ground Truth: {test_question.get('program_answer') or test_question.get('original_answer')}")
        print(f"  Retrieved chunks: 5")
        print(f"  Context length: {len(context)} chars")
        print(f"  First 200 chars: {context[:200]}...")


def main():
    datasets = {
        "finqa": "data/benchmarks/t2-ragbench-FinQA",
        "convfinqa": "data/benchmarks/t2-ragbench-ConvFinQA",
        "tatqa": "data/benchmarks/t2-ragbench-TAT-DQA",
    }
    
    print("="*60)
    print("IMPROVED INDEX BUILDER - WITH ENHANCED EXTRACTION")
    print("="*60)
    print("\nThis will:")
    print("  1. Build improved RAG indexes (2000 examples, better chunking)")
    print("  2. Build improved GraphRAG (2000 examples, markdown table extraction)")
    print("  3. Test retrieval quality")
    print("\nExpected improvements:")
    print("  - RAG: Same quality, 4x more data")
    print("  - GraphRAG: 10-20x more metrics extracted (markdown table parsing)")
    print("\nEstimated time: 15-20 minutes")
    
    build_improved_rag_indexes(datasets, max_examples=2000)
    
    build_improved_graphrag(datasets, max_examples=2000)
    
    test_retrieval_quality(datasets)
    
    print("\n" + "="*60)
    print("INDEX BUILDING COMPLETE!")
    print("="*60)
    print("\nImproved indexes saved as:")
    print("  - rag_finqa_v2")
    print("  - rag_convfinqa_v2")
    print("  - rag_tatqa_v2")
    print("  - GraphRAG with enhanced extraction")
    print("\nNext steps:")
    print("  1. Run: python scripts/run_parallel_benchmarks.py")
    print("  2. Compare: python scripts/compare_all_runs.py")


if __name__ == "__main__":
    main()