import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from typing import List, Dict


class RAGRetriever:
    """RAG retriever for document-based question answering"""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "financial_docs"
    ):
        """
        Initialize RAG retriever
        
        Args:
            embedding_model_name: Name of sentence-transformers model
            collection_name: Name for vector store collection
        """
        print("Initializing RAG Retriever...")
        
        # Initialize embeddings
        self.embedder = EmbeddingModel(embedding_model_name)
        
        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_dim=self.embedder.embedding_dim,
            persist_dir=f"data/vector_db/{collection_name}"
        )
        
        print("RAG Retriever ready!")
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: List[Dict] = None,
        batch_size: int = 32
    ):
        """
        Index documents for retrieval
        
        Args:
            documents: List of document texts to index
            metadatas: Optional metadata for each document
            batch_size: Batch size for embedding
        """
        print(f"\nIndexing {len(documents)} documents...")
        
        # Embed documents
        embeddings = self.embedder.embed_batch(documents, batch_size=batch_size)
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"Indexed {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with retrieved documents and metadata
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
    
    def retrieve_context_string(
        self,
        query: str,
        top_k: int = 3,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Retrieve and format context as a single string
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            separator: Separator between documents
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results['documents']:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(results['documents']):
            context_parts.append(f"[Document {i+1}]\n{doc}")
        
        return separator.join(context_parts)
    
    def save(self, name: str = "rag_index"):
        """Save vector store to disk"""
        self.vector_store.save(name)
    
    def load(self, name: str = "rag_index"):
        """Load vector store from disk"""
        return self.vector_store.load(name)
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return {
            'embedding_model': self.embedder.model_name,
            'embedding_dim': self.embedder.embedding_dim,
            **self.vector_store.get_stats()
        }


def main():
    """Test RAG retriever"""
    print("="*60)
    print("Testing RAG Retriever")
    print("="*60)
    
    # Initialize retriever
    retriever = RAGRetriever(collection_name="test_rag")
    
    # Clear any existing data
    retriever.vector_store.clear()
    
    # Sample financial-style documents
    documents = [
        "In fiscal year 2009, Analog Devices reported interest expense of $3.8 million. This represented a decrease from the previous year.",
        "The company's total revenue for 2009 was $2.5 billion, with gross margins of 60%.",
        "Abiomed, Inc. recorded $3.3 million in stock-based compensation expense during fiscal 2012.",
        "Intel's available-for-sale investments comprised 53.2% of total cash and investments as of December 29, 2012.",
        "Paris is the capital city of France with a population of approximately 2.2 million people."
    ]
    
    metadatas = [
        {"company": "Analog Devices", "year": 2009, "topic": "interest_expense"},
        {"company": "Analog Devices", "year": 2009, "topic": "revenue"},
        {"company": "Abiomed", "year": 2012, "topic": "compensation"},
        {"company": "Intel", "year": 2012, "topic": "investments"},
        {"company": "None", "year": 0, "topic": "geography"}
    ]
    
    # Index documents
    retriever.index_documents(documents, metadatas)
    
    # Test retrieval
    print("\n" + "="*60)
    print("Testing Retrieval")
    print("="*60)
    
    query = "What was Analog Devices' interest expense in 2009?"
    
    print(f"\nQuery: {query}")
    
    # Get top 3 results
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nTop 3 Retrieved Documents:")
    for i, (doc, dist, meta) in enumerate(zip(
        results['documents'],
        results['distances'],
        results['metadatas']
    )):
        print(f"\n{i+1}. Similarity: {dist:.4f}")
        print(f"   Metadata: {meta}")
        print(f"   Text: {doc[:100]}...")
    
    # Test context string formatting
    print("\n" + "="*60)
    print("Testing Context String")
    print("="*60)
    
    context = retriever.retrieve_context_string(query, top_k=2)
    print(f"\n📄 Formatted Context:\n{context}")
    
    # Stats
    print("\n" + "="*60)
    print(f"📈 Stats: {retriever.get_stats()}")
    print("="*60)


if __name__ == "__main__":
    main()