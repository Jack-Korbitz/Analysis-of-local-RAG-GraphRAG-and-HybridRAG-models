"""
Vector Store using FAISS
"""

import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional
from pathlib import Path


class VectorStore:
    """FAISS vector store for document retrieval"""
    
    def __init__(self, embedding_dim: int = 384, persist_dir: str = "data/vector_db"):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of embeddings (384 for MiniLM)
            persist_dir: Directory to persist the index
        """
        self.embedding_dim = embedding_dim
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index (using cosine similarity via normalized L2)
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product after normalization = cosine
        
        # Storage for documents and metadata
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        print(f"   Vector store initialized (FAISS)")
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Documents in store: {self.index.ntotal}")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings (will be normalized)
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents
        """
        if ids is None:
            start_idx = len(self.documents)
            ids = [f"doc_{start_idx + i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings_normalized.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        print(f"   Added {len(documents)} documents to vector store")
        print(f"   Total documents: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Dict with 'documents', 'distances', 'metadatas', 'ids'
        """
        if self.index.ntotal == 0:
            return {
                'documents': [],
                'distances': [],
                'metadatas': [],
                'ids': []
            }
        
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_normalized, min(top_k, self.index.ntotal))
        
        # Get results
        results = {
            'documents': [self.documents[i] for i in indices[0]],
            'distances': distances[0].tolist(),
            'metadatas': [self.metadatas[i] for i in indices[0]],
            'ids': [self.ids[i] for i in indices[0]]
        }
        
        return results
    
    def save(self, name: str = "faiss_index"):
        """Save index and metadata to disk"""
        index_path = self.persist_dir / f"{name}.index"
        metadata_path = self.persist_dir / f"{name}.pkl"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"  Saved vector store to {self.persist_dir}")
    
    def load(self, name: str = "faiss_index"):
        """Load index and metadata from disk"""
        index_path = self.persist_dir / f"{name}.index"
        metadata_path = self.persist_dir / f"{name}.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            print(f"   No saved index found at {self.persist_dir}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadatas = data['metadatas']
            self.ids = data['ids']
            self.embedding_dim = data['embedding_dim']
        
        print(f"  Loaded vector store from {self.persist_dir}")
        print(f"   Documents: {self.index.ntotal}")
        return True
    
    def clear(self):
        """Clear all documents"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        self.metadatas = []
        self.ids = []
        print(f"  Cleared vector store")
    
    def get_stats(self) -> Dict:
        """Get store statistics"""
        return {
            'document_count': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'persist_dir': str(self.persist_dir)
        }


def main():
    """Test vector store"""
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.rag.embeddings import EmbeddingModel
    
    print("="*60)
    print("Testing Vector Store (FAISS)")
    print("="*60)
    
    # Initialize
    embedder = EmbeddingModel()
    vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
    
    # Sample documents
    documents = [
        "The capital of France is Paris. It has a population of 2.2 million.",
        "London is the capital of England with over 9 million people.",
        "Python is a programming language created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France."
    ]
    
    # Embed documents
    print("\n  Embedding documents...")
    embeddings = embedder.embed_batch(documents)
    
    # Add to vector store
    print("\n  Adding to vector store...")
    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))]
    )
    
    # Test search
    print("\n  Testing search...")
    query = "What is the capital of France?"
    query_embedding = embedder.embed_text(query)
    
    results = vector_store.search(query_embedding, top_k=3)
    
    print(f"\n  Query: '{query}'")
    print(f"\n  Top 3 Results:")
    for i, (doc, distance) in enumerate(zip(results['documents'], results['distances'])):
        print(f"\n{i+1}. Similarity: {distance:.4f}")
        print(f"   {doc}")
    
    # Test save/load
    print("\n  Testing save/load...")
    vector_store.save("test_index")
    
    new_store = VectorStore(embedding_dim=embedder.embedding_dim)
    new_store.load("test_index")
    
    # Stats
    print(f"\n  Stats: {vector_store.get_stats()}")


if __name__ == "__main__":
    main()