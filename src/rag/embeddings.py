from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    """Wrapper for sentence transformer embedding models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of sentence-transformers model
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Embedding model loaded: {model_name}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            Array of embedding vectors
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


def main():
    """Test embedding model"""
    print("="*60)
    print("Testing Embedding Model")
    print("="*60)
    
    embedder = EmbeddingModel()
    
    # Test single embedding
    text = "What is the capital of France?"
    embedding = embedder.embed_text(text)
    
    print(f"\nEmbedded text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dim: {embedder.embedding_dim}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test batch
    texts = [
        "Paris is the capital of France.",
        "London is the capital of England.",
        "What is 2 + 2?"
    ]
    
    embeddings = embedder.embed_batch(texts)
    print(f"\nEmbedded {len(texts)} texts")
    print(f"   Batch shape: {embeddings.shape}")


if __name__ == "__main__":
    main()