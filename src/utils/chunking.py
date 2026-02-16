# Document chunking utilities


from typing import List, Dict
import re


class DocumentChunker:
    """Chunk documents into smaller, retrievable pieces"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separator: str = "\n"
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            separator: Separator to split on (prefer sentence boundaries)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk a single text into smaller pieces
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dicts with 'text' and 'metadata'
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Split into sentences first (better boundaries)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': metadata or {}
                })
                
                # Start new chunk with overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else chunk_text
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'metadata': metadata or {}
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved)
        # Split on periods, question marks, exclamation points followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_dataset_contexts(
        self,
        dataset_examples: List[Dict],
        context_field: str = 'context'
    ) -> List[Dict]:
        """
        Chunk contexts from a dataset
        
        Args:
            dataset_examples: List of dataset examples
            context_field: Field name containing context text
            
        Returns:
            List of chunked documents with metadata
        """
        all_chunks = []
        
        for i, example in enumerate(dataset_examples):
            context = example.get(context_field, '')
            
            if not context or len(context) < 50:
                continue
            
            # Build metadata
            metadata = {
                'example_id': example.get('id', f'ex_{i}'),
                'context_id': example.get('context_id', f'ctx_{i}'),
                'company': example.get('company_name', 'Unknown'),
                'year': str(example.get('report_year', 'Unknown')),
                'source': 'dataset',
                'question': example.get('question', '')[:100]  # Store related question
            }
            
            # Chunk the context
            chunks = self.chunk_text(context, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks


def main():
    """Test document chunker"""
    print("="*60)
    print("Testing Document Chunker")
    print("="*60)
    
    # Sample financial text
    text = """
    In fiscal year 2009, Analog Devices reported interest expense of $3.8 million. 
    This represented a decrease from the previous year. The company's total revenue 
    for 2009 was $2.5 billion, with gross margins of 60%. Net income for the year 
    was $450 million. Operating expenses included research and development costs 
    of $550 million and sales and marketing expenses of $420 million. The company 
    employed approximately 8,500 people worldwide. Cash and cash equivalents at 
    year end totaled $1.2 billion.
    """
    
    # Initialize chunker
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    
    # Chunk the text
    chunks = chunker.chunk_text(
        text,
        metadata={'company': 'Analog Devices', 'year': 2009}
    )
    
    print(f"\n📄 Original text length: {len(text)} characters")
    print(f"📦 Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n{'='*60}")
        print(f"Chunk {i+1}:")
        print(f"{'='*60}")
        print(f"Length: {len(chunk['text'])} characters")
        print(f"Metadata: {chunk['metadata']}")
        print(f"\nText:\n{chunk['text']}")


if __name__ == "__main__":
    main()