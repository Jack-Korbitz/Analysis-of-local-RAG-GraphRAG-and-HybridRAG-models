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
    
    def _extract_full_context(self, example: Dict) -> str:
        """
        Extract full document context from an example, handling all three dataset formats:
        - FinQA / ConvFinQA: pre_text + table + post_text
        - TAT-DQA: context field only
        """
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

    def chunk_table_aware(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text while keeping markdown table blocks intact.
        Table blocks (lines starting with |) are never split mid-row —
        the entire table is kept as one chunk regardless of chunk_size.
        """
        if not text or len(text.strip()) == 0:
            return []

        chunks = []
        lines = text.split('\n')
        table_block: List[str] = []
        prose_block: List[str] = []

        def flush_prose():
            if prose_block:
                prose_text = '\n'.join(prose_block).strip()
                if len(prose_text) > 50:
                    chunks.extend(self.chunk_text(prose_text, metadata))
                prose_block.clear()

        def flush_table():
            if table_block:
                table_text = '\n'.join(table_block).strip()
                if len(table_text) > 20:
                    chunks.append({'text': table_text, 'metadata': metadata or {}})
                table_block.clear()

        for line in lines:
            if line.strip().startswith('|'):
                flush_prose()
                table_block.append(line)
            else:
                flush_table()
                prose_block.append(line)

        flush_prose()
        flush_table()
        return chunks

    def chunk_dataset_contexts(
        self,
        dataset_examples: List[Dict],
        context_field: str = 'context'
    ) -> List[Dict]:
        """
        Chunk contexts from a dataset.
        Uses table-aware chunking so markdown table rows are never split mid-row.
        Handles FinQA/ConvFinQA (pre_text+table+post_text) and TAT-DQA (context).
        """
        all_chunks = []

        for i, example in enumerate(dataset_examples):
            context = self._extract_full_context(example)

            if not context or len(context) < 50:
                continue

            metadata = {
                'example_id': example.get('id', f'ex_{i}'),
                'context_id': example.get('context_id', f'ctx_{i}'),
                'company': example.get('company_name', 'Unknown'),
                'year': str(example.get('report_year', 'Unknown')),
                'source': 'dataset',
                'question': example.get('question', '')[:100]
            }

            chunks = self.chunk_table_aware(context, metadata)
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
    
    print(f"\n  Original text length: {len(text)} characters")
    print(f"    Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n{'='*60}")
        print(f"Chunk {i+1}:")
        print(f"{'='*60}")
        print(f"Length: {len(chunk['text'])} characters")
        print(f"Metadata: {chunk['metadata']}")
        print(f"\nText:\n{chunk['text']}")


if __name__ == "__main__":
    main()