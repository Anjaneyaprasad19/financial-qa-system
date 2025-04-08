"""
Document processing module for financial question-answering system.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Class to process financial documents for RAG system.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        processed_dir: Optional[str] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for processing.
            chunk_overlap: Overlap between consecutive chunks.
            processed_dir: Directory to save processed documents.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_dir = processed_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        if processed_dir and not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
    
    def _format_table(self, table: Dict[str, Any]) -> str:
        """
        Format a table into a string representation.
        
        Args:
            table: The table data.
            
        Returns:
            Formatted table as string.
        """
        if not table:
            return ""
            
        # Handle different table formats
        if isinstance(table, str):
            return table
            
        # If table is a dictionary with specific format
        # This is a simplified version, adapt based on actual data structure
        try:
            formatted_table = "TABLE:\n"
            # Add headers
            if 'header' in table:
                formatted_table += " | ".join(table['header']) + "\n"
                formatted_table += "-" * (len(formatted_table) - 1) + "\n"
                
            # Add rows
            if 'rows' in table:
                for row in table['rows']:
                    formatted_table += " | ".join(str(cell) for cell in row) + "\n"
                    
            return formatted_table
        except Exception as e:
            logger.warning(f"Error formatting table: {e}. Returning raw representation.")
            return str(table)
    
    def process_document(self, document: Dict[str, Any]) -> List[Document]:
        """
        Process a financial document into chunks.
        
        Args:
            document: Financial document data.
            
        Returns:
            List of document chunks.
        """
        doc_id = document.get('id', 'unknown')
        pre_text = document.get('pre_text', '')
        post_text = document.get('post_text', '')
        table = document.get('table', {})
        
        # Format the full document
        formatted_table = self._format_table(table)
        full_text = f"{pre_text}\n\n{formatted_table}\n\n{post_text}"
        
        # Create document chunks
        chunks = self.text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{'id': doc_id, 'source': 'finqa'}]
        )
        
        return chunks
    
    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Process a batch of financial documents.
        
        Args:
            documents: List of financial documents.
            
        Returns:
            List of processed document chunks.
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.process_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def save_processed_chunks(self, chunks: List[Document], filename: str) -> str:
        """
        Save processed document chunks to disk.
        
        Args:
            chunks: List of document chunks.
            filename: Name to save the file as.
            
        Returns:
            Path to the saved file.
        """
        if not self.processed_dir:
            raise ValueError("Processed directory not specified")
            
        output_path = os.path.join(self.processed_dir, filename)
        
        # Convert langchain Documents to serializable format
        serializable_chunks = []
        for chunk in chunks:
            serializable_chunks.append({
                'page_content': chunk.page_content,
                'metadata': chunk.metadata
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(chunks)} processed chunks to {output_path}")
        return output_path
    
    def load_processed_chunks(self, filename: str) -> List[Document]:
        """
        Load processed document chunks from disk.
        
        Args:
            filename: Name of the file to load.
            
        Returns:
            List of document chunks.
        """
        if not self.processed_dir:
            raise ValueError("Processed directory not specified")
            
        input_path = os.path.join(self.processed_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            serialized_chunks = json.load(f)
            
        # Convert back to langchain Documents
        chunks = []
        for item in serialized_chunks:
            chunks.append(Document(
                page_content=item['page_content'],
                metadata=item['metadata']
            ))
            
        logger.info(f"Loaded {len(chunks)} processed chunks from {input_path}")
        return chunks
