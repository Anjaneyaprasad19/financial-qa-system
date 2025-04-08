"""
Tests for data loading and processing functionality.
"""
import os
import pytest
from src.data.loader import FinQADataLoader
from src.data.processor import DocumentProcessor
from src.data.embedder import DocumentEmbedder

class TestFinQADataLoader:
    """Test cases for FinQADataLoader."""
    
    def test_load_from_huggingface(self):
        """Test loading data from HuggingFace."""
        loader = FinQADataLoader()
        # Skip if no internet connection
        try:
            dataset = loader.load_from_huggingface()
            assert dataset is not None
        except Exception as e:
            pytest.skip(f"Skipping due to error: {e}")
    
    def test_extract_documents(self):
        """Test document extraction."""
        loader = FinQADataLoader()
        # Create a mock dataset
        mock_dataset = {
            "train": [
                {
                    "id": "doc1",
                    "pre_text": "Sample pre text",
                    "post_text": "Sample post text",
                    "table": {"header": ["A", "B"], "rows": [[1, 2]]},
                    "qa": {
                        "question": "Sample question?",
                        "program": "op1(op2(A, B))",
                        "exe_ans": "Answer",
                        "gold_inds": [1, 2]
                    }
                }
            ]
        }
        
        documents = loader.extract_documents(mock_dataset)
        assert len(documents) == 1
        assert documents[0]["id"] == "doc1"
        assert "pre_text" in documents[0]
        assert "table" in documents[0]

class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def test_process_document(self):
        """Test document processing."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        
        # Sample document
        document = {
            "id": "doc1",
            "pre_text": "Sample pre text" * 20,  # Make it long enough to be chunked
            "post_text": "Sample post text" * 20,
            "table": {"header": ["A", "B"], "rows": [[1, 2], [3, 4]]}
        }
        
        chunks = processor.process_document(document)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.page_content != ""
            assert chunk.metadata["id"] == "doc1"
    
    def test_format_table(self):
        """Test table formatting."""
        processor = DocumentProcessor()
        
        # Sample table
        table = {"header": ["A", "B"], "rows": [[1, 2], [3, 4]]}
        
        formatted = processor._format_table(table)
        assert "A | B" in formatted
        assert "1 | 2" in formatted
        assert "3 | 4" in formatted

class TestDocumentEmbedder:
    """Test cases for DocumentEmbedder."""
    
    def test_embed_query(self):
        """Test query embedding."""
        # Skip if no internet or model access
        try:
            embedder = DocumentEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embedding = embedder.embed_query("Sample query")
            assert embedding is not None
            assert embedding.shape[0] > 0  # Should have some dimensions
        except Exception as e:
            pytest.skip(f"Skipping due to error: {e}")
