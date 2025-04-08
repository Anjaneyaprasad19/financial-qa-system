"""
Tests for retrieval functionality.
"""
import os
import numpy as np
import pytest
from langchain.docstore.document import Document
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import DocumentRetriever
from src.data.embedder import DocumentEmbedder

class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore."""
    
    def test_add_and_search(self):
        """Test adding documents and searching."""
        # Create a test vector store
        vector_store = FAISSVectorStore(embedding_dimension=4)
        
        # Create test documents and embeddings
        docs = [
            Document(page_content="Test document 1", metadata={"id": 1}),
            Document(page_content="Test document 2", metadata={"id": 2})
        ]
        
        # Simple mock embeddings (4-dimensional)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Add documents
        vector_store.add_documents(docs, embeddings)
        
        # Search using the first embedding
        results = vector_store.search(embeddings[0], k=1)
        
        assert len(results) == 1
        assert results[0][0].page_content == "Test document 1"
    
    @pytest.mark.skipif(not os.path.exists("test_temp"), reason="Temp directory not available")
    def test_save_load(self):
        """Test saving and loading vector store."""
        # Create temp directory if it doesn't exist
        os.makedirs("test_temp", exist_ok=True)
        
        # Create a test vector store
        vector_store = FAISSVectorStore(
            embedding_dimension=4,
            store_dir="test_temp"
        )
        
        # Create test documents and embeddings
        docs = [
            Document(page_content="Test document 1", metadata={"id": 1}),
            Document(page_content="Test document 2", metadata={"id": 2})
        ]
        
        # Simple mock embeddings (4-dimensional)
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Add documents
        vector_store.add_documents(docs, embeddings)
        
        # Save
        vector_store.save("test_index")
        
        # Load in a new instance
        new_store = FAISSVectorStore(
            embedding_dimension=4,
            store_dir="test_temp"
        )
        new_store.load("test_index")
        
        # Search using the first embedding
        results = new_store.search(embeddings[0], k=1)
        
        assert len(results) == 1
        assert results[0][0].page_content == "Test document 1"

class TestDocumentRetriever:
    """Test cases for DocumentRetriever."""
    
    def test_format_retrieved_contexts(self):
        """Test formatting of retrieved documents."""
        # Mock components
        embedder = None  # Not needed for this test
        vector_store = None  # Not needed for this test
        
        retriever = DocumentRetriever(embedder, vector_store)
        
        # Create mock retrieved docs
        retrieved_docs = [
            (Document(page_content="Test document 1", metadata={"id": 1}), 0.9),
            (Document(page_content="Test document 2", metadata={"id": 2}), 0.7)
        ]
        
        # Format with scores
        context_with_scores = retriever.format_retrieved_contexts(retrieved_docs, include_scores=True)
        assert "Document 1" in context_with_scores
        assert "Relevance: 0.9" in context_with_scores
        assert "Test document 1" in context_with_scores
        
        # Format without scores
        context_without_scores = retriever.format_retrieved_contexts(retrieved_docs, include_scores=False)
        assert "Document 1" in context_without_scores
        assert "Relevance: 0.9" not in context_without_scores
        assert "Test document 1" in context_without_scores
