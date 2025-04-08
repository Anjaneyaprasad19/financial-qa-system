"""
Document retriever for financial question-answering system.
"""
import logging
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
from langchain.docstore.document import Document

from src.data.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Class to retrieve relevant documents for a given query.
    """
    def __init__(
        self,
        embedder: DocumentEmbedder,
        vector_store: FAISSVectorStore,
        top_k: int = 5
    ):
        """
        Initialize the document retriever.
        
        Args:
            embedder: Document embedder to generate query embeddings.
            vector_store: Vector store for document lookup.
            top_k: Number of documents to retrieve.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text.
            top_k: Optional override for number of documents to retrieve.
            
        Returns:
            List of (document, score) tuples.
        """
        k = top_k if top_k is not None else self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve documents
        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k
        )
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def format_retrieved_contexts(
        self,
        retrieved_docs: List[Tuple[Document, float]],
        include_scores: bool = True
    ) -> str:
        """
        Format retrieved documents into a single context string.
        
        Args:
            retrieved_docs: List of (document, score) tuples.
            include_scores: Whether to include retrieval scores.
            
        Returns:
            Formatted context string.
        """
        if not retrieved_docs:
            return "No relevant information found."
            
        context = ""
        for i, (doc, score) in enumerate(retrieved_docs):
            context += f"\n--- Document {i+1}"
            if include_scores:
                context += f" (Relevance: {score:.4f})"
            context += " ---\n"
            context += doc.page_content
            context += "\n"
            
        return context
