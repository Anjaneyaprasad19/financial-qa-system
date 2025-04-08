"""
Document embedder module for financial question-answering system.
"""
import os
import logging
from typing import List, Optional, Dict, Any, Union

import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Class to generate embeddings for documents.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        embeddings_dir: Optional[str] = None
    ):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the embedding model to use.
            device: Device to run the model on ('cpu' or 'cuda').
            embeddings_dir: Directory to save embeddings.
        """
        self.model_name = model_name
        self.device = device
        self.embeddings_dir = embeddings_dir
        
        logger.info(f"Initializing embedding model: {model_name} on {device}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
        
        if embeddings_dir and not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)
    
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed.
            
        Returns:
            NumPy array of document embeddings.
        """
        texts = [doc.page_content for doc in documents]
        logger.info(f"Generating embeddings for {len(texts)} documents")
        
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text to embed.
            
        Returns:
            NumPy array of query embedding.
        """
        try:
            embedding = self.embedding_model.embed_query(query)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
