"""
Vector store implementation for financial question-answering system.
"""
import os
import logging
import pickle
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import faiss
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS


logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Class to manage FAISS vector store for document retrieval.
    """
    def __init__(
        self,
        embedding_dimension: int = 384,  # Default for MiniLM-L6-v2
        index_type: str = "IndexFlatL2",
        store_dir: Optional[str] = None
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            embedding_dimension: Dimension of embeddings.
            index_type: Type of FAISS index to use.
            store_dir: Directory to save vector store.
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.store_dir = store_dir
        self.index = None
        self.document_store = {}  # Maps index ID to document
        
        if store_dir and not os.path.exists(store_dir):
            os.makedirs(store_dir)
            
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index based on the specified type."""
        if self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(self.embedding_dimension)
            nlist = 100  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
            # Need to train this type of index with some data
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
        logger.info(f"Initialized FAISS index of type {self.index_type}")
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents to add.
            embeddings: NumPy array of document embeddings.
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})"
            )
            
        # Store documents
        start_id = len(self.document_store)
        for i, doc in enumerate(documents):
            self.document_store[start_id + i] = doc
            
        # Add vectors to the index
        if self.index_type == "IndexIVFFlat" and not self.index.is_trained:
            if embeddings.shape[0] > 1:
                logger.info(f"Training IVF index with {embeddings.shape[0]} vectors")
                self.index.train(embeddings)
            else:
                logger.warning("Not enough vectors to train IVF index")
                
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Embedding of the query.
            k: Number of results to return.
            
        Returns:
            List of (document, score) tuples.
        """
        if not self.index:
            raise ValueError("Index has not been initialized")
            
        # Ensure the query embedding is properly shaped
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Perform the search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.document_store:  # -1 means no result
                # Convert distance to similarity score (1 - normalized_distance)
                # This assumes we're using L2 distance on normalized vectors
                similarity = 1.0 - (distances[0][i] / 2.0)  # scale to [0,1]
                results.append((self.document_store[idx], similarity))
                
        return results
    
    def save(self, filename: str) -> str:
        """
        Save the vector store to disk.
        
        Args:
            filename: Base name to save the files as.
            
        Returns:
            Path to the saved index file.
        """
        if not self.store_dir:
            raise ValueError("Store directory not specified")
            
        index_path = os.path.join(self.store_dir, f"{filename}.index")
        docstore_path = os.path.join(self.store_dir, f"{filename}.docstore")
        
        # Save the FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save the document store
        with open(docstore_path, 'wb') as f:
            pickle.dump(self.document_store, f)
            
        logger.info(f"Saved vector store to {index_path} and {docstore_path}")
        return index_path
    
    def load(self, filename: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            filename: Base name of the files to load.
        """
        if not self.store_dir:
            raise ValueError("Store directory not specified")
            
        index_path = os.path.join(self.store_dir, f"{filename}.index")
        docstore_path = os.path.join(self.store_dir, f"{filename}.docstore")
        
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the document store
        with open(docstore_path, 'rb') as f:
            self.document_store = pickle.load(f)
            
        logger.info(f"Loaded vector store from {index_path} and {docstore_path}")
