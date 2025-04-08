"""
Answer generator for financial question-answering system.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from langchain.docstore.document import Document

from src.retrieval.retriever import DocumentRetriever
from src.generation.llm import LLMManager

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Class to generate answers to financial questions.
    """
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_manager: LLMManager,
        max_context_length: int = 4000
    ):
        """
        Initialize the answer generator.
        
        Args:
            retriever: Document retriever component.
            llm_manager: LLM manager for generation.
            max_context_length: Maximum length of context to include.
        """
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.max_context_length = max_context_length
    
    def _prepare_context(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[str, List[Tuple[Document, float]]]:
        """
        Prepare context for generation by retrieving relevant documents.
        
        Args:
            query: Query text.
            top_k: Number of documents to retrieve.
            
        Returns:
            Tuple of (formatted context string, list of retrieved documents)
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        # Format context
        context = self.retriever.format_retrieved_contexts(retrieved_docs)
        
        # Trim context if too long
        if len(context) > self.max_context_length:
            logger.warning(f"Context too long ({len(context)} chars), trimming to {self.max_context_length}")
            context = context[:self.max_context_length] + "..."
            
        return context, retrieved_docs
    
    def generate(
        self,
        query: str,
        top_k: int = 5,
        check_confidence: bool = True,
        return_full_result: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate an answer to a query.
        
        Args:
            query: Query text.
            top_k: Number of documents to retrieve.
            check_confidence: Whether to check confidence score.
            return_full_result: Whether to return full result with metadata.
            
        Returns:
            Generated answer or dict with answer and metadata.
        """
        # Prepare context
        context, retrieved_docs = self._prepare_context(query, top_k=top_k)
        
        # Generate answer
        answer_data = self.llm_manager.generate_answer(
            question=query,
            context=context,
            return_metadata=True
        )
        
        answer = answer_data["answer"]
        
        # Check confidence if requested
        confidence_data = None
        if check_confidence:
            confidence_data = self.llm_manager.check_confidence(
                context=context,
                answer=answer,
                return_metadata=True
            )
            
            logger.info(f"Confidence score: {confidence_data['score']}")
        
        if return_full_result:
            result = {
                "query": query,
                "answer": answer,
                "retrieved_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in retrieved_docs
                ],
                "answer_metadata": answer_data.get("metadata"),
            }
            
            if confidence_data:
                result["confidence_score"] = confidence_data["score"]
                result["confidence_explanation"] = confidence_data["explanation"]
                
            return result
        else:
            if confidence_data:
                return f"{answer}\n\nConfidence: {confidence_data['score']:.1f}/100"
            else:
                return answer
