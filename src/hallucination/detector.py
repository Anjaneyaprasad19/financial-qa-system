"""
Hallucination detection for financial question-answering system.
"""
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple

from src.generation.llm import LLMManager
from src.hallucination.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """
    Class to detect hallucinations in generated answers.
    """
    def __init__(
        self,
        llm_manager: LLMManager,
        confidence_scorer: ConfidenceScorer,
        verification_rounds: int = 1
    ):
        """
        Initialize the hallucination detector.
        
        Args:
            llm_manager: LLM manager for verification.
            confidence_scorer: Confidence scorer.
            verification_rounds: Number of verification rounds.
        """
        self.llm_manager = llm_manager
        self.confidence_scorer = confidence_scorer
        self.verification_rounds = verification_rounds
    
    def verify_factual_consistency(
        self,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Verify factual consistency of an answer against context.
        
        Args:
            answer: Generated answer.
            context: Context text.
            
        Returns:
            Dict with verification results.
        """
        verification_prompt = f"""
        You are a financial fact-checker. Your task is to verify whether the following answer
        is factually consistent with the provided context.
        
        Context:
        {context}
        
        Answer to verify:
        {answer}
        
        Check each factual claim in the answer and determine if it is:
        1. Supported by the context (provide the specific part of the context that supports it)
        2. Contradicted by the context (provide the contradiction)
        3. Not mentioned in the context
        
        First list each factual claim from the answer, then verify each one in detail.
        Finally, give an overall assessment and a factual consistency score from 0 to 100.
        """
        
        try:
            # Get verification from LLM
            verification_result = self.llm_manager.chat_model.predict(verification_prompt)
            
            # Extract the confidence score
            confidence_score = self.confidence_scorer.extract_numeric_score(verification_result)
            
            # Simple checks for hallucination signals in the verification
            contains_contradiction = re.search(r'contradict|incorrect|false|wrong|no support', 
                                              verification_result, re.IGNORECASE) is not None
            
            contains_support = re.search(r'supported|correct|accurate|true|consistent', 
                                        verification_result, re.IGNORECASE) is not None
            
            # Determine if hallucination is detected
            is_hallucination = (
                confidence_score < self.confidence_scorer.threshold or
                (contains_contradiction and not contains_support)
            )
            
            return {
                "verification_result": verification_result,
                "confidence_score": confidence_score,
                "is_hallucination": is_hallucination,
                "contains_contradiction": contains_contradiction,
                "contains_support": contains_support,
            }
            
        except Exception as e:
            logger.error(f"Error in verification: {e}")
            return {
                "verification_result": f"Error in verification: {str(e)}",
                "confidence_score": 0.0,
                "is_hallucination": True,
                "error": str(e),
            }
    
    def detect(
        self,
        answer: str,
        query: str,
        context: str,
        retrieval_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in a generated answer.
        
        Args:
            answer: Generated answer.
            query: Original query.
            context: Context text.
            retrieval_scores: Document retrieval scores.
            
        Returns:
            Dict with hallucination detection results.
        """
        # Perform factual consistency verification
        verification = self.verify_factual_consistency(answer, context)
        
        # Get confidence score from verification
        llm_confidence = verification["confidence_score"]
        
        # Evaluate hallucination
        hallucination_assessment = self.confidence_scorer.evaluate_hallucination(
            answer=answer,
            llm_confidence=llm_confidence,
            retrieval_scores=retrieval_scores
        )
        
        # Combine results
        result = {
            "answer": answer,
            "query": query,
            "confidence_score": hallucination_assessment["confidence_score"],
            "is_hallucination": hallucination_assessment["is_hallucination"],
            "verification": verification["verification_result"],
            "confidence_components": hallucination_assessment["components"],
        }
        
        # Add warning if hallucination detected
        if result["is_hallucination"]:
            result["warning"] = (
                "The generated answer may contain hallucinated information "
                "not supported by the source documents."
            )
            
        return result
