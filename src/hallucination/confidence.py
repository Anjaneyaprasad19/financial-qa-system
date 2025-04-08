"""
Confidence scoring for hallucination detection.
"""
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """
    Class to compute confidence scores for generated answers.
    """
    def __init__(
        self,
        threshold: float = 70.0,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the confidence scorer.
        
        Args:
            threshold: Threshold for acceptable confidence.
            weights: Weights for different scoring components.
        """
        self.threshold = threshold
        
        # Default weights for different components
        self.weights = weights or {
            "llm_confidence": 0.6,     # LLM's self-reported confidence
            "retrieval_score": 0.2,    # Document retrieval score
            "consistency_score": 0.2,   # Consistency across generated answers
        }
        
        # Normalize weights
        weight_sum = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= weight_sum
    
    def calculate_token_confidence(
        self,
        token_probs: List[float]
    ) -> float:
        """
        Calculate confidence based on token probabilities.
        
        Args:
            token_probs: List of token probabilities.
            
        Returns:
            Confidence score (0-100).
        """
        if not token_probs:
            return 0.0
            
        # Take the average probability
        avg_prob = np.mean(token_probs)
        
        # Scale to 0-100
        confidence = avg_prob * 100.0
        
        return confidence
    
    def extract_numeric_score(self, text: str) -> float:
        """
        Extract numeric confidence score from text.
        
        Args:
            text: Text containing confidence score.
            
        Returns:
            Extracted score (0-100) or default.
        """
        # Try various patterns to extract the score
        patterns = [
            r"Confidence[ ]?(?:Score)?:?[ ]?(\d+(?:\.\d+)?)",  # Confidence Score: 85
            r"Score:[ ]?(\d+(?:\.\d+)?)",                     # Score: 85
            r"(\d+(?:\.\d+)?)(?:\/|\s?out of\s?)100",         # 85/100 or 85 out of 100
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure score is in range 0-100
                    return max(0.0, min(100.0, score))
                except ValueError:
                    pass
        
        # Default score if no pattern matches
        logger.warning(f"Could not extract confidence score from: {text[:100]}...")
        return 50.0  # Neutral score
    
    def evaluate_hallucination(
        self,
        answer: str,
        llm_confidence: float,
        retrieval_scores: List[float],
        consistency_checks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if an answer contains hallucinations.
        
        Args:
            answer: Generated answer.
            llm_confidence: LLM's self-reported confidence.
            retrieval_scores: Document retrieval scores.
            consistency_checks: Multiple generations for consistency.
            
        Returns:
            Dict with hallucination assessment.
        """
        # Calculate confidence from retrieval scores
        avg_retrieval_score = np.mean(retrieval_scores) if retrieval_scores else 0.5
        retrieval_confidence = avg_retrieval_score * 100.0
        
        # Calculate consistency score if available
        if consistency_checks and len(consistency_checks) > 1:
            # This is a simplified approach - in practice, you would want to
            # compare the semantic similarity of multiple generations
            consistency_confidence = 80.0  # Placeholder
        else:
            consistency_confidence = 70.0  # Default without consistency checks
        
        # Calculate weighted confidence score
        weighted_confidence = (
            self.weights["llm_confidence"] * llm_confidence +
            self.weights["retrieval_score"] * retrieval_confidence +
            self.weights["consistency_score"] * consistency_confidence
        )
        
        # Determine if hallucination is detected
        is_hallucination = weighted_confidence < self.threshold
        
        # Prepare result
        result = {
            "confidence_score": weighted_confidence,
            "is_hallucination": is_hallucination,
            "components": {
                "llm_confidence": llm_confidence,
                "retrieval_confidence": retrieval_confidence,
                "consistency_confidence": consistency_confidence,
            },
            "threshold": self.threshold,
        }
        
        return result
