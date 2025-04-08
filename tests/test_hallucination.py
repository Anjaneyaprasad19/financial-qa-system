"""
Tests for hallucination detection functionality.
"""
import pytest
from unittest.mock import MagicMock
from src.hallucination.confidence import ConfidenceScorer
from src.hallucination.detector import HallucinationDetector

class TestConfidenceScorer:
    """Test cases for ConfidenceScorer."""
    
    def test_calculate_token_confidence(self):
        """Test token confidence calculation."""
        scorer = ConfidenceScorer(threshold=70.0)
        
        # Test with high probabilities
        high_probs = [0.9, 0.85, 0.95]
        high_confidence = scorer.calculate_token_confidence(high_probs)
        assert high_confidence == pytest.approx(90.0, abs=0.1)
        
        # Test with low probabilities
        low_probs = [0.3, 0.4, 0.5]
        low_confidence = scorer.calculate_token_confidence(low_probs)
        assert low_confidence == pytest.approx(40.0, abs=0.1)
        
        # Test with empty list
        empty_confidence = scorer.calculate_token_confidence([])
        assert empty_confidence == 0.0
    
    def test_extract_numeric_score(self):
        """Test numeric score extraction."""
        scorer = ConfidenceScorer()
        
        # Test various formats
        assert scorer.extract_numeric_score("Confidence Score: 85") == 85.0
        assert scorer.extract_numeric_score("After analysis, I give this a Score: 72.5") == 72.5
        assert scorer.extract_numeric_score("The answer gets 90/100") == 90.0
        assert scorer.extract_numeric_score("This scores 65 out of 100") == 65.0
        
        # Test default when no match
        assert scorer.extract_numeric_score("No score here") == 50.0
    
    def test_evaluate_hallucination(self):
        """Test hallucination evaluation."""
        scorer = ConfidenceScorer(threshold=70.0)
        
        # Test case: high confidence, not a hallucination
        high_result = scorer.evaluate_hallucination(
            answer="Test answer",
            llm_confidence=85.0,
            retrieval_scores=[0.9, 0.85]
        )
        assert high_result["is_hallucination"] is False
        assert high_result["confidence_score"] > 70.0
        
        # Test case: low confidence, is a hallucination
        low_result = scorer.evaluate_hallucination(
            answer="Test answer",
            llm_confidence=45.0,
            retrieval_scores=[0.3, 0.4]
        )
        assert low_result["is_hallucination"] is True
        assert low_result["confidence_score"] < 70.0

class TestHallucinationDetector:
    """Test cases for HallucinationDetector."""
    
    def test_verify_factual_consistency(self):
        """Test factual consistency verification."""
        # Mock components
        llm_manager = MagicMock()
        confidence_scorer = MagicMock()
        
        # Mock LLM behavior
        llm_manager.chat_model.predict.return_value = (
            "After analyzing the answer against the context, I found that "
            "several claims are supported by the context. There are no contradictions. "
            "Confidence Score: 85"
        )
        
        # Mock confidence scorer
        confidence_scorer.extract_numeric_score.return_value = 85.0
        confidence_scorer.threshold = 70.0
        
        # Create detector
        detector = HallucinationDetector(
            llm_manager=llm_manager,
            confidence_scorer=confidence_scorer
        )
        
        # Verify consistency
        result = detector.verify_factual_consistency(
            answer="Test answer",
            context="Test context"
        )
        
        # Assertions
        assert result["confidence_score"] == 85.0
        assert result["is_hallucination"] is False
        assert result["contains_support"] is True
        assert result["contains_contradiction"] is False
