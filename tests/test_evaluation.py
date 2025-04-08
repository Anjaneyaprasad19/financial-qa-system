"""
Tests for evaluation functionality.
"""
import pytest
from src.evaluation.metrics import calculate_metrics, calculate_hallucination_metrics

class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_calculate_metrics(self):
        """Test basic evaluation metrics calculation."""
        # Test data
        predictions = ["Answer 1", "Answer 2", "Answer 3", "Answer 4"]
        references = ["Answer 1", "Different", "Answer 3", "Different"]
        confidence_scores = [90.0, 60.0, 85.0, 40.0]
        
        # Calculate metrics
        metrics = calculate_metrics(
            predictions=predictions,
            references=references,
            confidence_scores=confidence_scores
        )
        
        # Assertions
        assert "exact_match_rate" in metrics
        assert metrics["exact_match_rate"] == 0.5  # 2 out of 4 match
        assert "calibration_error" in metrics
        
        # Test without confidence scores
        metrics_no_conf = calculate_metrics(
            predictions=predictions,
            references=references
        )
        
        assert "exact_match_rate" in metrics_no_conf
        assert "calibration_error" not in metrics_no_conf
        
        # Test error case
        with pytest.raises(ValueError):
            calculate_metrics(
                predictions=["Answer 1"],
                references=["Answer 1", "Answer 2"]
            )
    
    def test_calculate_hallucination_metrics(self):
        """Test hallucination detection metrics calculation."""
        # Test data
        predicted = [True, False, True, False, True]
        actual = [True, False, False, True, True]
        
        # Calculate metrics
        metrics = calculate_hallucination_metrics(
            predicted_hallucinations=predicted,
            actual_hallucinations=actual
        )
        
        # Assertions
        assert "hallucination_precision" in metrics
        assert "hallucination_recall" in metrics
        assert "hallucination_f1" in metrics
        
        # Calculate expected values
        # True Positives = 2 (Both predicted and actual True)
        # False Positives = 1 (Predicted True but actual False)
        # False Negatives = 1 (Predicted False but actual True)
        
        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
        # F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
        
        assert metrics["hallucination_precision"] == pytest.approx(2/3, abs=0.01)
        assert metrics["hallucination_recall"] == pytest.approx(2/3, abs=0.01)
        assert metrics["hallucination_f1"] == pytest.approx(2/3, abs=0.01)
