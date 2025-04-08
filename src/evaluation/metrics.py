"""Evaluation metrics for the QA system."""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics(
    predictions: List[str],
    references: List[str],
    confidence_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculate basic evaluation metrics."""
    if len(predictions) != len(references):
        raise ValueError("Number of predictions does not match number of references")
    
    # Exact match
    exact_matches = [p.strip() == r.strip() for p, r in zip(predictions, references)]
    exact_match_rate = sum(exact_matches) / len(exact_matches)
    
    # Add confidence calibration if scores are provided
    calibration_error = None
    if confidence_scores:
        if len(confidence_scores) != len(predictions):
            raise ValueError("Number of confidence scores does not match number of predictions")
        
        # Normalize confidence scores to [0, 1]
        normalized_scores = [s / 100.0 for s in confidence_scores]
        
        # Calculate calibration error (mean squared difference between confidence and correctness)
        calibration_error = np.mean([(s - c) ** 2 for s, c in zip(normalized_scores, exact_matches)])
    
    # Return metrics
    metrics = {
        "exact_match_rate": exact_match_rate,
    }
    
    if calibration_error is not None:
        metrics["calibration_error"] = calibration_error
    
    return metrics

def calculate_hallucination_metrics(
    predicted_hallucinations: List[bool],
    actual_hallucinations: List[bool]
) -> Dict[str, float]:
    """Calculate metrics for hallucination detection."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_hallucinations,
        predicted_hallucinations,
        average='binary',
        zero_division=0
    )
    
    return {
        "hallucination_precision": precision,
        "hallucination_recall": recall,
        "hallucination_f1": f1,
    }
