"""Benchmark functionality for the QA system."""
import time
import logging
from typing import List, Dict, Any, Optional

from src.data.loader import FinQADataLoader
from src.generation.generator import AnswerGenerator
from src.hallucination.detector import HallucinationDetector
from src.evaluation.metrics import calculate_metrics, calculate_hallucination_metrics

logger = logging.getLogger(__name__)

class QABenchmark:
    """Class to benchmark the financial QA system."""
    
    def __init__(
        self,
        data_loader: FinQADataLoader,
        generator: AnswerGenerator,
        detector: HallucinationDetector,
        sample_size: Optional[int] = None
    ):
        """
        Initialize the benchmark.
        
        Args:
            data_loader: Data loader for test data.
            generator: Answer generator component.
            detector: Hallucination detector component.
            sample_size: Number of test examples to use (None for all).
        """
        self.data_loader = data_loader
        self.generator = generator
        self.detector = detector
        self.sample_size = sample_size
    
    def run(self) -> Dict[str, Any]:
        """
        Run the benchmark evaluation.
        
        Returns:
            Dictionary with benchmark results.
        """
        # Load test data
        dataset = self.data_loader.load()
        qa_pairs = self.data_loader.extract_qa_pairs(dataset)
        
        # Limit to sample size if specified
        if self.sample_size and self.sample_size < len(qa_pairs):
            qa_pairs = qa_pairs[:self.sample_size]
        
        logger.info(f"Running benchmark on {len(qa_pairs)} test examples")
        
        predictions = []
        references = []
        confidence_scores = []
        hallucination_predictions = []
        hallucination_actuals = []  # Would need manual annotation
        
        total_time = 0.0
        
        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            reference = qa["answer"]
            
            # Time the generation and detection
            start_time = time.time()
            
            # Generate answer
            result = self.generator.generate(
                query=question,
                check_confidence=True,
                return_full_result=True
            )
            
            # Get context from retrieved documents
            context = "\n".join([doc["content"] for doc in result["retrieved_documents"]])
            
            # Get retrieval scores
            retrieval_scores = [doc["score"] for doc in result["retrieved_documents"]]
            
            # Detect hallucinations
            detection_result = self.detector.detect(
                answer=result["answer"],
                query=question,
                context=context,
                retrieval_scores=retrieval_scores
            )
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # Store results
            predictions.append(result["answer"])
            references.append(reference)
            confidence_scores.append(detection_result["confidence_score"])
            hallucination_predictions.append(detection_result["is_hallucination"])
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(qa_pairs)} examples")
        
        # Calculate metrics
        qa_metrics = calculate_metrics(
            predictions=predictions,
            references=references,
            confidence_scores=confidence_scores
        )
        
        # For hallucination detection metrics, we would need human annotation
        # This is a placeholder assuming we had actual hallucination labels
        # hallucination_actuals = [False] * len(hallucination_predictions)
        # hallucination_metrics = calculate_hallucination_metrics(
        #     predicted_hallucinations=hallucination_predictions,
        #     actual_hallucinations=hallucination_actuals
        # )
        
        # Combine results
        results = {
            "num_examples": len(qa_pairs),
            "total_time": total_time,
            "avg_time_per_example": total_time / len(qa_pairs),
            "metrics": qa_metrics,
            # "hallucination_metrics": hallucination_metrics,
        }
        
        return results
