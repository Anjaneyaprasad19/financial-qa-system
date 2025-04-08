"""
Data loading module for financial question-answering system.
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)

class FinQADataLoader:
    """
    Class to load and parse FinQA dataset.
    """
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the FinQA data loader.
        
        Args:
            data_path: Optional path to local dataset files. If not provided,
                       will attempt to download from Hugging Face.
        """
        self.data_path = data_path
        
    def load_from_huggingface(self) -> Dict[str, Any]:
        """
        Load the FinQA dataset from HuggingFace.
        
        Returns:
            Dictionary containing the dataset splits.
        """
        logger.info("Loading FinQA dataset from HuggingFace")
        try:
            dataset = load_dataset("ibm-research/finqa")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from HuggingFace: {e}")
            raise
    
    def load_from_local(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load the FinQA dataset from local files.
        
        Returns:
            Dictionary containing the dataset splits.
        """
        if not self.data_path:
            raise ValueError("Data path not provided for local loading.")
        
        logger.info(f"Loading FinQA dataset from {self.data_path}")
        
        try:
            dataset = {}
            for split in ["train", "validation", "test"]:
                split_path = os.path.join(self.data_path, f"{split}.json")
                if os.path.exists(split_path):
                    with open(split_path, 'r', encoding='utf-8') as f:
                        dataset[split] = json.load(f)
                else:
                    logger.warning(f"Split {split} not found at {split_path}")
            
            if not dataset:
                raise FileNotFoundError(f"No dataset files found in {self.data_path}")
                
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset from local files: {e}")
            raise
    
    def load(self) -> Dict[str, Any]:
        """
        Load the FinQA dataset from either local files or HuggingFace.
        
        Returns:
            Dictionary containing the dataset splits.
        """
        if self.data_path and os.path.exists(self.data_path):
            return self.load_from_local()
        else:
            return self.load_from_huggingface()
    
    @staticmethod
    def extract_documents(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract the financial documents from the dataset.
        
        Args:
            dataset: The loaded dataset.
            
        Returns:
            List of financial documents.
        """
        documents = []
        
        # Handle HuggingFace dataset format
        if hasattr(dataset, 'values') and callable(getattr(dataset, 'values')):
            for split in dataset:
                for item in dataset[split]:
                    doc = {
                        'id': item['id'],
                        'pre_text': item['pre_text'],
                        'post_text': item['post_text'],
                        'table': item['table'],
                    }
                    documents.append(doc)
        # Handle local JSON format
        else:
            for split in dataset:
                for item in dataset[split]:
                    doc = {
                        'id': item['id'],
                        'pre_text': item['pre_text'],
                        'post_text': item['post_text'],
                        'table': item['table'],
                    }
                    documents.append(doc)
                    
        return documents
    
    @staticmethod
    def extract_qa_pairs(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract the question-answer pairs from the dataset.
        
        Args:
            dataset: The loaded dataset.
            
        Returns:
            List of QA pairs.
        """
        qa_pairs = []
        
        # Handle HuggingFace dataset format
        if hasattr(dataset, 'values') and callable(getattr(dataset, 'values')):
            for split in dataset:
                for item in dataset[split]:
                    if 'qa' in item:
                        qa = {
                            'id': item['id'],
                            'question': item['qa']['question'],
                            'answer': item['qa']['exe_ans'],
                            'program': item['qa']['program'],
                            'gold_inds': item['qa']['gold_inds'],
                        }
                        qa_pairs.append(qa)
        # Handle local JSON format
        else:
            for split in dataset:
                for item in dataset[split]:
                    if 'qa' in item:
                        qa = {
                            'id': item['id'],
                            'question': item['qa']['question'],
                            'answer': item['qa']['exe_ans'],
                            'program': item['qa']['program'],
                            'gold_inds': item['qa']['gold_inds'],
                        }
                        qa_pairs.append(qa)
                        
        return qa_pairs
