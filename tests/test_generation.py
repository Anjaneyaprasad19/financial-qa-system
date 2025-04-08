"""
Tests for answer generation functionality.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.generation.llm import LLMManager
from src.generation.generator import AnswerGenerator

class TestLLMManager:
    """Test cases for LLMManager."""
    
    @patch('openai.api_key', 'mock-api-key')
    @patch('src.generation.llm.ChatOpenAI')
    def test_generate_answer(self, mock_chat_openai):
        """Test answer generation."""
        # Setup mock
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        # Mock generation result
        mock_response = MagicMock()
        mock_generation = MagicMock()
        mock_generation.text = "Mock answer"
        mock_response.generations = [[mock_generation]]
        mock_response.llm_output = {"token_usage": {"total_tokens": 100}}
        mock_instance.generate.return_value = mock_response
        
        # Create LLM manager
        llm_manager = LLMManager(model_name="gpt-4", api_key="mock-api-key")
        
        # Generate answer
        result = llm_manager.generate_answer(
            question="Test question?",
            context="Test context",
            return_metadata=True
        )
        
        # Assertions
        assert result["answer"] == "Mock answer"
        assert "metadata" in result
        
        # Test without metadata
        result_simple = llm_manager.generate_answer(
            question="Test question?",
            context="Test context",
            return_metadata=False
        )
        
        assert result_simple == "Mock answer"
    
    @patch('openai.api_key', 'mock-api-key')
    @patch('src.generation.llm.ChatOpenAI')
    def test_check_confidence(self, mock_chat_openai):
        """Test confidence checking."""
        # Setup mock
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        # Mock generation result
        mock_response = MagicMock()
        mock_generation = MagicMock()
        mock_generation.text = "After analyzing the context and answer, I conclude with Confidence Score: 85"
        mock_response.generations = [[mock_generation]]
        mock_response.llm_output = {"token_usage": {"total_tokens": 100}}
        mock_instance.generate.return_value = mock_response
        
        # Create LLM manager
        llm_manager = LLMManager(model_name="gpt-4", api_key="mock-api-key")
        
        # Check confidence
        result = llm_manager.check_confidence(
            context="Test context",
            answer="Test answer",
            return_metadata=True
        )
        
        # Assertions
        assert result["score"] == 85.0
        assert "explanation" in result
        assert "metadata" in result

class TestAnswerGenerator:
    """Test cases for AnswerGenerator."""
    
    def test_prepare_context(self):
        """Test context preparation."""
        # Mock components
        retriever = MagicMock()
        llm_manager = MagicMock()
        
        # Mock retriever behavior
        retrieved_docs = [
            (MagicMock(), 0.9),
            (MagicMock(), 0.8)
        ]
        retriever.retrieve.return_value = retrieved_docs
        retriever.format_retrieved_contexts.return_value = "Formatted context"
        
        # Create generator
        generator = AnswerGenerator(
            retriever=retriever,
            llm_manager=llm_manager,
            max_context_length=1000
        )
        
        # Prepare context
        context, docs = generator._prepare_context("Test query")
        
        # Assertions
        assert context == "Formatted context"
        assert docs == retrieved_docs
        retriever.retrieve.assert_called_once_with("Test query", top_k=5)
