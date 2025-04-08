"""
LLM integration for financial question-answering system.
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union

import openai
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logger = logging.getLogger(__name__)

# Ensure API key is loaded from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LLMManager:
    """
    Class to manage LLM interactions for the QA system.
    """
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM manager.
        
        Args:
            model_name: Name of the OpenAI model to use.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum number of tokens to generate.
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        else:
            raise ValueError("OpenAI API key not provided")
        
        # Initialize LangChain models
        if model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4"):
            self.chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.completion_model = OpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        logger.info(f"Initialized LLM manager with model {model_name}")
        
        # Initialize prompt templates
        self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self):
        """Initialize various prompt templates for different tasks."""
        # QA prompt template
        self.qa_system_template = """You are a financial analyst assistant specializing in answering questions based on financial documents.
        Use only the provided context to answer the question. If the context doesn't contain the relevant information,
        say "I don't have enough information to answer this question" instead of making up an answer.
        Be precise, accurate, and base your answer exclusively on the provided financial data.
        
        Context:
        {context}
        """
        
        self.qa_human_template = """Question: {question}"""
        
        self.qa_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.qa_system_template),
            HumanMessagePromptTemplate.from_template(self.qa_human_template)
        ])
        
        # Confidence check prompt template
        self.confidence_system_template = """You are a financial data verification system.
        You need to verify if the provided answer is supported by the context.
        Analyze carefully and provide a confidence score from 0 to 100, where:
        - 0 means the answer is completely unsupported by or contradictory to the context
        - 100 means the answer is fully supported by the context with no discrepancies
        
        Context:
        {context}
        
        Answer to verify:
        {answer}
        """
        
        self.confidence_human_template = """What is your confidence score for this answer?
        Explain your reasoning step by step, then conclude with 'Confidence Score: X' where X is the numerical score."""
        
        self.confidence_chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.confidence_system_template),
            HumanMessagePromptTemplate.from_template(self.confidence_human_template)
        ])
    
    def generate_answer(
        self,
        question: str,
        context: str,
        return_metadata: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate an answer to a question given context.
        
        Args:
            question: Question to answer.
            context: Context information.
            return_metadata: Whether to return metadata.
            
        Returns:
            Generated answer or dict with answer and metadata.
        """
        try:
            # Prepare the prompt
            messages = self.qa_chat_prompt.format_messages(
                context=context,
                question=question
            )
            
            # Generate the answer
            response = self.chat_model.generate([messages])
            
            # Extract the answer
            answer = response.generations[0][0].text.strip()
            
            if return_metadata:
                return {
                    "answer": answer,
                    "model": self.model_name,
                    "metadata": response.llm_output,
                }
            else:
                return answer
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            if return_metadata:
                return {
                    "answer": "Sorry, I encountered an error while processing your question.",
                    "error": str(e)
                }
            else:
                return "Sorry, I encountered an error while processing your question."
    
    def check_confidence(
        self,
        context: str,
        answer: str,
        return_metadata: bool = False
    ) -> Union[Dict[str, Any], float]:
        """
        Check the confidence of an answer against the context.
        
        Args:
            context: Context information.
            answer: Generated answer to check.
            return_metadata: Whether to return metadata.
            
        Returns:
            Confidence score (0-100) or dict with score and metadata.
        """
        try:
            # Prepare the prompt
            messages = self.confidence_chat_prompt.format_messages(
                context=context,
                answer=answer
            )
            
            # Generate the evaluation
            response = self.chat_model.generate([messages])
            
            # Extract the evaluation
            evaluation = response.generations[0][0].text.strip()
            
            # Parse the confidence score
            try:
                # Look for the pattern "Confidence Score: X"
                if "Confidence Score:" in evaluation:
                    score_part = evaluation.split("Confidence Score:")[1].strip()
                    # Extract the numeric part
                    score = float(score_part.split()[0])
                else:
                    # Fallback if the pattern is not found
                    score = 50.0  # Neutral score
                    logger.warning("Confidence score not found in evaluation. Using neutral score.")
            except Exception as e:
                logger.error(f"Error parsing confidence score: {e}")
                score = 50.0  # Neutral score
            
            if return_metadata:
                return {
                    "score": score,
                    "explanation": evaluation,
                    "metadata": response.llm_output,
                }
            else:
                return score
                
        except Exception as e:
            logger.error(f"Error checking confidence: {e}")
            if return_metadata:
                return {
                    "score": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "error": str(e)
                }
            else:
                return 0.0
