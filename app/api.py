"""API implementation for the QA system."""
import os
from typing import Dict, Any, Optional

import yaml
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.data.loader import FinQADataLoader
from src.data.processor import DocumentProcessor
from src.data.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import DocumentRetriever
from src.generation.llm import LLMManager
from src.generation.generator import AnswerGenerator
from src.hallucination.confidence import ConfidenceScorer
from src.hallucination.detector import HallucinationDetector
from src.utils.config import load_config

# Create FastAPI app
app = FastAPI(
    title="Financial QA System API",
    description="API for financial domain-specific QA with hallucination detection",
    version="0.1.0",
)

# --- Models ---

class QueryRequest(BaseModel):
    """Request model for QA query."""
    
    query: str
    check_confidence: bool = True
    full_result: bool = False


class QueryResponse(BaseModel):
    """Response model for QA query."""
    
    answer: str
    confidence_score: Optional[float] = None
    is_hallucination: Optional[bool] = None
    warning: Optional[str] = None
    verification: Optional[str] = None
    retrieved_documents: Optional[list] = None


# --- Dependencies ---

def get_components():
    """Dependency to get system components."""
    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    data_root = os.getenv("DATA_ROOT", "data")
    
    config = load_config(config_path)
    
    # Set up directories
    data_dir = os.path.join(data_root, "raw")
    processed_dir = os.path.join(data_root, "processed")
    embeddings_dir = os.path.join(data_root, "embeddings")
    
    # Initialize components
    # Data loader
    data_loader = FinQADataLoader(data_path=data_dir)
    
    # Document processor
    processor = DocumentProcessor(
        chunk_size=config["processing"]["chunk_size"],
        chunk_overlap=config["processing"]["chunk_overlap"],
        processed_dir=processed_dir
    )
    
    # Document embedder
    embedder = DocumentEmbedder(
        model_name=config["embedding"]["model_name"],
        device=config["embedding"]["device"],
        embeddings_dir=embeddings_dir
    )
    
    # Vector store
    vector_store = FAISSVectorStore(
        embedding_dimension=config["retrieval"]["embedding_dimension"],
        index_type=config["retrieval"]["index_type"],
        store_dir=embeddings_dir
    )
    
    # Load vector store if it exists
    vector_store_path = os.path.join(embeddings_dir, "finqa.index")
    if os.path.exists(vector_store_path):
        vector_store.load("finqa")
    else:
        raise HTTPException(
            status_code=500,
            detail="Vector store not found. Run data processing first."
        )
    
    # Document retriever
    retriever = DocumentRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=config["retrieval"]["top_k"]
    )
    
    # LLM manager
    llm_manager = LLMManager(
        model_name=config["generation"]["model_name"],
        temperature=config["generation"]["temperature"],
        max_tokens=config["generation"]["max_tokens"]
    )
    
    # Answer generator
    generator = AnswerGenerator(
        retriever=retriever,
        llm_manager=llm_manager,
        max_context_length=config["generation"]["max_context_length"]
    )
    
    # Confidence scorer
    confidence_scorer = ConfidenceScorer(
        threshold=config["hallucination"]["threshold"],
        weights=config["hallucination"]["weights"]
    )
    
    # Hallucination detector
    detector = HallucinationDetector(
        llm_manager=llm_manager,
        confidence_scorer=confidence_scorer,
        verification_rounds=config["hallucination"]["verification_rounds"]
    )
    
    components = {
        "generator": generator,
        "detector": detector,
    }
    
    return components


# --- Routes ---

@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"message": "Financial QA System API"}


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    components: Dict[str, Any] = Depends(get_components)
):
    """Process a QA query."""
    generator = components["generator"]
    detector = components["detector"]
    
    try:
        # Generate answer with full result
        result = generator.generate(
            query=request.query,
            check_confidence=request.check_confidence,
            return_full_result=True
        )
        
        # Basic response with just the answer
        response = {"answer": result["answer"]}
        
        if request.check_confidence:
            # Get context from retrieved documents
            context = "\n".join([doc["content"] for doc in result["retrieved_documents"]])
            
            # Get retrieval scores
            retrieval_scores = [doc["score"] for doc in result["retrieved_documents"]]
            
            # Detect hallucinations
            detection_result = detector.detect(
                answer=result["answer"],
                query=request.query,
                context=context,
                retrieval_scores=retrieval_scores
            )
            
            # Add detection results to response
            response["confidence_score"] = detection_result["confidence_score"]
            response["is_hallucination"] = detection_result["is_hallucination"]
            
            if detection_result["is_hallucination"]:
                response["warning"] = (
                    "The generated answer may contain hallucinated information "
                    "not supported by the source documents."
                )
                
            if request.full_result:
                response["verification"] = detection_result["verification"]
        
        # Add retrieved documents if requested
        if request.full_result:
            response["retrieved_documents"] = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": doc["score"]
                }
                for doc in result["retrieved_documents"]
            ]
            
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
