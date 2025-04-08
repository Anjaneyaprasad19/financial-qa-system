"""
Main application for financial question-answering system.
"""
import os
import argparse
import logging
from typing import Dict, List, Any, Optional

import yaml
from dotenv import load_dotenv

from src.data.loader import FinQADataLoader
from src.data.processor import DocumentProcessor
from src.data.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import DocumentRetriever
from src.generation.llm import LLMManager
from src.generation.generator import AnswerGenerator
from src.hallucination.confidence import ConfidenceScorer
from src.hallucination.detector import HallucinationDetector

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_system(config: Dict[str, Any], data_root: str) -> Dict[str, Any]:
    """Set up the financial QA system components."""
    # Set up directories
    data_dir = os.path.join(data_root, "raw")
    processed_dir = os.path.join(data_root, "processed")
    embeddings_dir = os.path.join(data_root, "embeddings")
    
    for directory in [data_dir, processed_dir, embeddings_dir]:
        os.makedirs(directory, exist_ok=True)
    
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
        "data_loader": data_loader,
        "processor": processor,
        "embedder": embedder,
        "vector_store": vector_store,
        "retriever": retriever,
        "llm_manager": llm_manager,
        "generator": generator,
        "confidence_scorer": confidence_scorer,
        "detector": detector,
    }
    
    return components

def process_dataset(components: Dict[str, Any], force_reload: bool = False) -> None:
    """Process the FinQA dataset and build the vector store."""
    data_loader = components["data_loader"]
    processor = components["processor"]
    embedder = components["embedder"]
    vector_store = components["vector_store"]
    
    # Check if processed files already exist
    processed_path = os.path.join(processor.processed_dir, "finqa_processed.json")
    vector_store_path = os.path.join(vector_store.store_dir, "finqa.index")
    
    if not force_reload and os.path.exists(processed_path) and os.path.exists(vector_store_path):
        logger.info("Using existing processed files...")
        # Load processed chunks
        chunks = processor.load_processed_chunks("finqa_processed.json")
        # Load vector store
        vector_store.load("finqa")
        return
    
    # Load dataset
    logger.info("Loading FinQA dataset...")
    dataset = data_loader.load()
    
    # Extract documents
    logger.info("Extracting documents...")
    documents = data_loader.extract_documents(dataset)
    
    # Process documents
    logger.info("Processing documents...")
    chunks = processor.process_batch(documents)
    
    # Save processed chunks
    logger.info("Saving processed chunks...")
    processor.save_processed_chunks(chunks, "finqa_processed.json")
    
    # Generate embeddings
    logger.info("Generating document embeddings...")
    embeddings = embedder.embed_documents(chunks)
    
    # Add to vector store
    logger.info("Building vector store...")
    vector_store.add_documents(chunks, embeddings)
    
    # Save vector store
    logger.info("Saving vector store...")
    vector_store.save("finqa")
    
def interactive_qa(components: Dict[str, Any]) -> None:
    """Run interactive QA session."""
    generator = components["generator"]
    detector = components["detector"]
    
    print("\n====== Financial QA System ======")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
        
        # Generate answer with full result
        result = generator.generate(
            query=query,
            check_confidence=True,
            return_full_result=True
        )
        
        # Get context from retrieved documents
        context = "\n".join([doc["content"] for doc in result["retrieved_documents"]])
        
        # Get retrieval scores
        retrieval_scores = [doc["score"] for doc in result["retrieved_documents"]]
        
        # Detect hallucinations
        detection_result = detector.detect(
            answer=result["answer"],
            query=query,
            context=context,
            retrieval_scores=retrieval_scores
        )
        
        # Display answer
        print("\n--- Answer ---")
        print(result["answer"])
        
        # Display confidence info
        print(f"\n--- Confidence Score: {detection_result['confidence_score']:.1f}/100 ---")
        
        if detection_result["is_hallucination"]:
            print("\nWARNING: This answer may contain hallucinated information!")
        
        # Option to show verification details
        show_details = input("\nShow verification details? (y/n): ")
        if show_details.lower() == 'y':
            print("\n--- Verification Details ---")
            print(detection_result["verification"])

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Financial QA System")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data", type=str, default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Force reload and reprocess the dataset"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up system components
    components = setup_system(config, args.data)
    
    # Process dataset
    process_dataset(components, force_reload=args.reload)
    
    if args.interactive:
        interactive_qa(components)

if __name__ == "__main__":
    main()
