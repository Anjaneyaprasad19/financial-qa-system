"""Streamlit demo for the financial QA system."""
import os
import sys
import yaml

import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Page configuration
st.set_page_config(
    page_title="Financial QA System",
    page_icon="💰",
    layout="wide",
)

# Title and description
st.title("Financial QA System with Hallucination Detection")
st.markdown(
    """This demo showcases a domain-specific question-answering system for financial data
    with built-in hallucination detection and confidence scoring."""
)

# Sidebar
st.sidebar.header("Configuration")

@st.cache_resource
def load_components():
    """Load and cache system components."""
    # Load configuration
    config_path = "configs/default.yaml"
    data_root = "data"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
        st.error("Vector store not found. Run data processing first.")
        st.stop()
    
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
    
    return {
        "generator": generator,
        "detector": detector,
    }

# Main content
try:
    components = load_components()
    generator = components["generator"]
    detector = components["detector"]
    
    # Input form
    st.subheader("Ask a financial question:")
    
    with st.form("query_form"):
        query = st.text_area("Enter your question:", height=100)
        check_confidence = st.checkbox("Detect hallucinations", value=True)
        show_details = st.checkbox("Show detailed results", value=False)
        
        submitted = st.form_submit_button("Submit")
    
    # Process query when submitted
    if submitted and query:
        with st.spinner("Processing query..."):
            # Generate answer with full result
            result = generator.generate(
                query=query,
                check_confidence=check_confidence,
                return_full_result=True
            )
            
            # Get context from retrieved documents
            context = "\n".join([doc["content"] for doc in result["retrieved_documents"]])
            
            # Get retrieval scores
            retrieval_scores = [doc["score"] for doc in result["retrieved_documents"]]
            
            # Detect hallucinations if requested
            detection_result = None
            if check_confidence:
                detection_result = detector.detect(
                    answer=result["answer"],
                    query=query,
                    context=context,
                    retrieval_scores=retrieval_scores
                )
        
        # Display answer
        st.subheader("Answer:")
        st.write(result["answer"])
        
        # Display confidence info if available
        if detection_result:
            confidence_score = detection_result["confidence_score"]
            
            # Determine color based on confidence
            if confidence_score >= 80:
                color = "green"
            elif confidence_score >= 60:
                color = "orange"
            else:
                color = "red"
                
            st.markdown(f"<h3 style='color:{color}'>Confidence Score: {confidence_score:.1f}/100</h3>", unsafe_allow_html=True)
            
            if detection_result["is_hallucination"]:
                st.warning("⚠️ This answer may contain hallucinated information!")
                
        # Display detailed results if requested
        if show_details:
            st.subheader("Retrieved Documents:")
            for i, doc in enumerate(result["retrieved_documents"]):
                with st.expander(f"Document {i+1} (Score: {doc['score']:.4f})"):
                    st.write(doc["content"])
                    st.write(f"Source: {doc['metadata']}")
            
            if detection_result:
                st.subheader("Verification Details:")
                st.write(detection_result["verification"])
                
                st.subheader("Confidence Components:")
                components_df = {
                    "Component": list(detection_result["confidence_components"].keys()),
                    "Score": list(detection_result["confidence_components"].values())
                }
                st.dataframe(components_df)

except Exception as e:
    st.error(f"Error loading system components: {str(e)}")

# Run with: streamlit run app/demo.py
