# Embedding Models Experimentation
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path for imports
sys.path.append(os.path.abspath('.'))

try:
    from src.data.loader import FinQADataLoader
    from src.data.processor import DocumentProcessor
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Install missing packages with: pip install langchain-community sentence-transformers")
    sys.exit(1)

# 1. Load Processed Documents
print("Loading processed documents...")
processed_dir = os.path.join('.', 'data', 'processed')
processor = DocumentProcessor(processed_dir=processed_dir)

try:
    chunks = processor.load_processed_chunks("finqa_processed.json")
    print(f"Loaded {len(chunks)} processed chunks")
except Exception as e:
    print(f"Error loading processed chunks: {e}")
    print("Please run 02_document_processing.py first")
    sys.exit(1)

# Take a sample for faster experimentation
sample_size = min(100, len(chunks))
sample_chunks = chunks[:sample_size]
print(f"Using {len(sample_chunks)} chunks for embedding model experimentation")

# 2. Define Embedding Models to Compare
embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2",  # Small general purpose
    # Uncomment more models if you want to test them
    # "sentence-transformers/all-mpnet-base-v2",  # Better quality, slower
    # "thenlper/gte-small",  # General Text Embeddings
]

model_info = {
    "sentence-transformers/all-MiniLM-L6-v2": {"dim": 384, "description": "Small general purpose"},
    "sentence-transformers/all-mpnet-base-v2": {"dim": 768, "description": "Better quality, slower"},
    "thenlper/gte-small": {"dim": 384, "description": "General Text Embeddings"},
}

# 3. Create sample queries for comparison
sample_queries = [
    "What was the company's revenue in 2018?",
    "How did the gross profit margin change from 2017 to 2018?",
    "What are the total operating expenses?"
]

# Extract text from chunks
chunk_texts = [chunk.page_content for chunk in sample_chunks]

# Function to time embedding generation
def generate_and_time_embeddings(model_name, texts, batch_size=32):
    print(f"Testing {model_name}")
    try:
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        
        # Time document embeddings (use just 10 docs for speed)
        test_texts = texts[:min(10, len(texts))]
        start_time = time.time()
        doc_embeddings = model.embed_documents(test_texts)
        doc_time = time.time() - start_time
        
        # Time query embeddings
        start_time = time.time()
        query_embeddings = [model.embed_query(q) for q in sample_queries]
        query_time = time.time() - start_time
        
        return {
            "model": model_name,
            "doc_embeddings": np.array(doc_embeddings),
            "query_embeddings": np.array(query_embeddings),
            "doc_time": doc_time,
            "query_time": query_time,
            "dimension": len(doc_embeddings[0]) if doc_embeddings else 0
        }
    except Exception as e:
        print(f"  Error with {model_name}: {e}")
        return None

# Generate embeddings for each model
results = []
for model_name in embedding_models:
    model_result = generate_and_time_embeddings(model_name, chunk_texts)
    if model_result is not None:
        results.append(model_result)
        print(f"  Completed. Dimension: {model_result['dimension']}, Doc time: {model_result['doc_time']:.2f}s, Query time: {model_result['query_time']:.2f}s")

# 4. Compare Performance
if not results:
    print("No embedding models were successfully tested")
    sys.exit(1)

comparison_data = []
for result in results:
    comparison_data.append({
        "Model": result["model"],
        "Description": model_info[result["model"]]["description"],
        "Dimension": result["dimension"],
        "Doc Embedding Time (s)": result["doc_time"],
        "Doc Embedding Time per Item (ms)": result["doc_time"] * 1000 / min(10, len(chunk_texts)),
        "Query Embedding Time (s)": result["query_time"],
        "Query Embedding Time per Item (ms)": result["query_time"] * 1000 / len(sample_queries)
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# 5. Test retrieval performance (basic)
print("\nTesting basic retrieval performance...")
for result in results:
    model_name = result["model"]
    print(f"\nModel: {model_name}")
    
    # Use the first query as an example
    query_idx = 0
    query = sample_queries[query_idx]
    query_embedding = result["query_embeddings"][query_idx]
    doc_embeddings = result["doc_embeddings"]
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Get top 3 results
    top_indices = np.argsort(-similarities)[:3]
    
    print(f"Query: '{query}'")
    print("Top 3 most relevant chunks:")
    
    for i, idx in enumerate(top_indices):
        if idx < len(chunk_texts):
            similarity = similarities[idx]
            print(f"#{i+1} (Score: {similarity:.4f}): {chunk_texts[idx][:100]}...")

print("\nEmbedding models evaluation complete!")
