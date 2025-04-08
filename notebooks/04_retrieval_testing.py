# Retrieval Testing for Financial QA System
#
# This script evaluates the retrieval component of our financial QA system,
# focusing on vector search effectiveness and parameter tuning.

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
    from src.data.processor import DocumentProcessor
    from src.data.embedder import DocumentEmbedder
    from src.retrieval.vector_store import FAISSVectorStore
    from src.retrieval.retriever import DocumentRetriever
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you've implemented all necessary components")
    sys.exit(1)

# 1. Load processed documents and set up directories
print("Setting up directories and loading data...")
data_dir = os.path.join('.', 'data')
processed_dir = os.path.join(data_dir, 'processed')
embeddings_dir = os.path.join(data_dir, 'embeddings')

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

processor = DocumentProcessor(processed_dir=processed_dir)

try:
    chunks = processor.load_processed_chunks("finqa_processed.json")
    print(f"Loaded {len(chunks)} processed document chunks")
except Exception as e:
    print(f"Error loading processed chunks: {e}")
    print("Please run 02_document_processing.py first")
    sys.exit(1)

# 2. Set up embedding model
print("\nSetting up embedding model...")
try:
    embedder = DocumentEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        embeddings_dir=embeddings_dir
    )
    print("Embedding model initialized successfully")
except Exception as e:
    print(f"Error initializing embedding model: {e}")
    sys.exit(1)

# 3. Create and configure vector store
print("\nSetting up vector store...")
try:
    vector_store = FAISSVectorStore(
        embedding_dimension=384,  # Default for MiniLM-L6-v2
        index_type="IndexFlatL2",
        store_dir=embeddings_dir
    )
    print("Vector store initialized successfully")
except Exception as e:
    print(f"Error initializing vector store: {e}")
    sys.exit(1)

# 4. Check if vector store already exists
index_path = os.path.join(embeddings_dir, "finqa.index")
if os.path.exists(index_path):
    print("Found existing FAISS index, loading...")
    try:
        vector_store.load("finqa")
        print("Loaded existing vector store successfully")
    except Exception as e:
        print(f"Error loading existing vector store: {e}")
        print("Will create a new one instead")
        
        # Generate embeddings
        print("\nGenerating embeddings for document chunks...")
        chunk_subset = chunks[:min(1000, len(chunks))]  # Limit to 1000 chunks for speed
        try:
            embeddings = embedder.embed_documents(chunk_subset)
            print(f"Generated {len(embeddings)} embeddings")
            
            # Add to vector store
            vector_store.add_documents(chunk_subset, embeddings)
            print(f"Added {len(chunk_subset)} documents to vector store")
            
            # Save vector store
            vector_store.save("finqa")
            print("Saved vector store to disk")
        except Exception as e:
            print(f"Error generating embeddings or building vector store: {e}")
            sys.exit(1)
else:
    print("No existing vector store found, creating new one...")
    # Generate embeddings
    print("\nGenerating embeddings for document chunks...")
    chunk_subset = chunks[:min(1000, len(chunks))]  # Limit to 1000 chunks for speed
    try:
        embeddings = embedder.embed_documents(chunk_subset)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Add to vector store
        embeddings_np = np.array(embeddings, dtype=np.float32)
        vector_store.add_documents(chunk_subset, embeddings_np)
        print(f"Added {len(chunk_subset)} documents to vector store")
        
        # Save vector store
        vector_store.save("finqa")
        print("Saved vector store to disk")
    except Exception as e:
        print(f"Error generating embeddings or building vector store: {e}")
        sys.exit(1)

# 5. Initialize retriever
print("\nInitializing document retriever...")
retriever = DocumentRetriever(
    embedder=embedder,
    vector_store=vector_store,
    top_k=5
)
print("Retriever initialized successfully")

# 6. Define sample queries
print("\nDefining sample financial queries...")
sample_queries = [
    "What was the company's revenue in 2018?",
    "How did the gross profit margin change from 2017 to 2018?",
    "What are the total operating expenses?",
    "Explain the increase in net income",
    "What is the debt to equity ratio?",
    "How much cash does the company have on hand?",
    "What was the dividend payout ratio last year?",
    "How much did research and development spending increase?",
    "What is the current market capitalization?",
    "How has the stock price performed over the last year?"
]

# 7. Test retrieval with different k values
print("\nTesting retrieval with different k values...")
k_values = [1, 3, 5, 10, 20]
k_results = []

for k in k_values:
    retrieval_times = []
    avg_scores = []
    
    print(f"Testing with k={k}...")
    for query in tqdm(sample_queries):
        # Time the retrieval
        start_time = time.time()
        results = retriever.retrieve(query, top_k=k)
        retrieval_time = time.time() - start_time
        
        # Calculate average score
        scores = [score for _, score in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        retrieval_times.append(retrieval_time)
        avg_scores.append(avg_score)
    
    k_results.append({
        "k": k,
        "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times),
        "avg_similarity_score": sum(avg_scores) / len(avg_scores),
        "max_avg_score": max(avg_scores) if avg_scores else 0,
        "min_avg_score": min(avg_scores) if avg_scores else 0
    })

# 8. Show retrieval comparison results
k_results_df = pd.DataFrame(k_results)
print("\nRetrieval results with different k values:")
print(k_results_df.to_string(index=False))

# 9. Plot the retrieval results if matplotlib is available
try:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([r["k"] for r in k_results], [r["avg_retrieval_time"] * 1000 for r in k_results], marker='o')
    plt.title('Retrieval Time vs k')
    plt.xlabel('k (number of results)')
    plt.ylabel('Avg Retrieval Time (ms)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([r["k"] for r in k_results], [r["avg_similarity_score"] for r in k_results], marker='o')
    plt.title('Average Similarity Score vs k')
    plt.xlabel('k (number of results)')
    plt.ylabel('Avg Similarity Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('retrieval_metrics.png')
    print("Saved retrieval metrics visualization to 'retrieval_metrics.png'")
    plt.close()
except Exception as e:
    print(f"Could not create visualization: {e}")

# 10. Examine retrieved documents for a sample query
print("\nExamining retrieved documents for a sample query...")
sample_query = sample_queries[0]
print(f"Query: '{sample_query}'")

results = retriever.retrieve(sample_query, top_k=3)
print(f"Retrieved {len(results)} documents:")

for i, (doc, score) in enumerate(results):
    print(f"\nDocument {i+1} (Score: {score:.4f}):")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")

# 11. Test formatting retrieved contexts
print("\nTesting context formatting...")
formatted_context = retriever.format_retrieved_contexts(results)
print("Formatted context (first 500 chars):")
print(formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context)

# 12. Find optimal k based on metrics
best_k_for_speed = k_results_df.loc[k_results_df['avg_retrieval_time'].idxmin()]['k']
best_k_for_score = k_results_df.loc[k_results_df['avg_similarity_score'].idxmax()]['k']

print(f"\nBest k for speed: {best_k_for_speed}")
print(f"Best k for similarity score: {best_k_for_score}")

# Recommend a balanced k value
recommended_k = 5  # Default
for row in k_results_df.itertuples():
    if row.avg_retrieval_time < 0.1 and row.avg_similarity_score > 0.7 * k_results_df['avg_similarity_score'].max():
        recommended_k = row.k
        break

print(f"Recommended k value (balancing speed and quality): {recommended_k}")

print("\nRetrieval testing complete!")
