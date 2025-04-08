# System Evaluation for Financial QA System
#
# This script evaluates the end-to-end financial QA system, including retrieval, 
# generation, and hallucination detection components.

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.abspath('.'))

# Try to import the necessary modules
try:
    from src.data.loader import FinQADataLoader
    from src.data.processor import DocumentProcessor
    from src.data.embedder import DocumentEmbedder
    from src.retrieval.vector_store import FAISSVectorStore
    from src.retrieval.retriever import DocumentRetriever
    from src.generation.llm import LLMManager
    from src.generation.generator import AnswerGenerator
    from src.hallucination.confidence import ConfidenceScorer
    from src.hallucination.detector import HallucinationDetector
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Some functionality may be limited")
    sys.exit(1)

# 1. Set up directories
print("Setting up directories...")
data_dir = os.path.join('.', 'data')
processed_dir = os.path.join(data_dir, 'processed')
embeddings_dir = os.path.join(data_dir, 'embeddings')
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

# 2. Load test data
print("\nLoading test data...")
data_path = os.path.join('.', 'data', 'raw')
data_loader = FinQADataLoader(data_path=data_path)

try:
    dataset = data_loader.load_from_local()
    test_data = dataset['test']
    print(f"Loaded {len(test_data)} test examples")
    
    # Extract QA pairs
    test_qa_pairs = []
    for item in test_data:
        if 'qa' in item:
            test_qa_pairs.append({
                'id': item['id'],
                'question': item['qa']['question'],
                'answer': item['qa']['exe_ans'],
                'program': item['qa']['program'] if 'program' in item['qa'] else None
            })
    
    print(f"Extracted {len(test_qa_pairs)} QA pairs for testing")
    
    # Take a subset for quicker evaluation if needed
    max_test_samples = 20  # Adjust based on your needs
    test_qa_pairs = test_qa_pairs[:min(max_test_samples, len(test_qa_pairs))]
    print(f"Using {len(test_qa_pairs)} QA pairs for evaluation")
    
except Exception as e:
    print(f"Error loading test data: {e}")
    # Create mock test data for demonstration
    test_qa_pairs = [
        {'id': 'test1', 'question': 'What was the revenue in 2018?', 'answer': '$10.5 million'},
        {'id': 'test2', 'question': 'How did the gross profit margin change from 2017 to 2018?', 'answer': 'Increased by 2.3%'},
        {'id': 'test3', 'question': 'What were the total operating expenses?', 'answer': '$5.2 million'}
    ]
    print(f"Created {len(test_qa_pairs)} mock test examples")

# 3. Load system components
print("\nLoading system components...")

# Initialize document processor
processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200,
    processed_dir=processed_dir
)

# Load processed chunks
try:
    chunks = processor.load_processed_chunks("finqa_processed.json")
    print(f"Loaded {len(chunks)} processed chunks")
except Exception as e:
    print(f"Error loading processed chunks: {e}")
    print("Please run document processing script first")
    sys.exit(1)

# Initialize embedder
try:
    embedder = DocumentEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        embeddings_dir=embeddings_dir
    )
    print("Initialized embedding model")
except Exception as e:
    print(f"Error initializing embedding model: {e}")
    sys.exit(1)

# Initialize vector store
try:
    vector_store = FAISSVectorStore(
        embedding_dimension=384,
        index_type="IndexFlatL2",
        store_dir=embeddings_dir
    )
    
    # Load existing index if available
    if os.path.exists(os.path.join(embeddings_dir, "finqa.index")):
        vector_store.load("finqa")
        print("Loaded existing FAISS index")
    else:
        print("No existing index found, please run retrieval testing script first")
        sys.exit(1)
except Exception as e:
    print(f"Error initializing vector store: {e}")
    sys.exit(1)

# Initialize retriever
retriever = DocumentRetriever(
    embedder=embedder,
    vector_store=vector_store,
    top_k=5  # Use the recommended k from retrieval testing
)
print("Initialized document retriever")

# Initialize LLM manager
# Check if OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set")
    print("You can set it with: export OPENAI_API_KEY=your-key-here")
    print("Evaluation will use mock responses instead of actual LLM calls")
    
    # Create a mock LLM manager
    class MockLLMManager:
        def generate_answer(self, question, context, return_metadata=False):
            answer = f"Mock answer to: {question}"
            if return_metadata:
                return {"answer": answer, "metadata": {}}
            return answer
            
        def check_confidence(self, context, answer, return_metadata=False):
            confidence = 85.0  # Mock high confidence
            if return_metadata:
                return {"score": confidence, "explanation": "Mock confidence explanation", "metadata": {}}
            return confidence
    
    llm_manager = MockLLMManager()
    print("Using mock LLM manager")
else:
    try:
        llm_manager = LLMManager(
            model_name="gpt-4",  # Or another available model
            temperature=0.0,
            max_tokens=1024
        )
        print("Initialized real LLM manager with OpenAI API")
    except Exception as e:
        print(f"Error initializing LLM manager: {e}")
        sys.exit(1)

# Initialize answer generator
generator = AnswerGenerator(
    retriever=retriever,
    llm_manager=llm_manager,
    max_context_length=4000
)
print("Initialized answer generator")

# Initialize confidence scorer and hallucination detector
confidence_scorer = ConfidenceScorer(
    threshold=70.0,
    weights={"llm_confidence": 0.6, "retrieval_score": 0.2, "consistency_score": 0.2}
)

detector = HallucinationDetector(
    llm_manager=llm_manager,
    confidence_scorer=confidence_scorer
)
print("Initialized hallucination detector")

# 4. Evaluate system
print("\nEvaluating system performance...")
results = []

for qa_pair in tqdm(test_qa_pairs):
    question = qa_pair['question']
    reference = qa_pair['answer']
    
    try:
        # Measure generation time
        start_time = time.time()
        
        # Generate answer
        generation_result = generator.generate(
            query=question,
            check_confidence=True,
            return_full_result=True
        )
        
        generation_time = time.time() - start_time
        
        # Extract answer and metadata
        answer = generation_result["answer"]
        
        # Get retrieval scores
        retrieval_scores = [doc["score"] for doc in generation_result["retrieved_documents"]]
        
        # Get context
        context = "\n".join([doc["content"] for doc in generation_result["retrieved_documents"]])
        
        # Detect hallucinations
        detection_result = detector.detect(
            answer=answer,
            query=question,
            context=context,
            retrieval_scores=retrieval_scores
        )
        
        confidence_score = detection_result["confidence_score"]
        is_hallucination = detection_result["is_hallucination"]
        
        # Store results
        results.append({
            "id": qa_pair["id"],
            "question": question,
            "reference": reference,
            "answer": answer,
            "generation_time": generation_time,
            "confidence_score": confidence_score,
            "is_hallucination": is_hallucination,
            "top_retrieved_score": max(retrieval_scores) if retrieval_scores else 0
        })
        
    except Exception as e:
        print(f"Error processing question '{question}': {e}")
        results.append({
            "id": qa_pair["id"],
            "question": question,
            "reference": reference,
            "answer": f"Error: {str(e)}",
            "generation_time": 0,
            "confidence_score": 0,
            "is_hallucination": True,
            "top_retrieved_score": 0
        })

# 5. Analyze results
if not results:
    print("No evaluation results generated")
    sys.exit(1)

results_df = pd.DataFrame(results)
print("\nEvaluation results:")
print(f"Processed {len(results_df)} test questions")

# Calculate metrics
avg_generation_time = results_df['generation_time'].mean()
avg_confidence = results_df['confidence_score'].mean()
hallucination_rate = results_df['is_hallucination'].mean() * 100
avg_top_retrieval_score = results_df['top_retrieved_score'].mean()

print(f"Average generation time: {avg_generation_time:.2f} seconds")
print(f"Average confidence score: {avg_confidence:.2f}/100")
print(f"Hallucination rate: {hallucination_rate:.2f}%")
print(f"Average top retrieval score: {avg_top_retrieval_score:.4f}")

# 6. Show sample results
print("\nSample evaluation results:")
for i, row in results_df.head(min(3, len(results_df))).iterrows():
    print(f"\nQuestion: {row['question']}")
    print(f"Generated Answer: {row['answer'][:200]}..." if len(row['answer']) > 200 else row['answer'])
    print(f"Reference Answer: {row['reference']}")
    print(f"Confidence Score: {row['confidence_score']:.2f}/100")
    print(f"Hallucination Detected: {'Yes' if row['is_hallucination'] else 'No'}")
    print(f"Generation Time: {row['generation_time']:.2f} seconds")
    print("-" * 80)

# 7. Create and save visualizations
try:
    # Confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['confidence_score'], bins=10)
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.savefig('confidence_distribution.png')
    plt.close()
    print("Saved confidence score distribution to 'confidence_distribution.png'")
    
    # Generation time vs confidence
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['generation_time'], results_df['confidence_score'])
    plt.title('Generation Time vs Confidence Score')
    plt.xlabel('Generation Time (seconds)')
    plt.ylabel('Confidence Score')
    plt.grid(True)
    plt.savefig('time_vs_confidence.png')
    plt.close()
    print("Saved generation time vs confidence visualization to 'time_vs_confidence.png'")
    
except Exception as e:
    print(f"Error creating visualizations: {e}")

# 8. Save results to file
output_file = 'evaluation_results.json'
try:
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to '{output_file}'")
except Exception as e:
    print(f"Error saving results: {e}")
print("\nSystem evaluation complete!")