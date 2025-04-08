# Document Processing for Financial QA System
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.abspath('.'))

try:
    from src.data.loader import FinQADataLoader
    from src.data.processor import DocumentProcessor
except ImportError:
    print("Could not import required modules. Make sure the src module is properly set up.")
    sys.exit(1)

# 1. Load the Dataset
print("Loading dataset...")
data_path = os.path.join('.', 'data', 'raw')
data_loader = FinQADataLoader(data_path=data_path)

try:
    dataset = data_loader.load_from_local()
    print(f"Dataset loaded with {len(dataset['train'])} training examples")

    # Extract documents
    documents = data_loader.extract_documents(dataset)
    print(f"Extracted {len(documents)} unique documents")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# 2. Set up document processor
print("Setting up document processor...")
processed_dir = os.path.join('.', 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)

# Try different chunking strategies
print("\nExperimenting with different chunking strategies...")
chunk_sizes = [500, 1000, 1500]
chunk_overlaps = [50, 200, 300]

results = []

for size in chunk_sizes:
    for overlap in chunk_overlaps:
        processor = DocumentProcessor(chunk_size=size, 
                                     chunk_overlap=overlap,
                                     processed_dir=processed_dir)
        
        # Process a sample of documents (just 20 for speed)
        sample_size = min(20, len(documents))
        sample_docs = documents[:sample_size]
        chunks = []
        
        print(f"Processing with size={size}, overlap={overlap}")
        for doc in tqdm(sample_docs, desc=f"Size {size}, Overlap {overlap}"):
            try:
                doc_chunks = processor.process_document(doc)
                chunks.extend(doc_chunks)
            except Exception as e:
                print(f"Error processing document {doc.get('id', 'unknown')}: {e}")
        
        # Calculate statistics
        chunk_lengths = [len(chunk.page_content.split()) for chunk in chunks]
        if chunk_lengths:
            results.append({
                'chunk_size': size,
                'chunk_overlap': overlap,
                'num_chunks': len(chunks),
                'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
                'min_chunk_length': min(chunk_lengths),
                'max_chunk_length': max(chunk_lengths)
            })
            print(f"Created {len(chunks)} chunks, avg length: {sum(chunk_lengths) / len(chunk_lengths):.1f} words")
        else:
            print("No chunks created!")

# Show chunking results
print("\nChunking results:")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select optimal parameters based on results
print("\nSelecting optimal chunking parameters...")
# Look for chunk size that gives good coverage without excessive fragmentation
if not results_df.empty:
    # Find a good balance between chunk size and number of chunks
    # You might want to customize this selection logic
    optimal_row = results_df.iloc[len(results_df) // 2]  # Middle option as a compromise
    optimal_size = optimal_row['chunk_size']
    optimal_overlap = optimal_row['chunk_overlap']
else:
    # Default values if no results
    optimal_size = 1000
    optimal_overlap = 200

print(f"Selected optimal parameters: size={optimal_size}, overlap={optimal_overlap}")

# Process all documents with selected parameters
print(f"\nProcessing all documents with optimal parameters...")
processor = DocumentProcessor(chunk_size=optimal_size,
                             chunk_overlap=optimal_overlap,
                             processed_dir=processed_dir)

all_chunks = []
for doc in tqdm(documents, desc="Processing documents"):
    try:
        doc_chunks = processor.process_document(doc)
        all_chunks.extend(doc_chunks)
    except Exception as e:
        print(f"Error processing document {doc.get('id', 'unknown')}: {e}")

print(f"Created {len(all_chunks)} total chunks")

# Save processed chunks
output_path = processor.save_processed_chunks(all_chunks, "finqa_processed.json")
print(f"Saved processed chunks to {output_path}")

print("\nDocument processing complete!")
