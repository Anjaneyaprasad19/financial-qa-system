# FinQA Dataset Exploration
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add project root to path for imports
sys.path.append(os.path.abspath('.'))

try:
    from src.data.loader import FinQADataLoader
except ImportError:
    print("Could not import FinQADataLoader. Make sure the src module is properly set up.")

# 1. Load the Dataset
print("Loading dataset from local files...")
data_path = os.path.join('.', 'data', 'raw')
data_loader = FinQADataLoader(data_path=data_path)

try:
    dataset = data_loader.load_from_local()
    print(f"Dataset loaded successfully with keys: {dataset.keys()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# 2. Look at the first example
try:
    example = dataset['train'][0]
    print(f"Example ID: {example['id']}")
    print(f"\nQuestion: {example['qa']['question']}")
    print(f"\nAnswer: {example['qa']['exe_ans']}")
    print(f"\nProgram: {example['qa']['program']}")
except Exception as e:
    print(f"Error examining example: {e}")

# 3. Analyzing the Dataset Structure
# Count number of examples in each split
print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")
print(f"Test examples: {len(dataset['test'])}")

# 4. Extract all questions and analyze lengths
try:
    train_questions = [item['qa']['question'] for item in dataset['train']]
    val_questions = [item['qa']['question'] for item in dataset['validation']]
    test_questions = [item['qa']['question'] for item in dataset['test']]
    
    # Calculate average question length
    avg_train_len = sum(len(q.split()) for q in train_questions) / len(train_questions) if train_questions else 0
    avg_val_len = sum(len(q.split()) for q in val_questions) / len(val_questions) if val_questions else 0
    avg_test_len = sum(len(q.split()) for q in test_questions) / len(test_questions) if test_questions else 0
    
    print(f"Average question length (words):")
    print(f"  Train: {avg_train_len:.2f}")
    print(f"  Validation: {avg_val_len:.2f}")
    print(f"  Test: {avg_test_len:.2f}")
except Exception as e:
    print(f"Error analyzing questions: {e}")

# 5. Analyze document structure
print("\nAnalyzing document structure...")
try:
    for i in range(min(3, len(dataset['train']))):
        example = dataset['train'][i]
        print(f"Example {i+1} - ID: {example['id']}")
        
        if isinstance(example['table'], dict) and 'header' in example['table']:
            print(f"Table headers: {example['table']['header']}")
            print(f"Number of rows: {len(example['table'].get('rows', []))}")
        else:
            print(f"Table format: {type(example['table'])}")
        
        print(f"Pre-text length: {len(example['pre_text'].split())} words")
        print(f"Post-text length: {len(example['post_text'].split())} words")
        print("\n" + "-"*50 + "\n")
except Exception as e:
    print(f"Error analyzing document structure: {e}")

# 6. Analyze financial operations
print("\nAnalyzing financial operations...")
try:
    def extract_operations(program):
        operations = []
        current_op = ''
        recording = False
        
        for char in program:
            if char.isalpha() or char == '_':
                if not recording:
                    recording = True
                    current_op = ''
                current_op += char
            elif char == '(' and recording:
                operations.append(current_op)
                recording = False
            elif not (char.isalpha() or char == '_') and recording:
                recording = False
        
        return operations

    # Extract operations from training set
    all_ops = []
    for item in dataset['train']:
        if 'qa' in item and 'program' in item['qa']:
            ops = extract_operations(item['qa']['program'])
            all_ops.extend(ops)

    op_counts = Counter(all_ops)
    print("Most common operations:")
    for op, count in op_counts.most_common(10):
        print(f"  {op}: {count}")
    
    # If matplotlib is working, create a visualization
    try:
        top_ops = dict(op_counts.most_common(10))
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(top_ops.keys()), y=list(top_ops.values()))
        plt.title('Most Common Financial Operations in FinQA')
        plt.xlabel('Operation')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('top_operations.png')
        print("\nSaved operations visualization to 'top_operations.png'")
        plt.close()
    except Exception as e:
        print(f"Could not create visualization: {e}")
        
except Exception as e:
    print(f"Error analyzing operations: {e}")

print("\nExploration complete!")
