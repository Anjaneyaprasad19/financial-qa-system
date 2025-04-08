"""Helper utilities for the QA system."""
import os
import json
import time
from typing import Dict, List, Any, Optional, Union

def ensure_directory(directory: str) -> str:
    """Ensure a directory exists and return its path."""
    os.makedirs(directory, exist_ok=True)
    return directory

def save_json(data: Union[Dict, List], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> Union[Dict, List]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """Format time in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

class Timer:
    """Simple timer class for measuring execution time."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or "Operation"
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} completed in {format_time(elapsed)}")
