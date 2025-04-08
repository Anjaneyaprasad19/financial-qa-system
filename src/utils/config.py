"""Configuration utilities for the QA system."""
import os
import yaml
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a nested configuration value using dot notation."""
    parts = path.split('.')
    current = config
    
    for part in parts:
        if part not in current:
            return default
        current = current[part]
    
    return current
