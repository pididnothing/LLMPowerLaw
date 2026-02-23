"""
Custom Dataset Loader for LLM Benchmarking
Supports multiple data formats and sources
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import random


class CustomDataset:
    """Base class for custom datasets"""
    
    def __init__(
        self,
        name: str,
        task_type: str,
        data: List[Dict[str, Any]],
        fields: Dict[str, str]
    ):
        self.name = name
        self.task_type = task_type
        self.data = data
        self.fields = fields
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_samples(self, num_samples: Optional[int] = None, shuffle: bool = True, seed: int = 42):
        """Get a subset of samples"""
        data = self.data.copy()
        
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        
        if num_samples is not None and num_samples < len(data):
            data = data[:num_samples]
            
        return data


class DatasetLoader:
    """Main dataset loader supporting multiple formats"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Load JSONL format"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    @staticmethod
    def load_json(file_path: str) -> List[Dict[str, Any]]:
        """Load JSON format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            # Convert dict to list of dicts
            if 'data' in data:
                data = data['data']
            elif 'examples' in data:
                data = data['examples']
            else:
                # Assume dict contains the dataset structure
                data = [data]
        
        return data
    
    @staticmethod
    def load_csv(file_path: str) -> List[Dict[str, Any]]:
        """Load CSV format"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    @staticmethod
    def load_custom_dataset(
        name: str,
        file_path: str,
        task_type: str,
        format: str,
        fields: Dict[str, str],
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> CustomDataset:
        """
        Load a custom dataset from file
        
        Args:
            name: Dataset name
            file_path: Path to the dataset file
            task_type: Type of task (classification, qa, generation, etc.)
            format: File format (jsonl, json, csv)
            fields: Mapping of standard field names to dataset field names
            num_samples: Number of samples to load (None for all)
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
            
        Returns:
            CustomDataset instance
        """
        # Load data based on format
        loaders = {
            'jsonl': DatasetLoader.load_jsonl,
            'json': DatasetLoader.load_json,
            'csv': DatasetLoader.load_csv
        }
        
        if format not in loaders:
            raise ValueError(f"Unsupported format: {format}. Supported formats: {list(loaders.keys())}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        data = loaders[format](file_path)
        
        # Create dataset
        dataset = CustomDataset(
            name=name,
            task_type=task_type,
            data=data,
            fields=fields
        )
        
        return dataset
    
    @staticmethod
    def create_example_datasets():
        """Create example dataset files for reference"""
        data_dir = Path("./data_loaders/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Example classification dataset (JSONL)
        classification_examples = [
            {"input": "This movie was absolutely fantastic!", "label": "positive"},
            {"input": "Terrible experience, would not recommend.", "label": "negative"},
            {"input": "It was okay, nothing special.", "label": "neutral"},
        ]
        with open(data_dir / "custom_classification.jsonl", 'w', encoding='utf-8') as f:
            for example in classification_examples:
                f.write(json.dumps(example) + '\n')
        
        # Example QA dataset (CSV)
        qa_examples = pd.DataFrame([
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "context": "France is a country in Europe with Paris as its capital."
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare",
                "context": "Romeo and Juliet is a famous play by William Shakespeare."
            }
        ])
        qa_examples.to_csv(data_dir / "custom_qa.csv", index=False)
        
        # Example generation dataset (JSON)
        generation_examples = {
            "data": [
                {
                    "prompt": "Write a haiku about artificial intelligence",
                    "reference": "Silicon neurons\nLearning patterns, seeking truth\nMind in the machine"
                },
                {
                    "prompt": "Explain quantum computing in simple terms",
                    "reference": "Quantum computers use quantum bits that can be 0 and 1 simultaneously."
                }
            ]
        }
        with open(data_dir / "custom_generation.json", 'w', encoding='utf-8') as f:
            json.dump(generation_examples, f, indent=2)
        
        print(f"Example datasets created in {data_dir}")


if __name__ == "__main__":
    # Create example datasets
    DatasetLoader.create_example_datasets()
    
    # Test loading
    loader = DatasetLoader()
    dataset = loader.load_custom_dataset(
        name="test_classification",
        file_path="./data_loaders/data/custom_classification.jsonl",
        task_type="classification",
        format="jsonl",
        fields={"text": "input", "label": "label"}
    )
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Sample: {dataset[0]}")
