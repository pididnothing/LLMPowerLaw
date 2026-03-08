"""
Experiment Configuration Manager
Handles loading and validating experiment configurations
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    provider: str
    model_id: str
    max_tokens: int
    temperature: float
    enabled: bool
    additional_params: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        # Extract standard fields
        standard_fields = {
            'name', 'provider', 'model_id', 'max_tokens', 
            'temperature', 'enabled'
        }
        
        # Store additional parameters
        additional_params = {
            k: v for k, v in data.items() 
            if k not in standard_fields
        }
        
        return cls(
            name=data['name'],
            provider=data['provider'],
            model_id=data['model_id'],
            max_tokens=data.get('max_tokens', 512),
            temperature=data.get('temperature', 0.0),
            enabled=data.get('enabled', True),
            additional_params=additional_params
        )


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    type: str  # promptbench, custom, huggingface
    task_type: str
    num_samples: Optional[int]
    enabled: bool
    additional_params: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetConfig':
        """Create DatasetConfig from dictionary"""
        standard_fields = {
            'name', 'type', 'task_type', 'num_samples', 'enabled'
        }
        
        additional_params = {
            k: v for k, v in data.items() 
            if k not in standard_fields
        }
        
        return cls(
            name=data['name'],
            type=data['type'],
            task_type=data['task_type'],
            num_samples=data.get('num_samples'),
            enabled=data.get('enabled', True),
            additional_params=additional_params
        )


@dataclass
class ExperimentTriplet:
    """Configuration for a single experiment triplet (model-dataset-technique)"""
    id: str
    model: str
    dataset: str
    prompting_technique: str
    description: str = ""
    enabled: bool = True
    num_samples: Optional[int] = None  # Override dataset num_samples
    max_tokens: Optional[int] = None   # Override model max_tokens
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentTriplet':
        """Create ExperimentTriplet from dictionary"""
        return cls(
            id=data['id'],
            model=data['model'],
            dataset=data['dataset'],
            prompting_technique=data['prompting_technique'],
            description=data.get('description', ''),
            enabled=data.get('enabled', True),
            num_samples=data.get('num_samples'),
            max_tokens=data.get('max_tokens')
        )


class ExperimentConfig:
    """Main configuration manager for experiments"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.models: List[ModelConfig] = []
        self.datasets: List[DatasetConfig] = []
        self.experiments: List[ExperimentTriplet] = []
        self.global_model_settings: Dict[str, Any] = {}
        self.global_dataset_settings: Dict[str, Any] = {}
        self.global_experiment_settings: Dict[str, Any] = {}
        
        # Load environment variables
        load_dotenv()
        
    def load_configs(self):
        """Load all configuration files"""
        self.load_model_config()
        self.load_dataset_config()
        self.load_experiment_config()
        
    def load_model_config(self):
        """Load model configuration"""
        config_path = self.config_dir / "models.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.models = [
            ModelConfig.from_dict(model_data)
            for model_data in config.get('models', [])
        ]
        
        self.global_model_settings = config.get('global_settings', {})
        
    def load_dataset_config(self):
        """Load dataset configuration"""
        config_path = self.config_dir / "datasets.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.datasets = [
            DatasetConfig.from_dict(dataset_data)
            for dataset_data in config.get('datasets', [])
        ]
        
        self.global_dataset_settings = config.get('global_settings', {})
    
    def load_experiment_config(self):
        """Load experiment triplet configuration"""
        config_path = self.config_dir / "experiments.yaml"
        
        # Experiments config is optional
        if not config_path.exists():
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.experiments = [
            ExperimentTriplet.from_dict(exp_data)
            for exp_data in config.get('experiments', [])
        ]
        
        self.global_experiment_settings = config.get('global_settings', {})
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models"""
        return [m for m in self.models if m.enabled]
    
    def get_enabled_datasets(self) -> List[DatasetConfig]:
        """Get list of enabled datasets"""
        return [d for d in self.datasets if d.enabled]
    
    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name"""
        for model in self.models:
            if model.name == name:
                return model
        return None
    
    def get_dataset_by_name(self, name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration by name"""
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None
    
    def get_enabled_experiments(self) -> List[ExperimentTriplet]:
        """Get list of enabled experiment triplets"""
        return [e for e in self.experiments if e.enabled]
    
    def get_experiment_by_id(self, exp_id: str) -> Optional[ExperimentTriplet]:
        """Get experiment by ID"""
        for exp in self.experiments:
            if exp.id == exp_id:
                return exp
        return None
    
    def use_experiment_mode(self) -> bool:
        """Check if experiment triplet mode should be used"""
        return len(self.get_enabled_experiments()) > 0
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        required_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'huggingface': 'HUGGINGFACE_TOKEN',
            'google': 'GOOGLE_API_KEY'
        }
        
        enabled_providers = set(m.provider for m in self.get_enabled_models())
        
        validation_results = {}
        for provider in enabled_providers:
            if provider in required_keys:
                key_name = required_keys[provider]
                validation_results[provider] = bool(os.getenv(key_name))
        
        return validation_results
    
    def summary(self) -> str:
        """Generate a summary of the current configuration"""
        enabled_models = self.get_enabled_models()
        enabled_datasets = self.get_enabled_datasets()
        enabled_experiments = self.get_enabled_experiments()
        
        summary = []
        summary.append("=" * 60)
        summary.append("EXPERIMENT CONFIGURATION SUMMARY")
        summary.append("=" * 60)
        
        # Check if using experiment triplet mode
        if self.use_experiment_mode():
            summary.append(f"\nMode: EXPERIMENT TRIPLET MODE")
            summary.append(f"\nEnabled Experiments ({len(enabled_experiments)}):")
            for exp in enabled_experiments:
                summary.append(f"  - [{exp.id}] {exp.model} → {exp.dataset} → {exp.prompting_technique}")
                if exp.description:
                    summary.append(f"    {exp.description}")
            summary.append(f"\nTotal Experiments: {len(enabled_experiments)}")
        else:
            summary.append(f"\nMode: AUTO-COMBINATION MODE")
            summary.append(f"\nEnabled Models ({len(enabled_models)}):")
            for model in enabled_models:
                summary.append(f"  - {model.name} ({model.provider})")
            
            summary.append(f"\nEnabled Datasets ({len(enabled_datasets)}):")
            for dataset in enabled_datasets:
                samples = dataset.num_samples or "all"
                summary.append(f"  - {dataset.name} ({dataset.type}, {samples} samples)")
            
            summary.append(f"\nTotal Experiments: {len(enabled_models) * len(enabled_datasets)}")
        
        # API Key validation
        api_validation = self.validate_api_keys()
        if api_validation:
            summary.append("\nAPI Key Status:")
            for provider, valid in api_validation.items():
                status = "✓" if valid else "✗"
                summary.append(f"  {status} {provider}")
        
        summary.append("=" * 60)
        return "\n".join(summary)


if __name__ == "__main__":
    # Test configuration loading
    config = ExperimentConfig()
    config.load_configs()
    print(config.summary())
