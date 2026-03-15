"""
Main Benchmark Runner
Executes LLM benchmarking experiments using PromptBench
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from copy import copy
import traceback
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.experiment_config import ExperimentConfig, ModelConfig, DatasetConfig
from experiments.prompt_manager import PromptManager, PromptTechnique
from experiments.local_model_handler import LocalModelHandler
from data_loaders.custom_loader import DatasetLoader
from utils.logger import ExperimentLogger
from utils.metrics import MetricsCalculator

try:
    import promptbench as pb
    from promptbench.models import LLMModel
    PROMPTBENCH_AVAILABLE = True
except ImportError:
    PROMPTBENCH_AVAILABLE = False
    print("Warning: PromptBench not installed. Install with: pip install promptbench")


class BenchmarkRunner:
    """Main class for running LLM benchmarking experiments"""
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "./results",
        experiment_name: Optional[str] = None,
        enable_prompting: bool = True
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"
        
        # Initialize logger
        self.logger = ExperimentLogger(
            output_dir=self.output_dir,
            experiment_name=self.experiment_name
        )
        
        self.metrics_calculator = MetricsCalculator()
        
        # Load dataset instructions (for label maps etc.)
        self.dataset_instructions = {}
        try:
            import yaml
            instructions_path = Path(__file__).parent.parent / "config" / "dataset_instructions.yaml"
            if instructions_path.exists():
                with open(instructions_path, 'r') as f:
                    raw = yaml.safe_load(f)
                self.dataset_instructions = raw.get('dataset_instructions', {})
        except Exception as e:
            print(f"Warning: Could not load dataset_instructions.yaml: {e}")
        
        # Initialize prompt manager
        self.enable_prompting = enable_prompting
        self.prompt_manager = None
        if enable_prompting:
            try:
                self.prompt_manager = PromptManager()
                self.prompt_manager.load_config()
                self.logger.log_info("Prompt manager initialized")
            except Exception as e:
                self.logger.log_warning(f"Could not initialize prompt manager: {e}")
        
        # Cache for loaded models (for local models)
        self.loaded_models = {}

    def _extract_input_and_label(
        self,
        example: Dict[str, Any],
        dataset_config: DatasetConfig
    ) -> Tuple[str, Any]:
        """Extract task input text and raw label from a dataset example."""
        if dataset_config.type == "custom":
            fields = dataset_config.additional_params.get('fields', {})
            text_field = fields.get('text') or fields.get('prompt') or fields.get('question')
            label_field = fields.get('label') or fields.get('answer') or fields.get('reference')

            return example.get(text_field, ""), example.get(label_field, "")

        if dataset_config.type == "huggingface":
            fields = dataset_config.additional_params.get('fields', {})
            text_field = fields.get('text', 'text')
            label_field = fields.get('label', 'label')

            input_text = example.get(text_field, "")
            true_label = example.get(label_field, "")
            if not input_text:
                input_text = str(example.get('sentence', example.get('question', example.get('text', example))))
            return input_text, true_label

        return str(example.get('text', example.get('question', example))), example.get('label', example.get('answer', ''))

    def _normalize_true_label(self, true_label: Any, dataset_config: DatasetConfig) -> Any:
        """Normalize raw labels using dataset label maps when available."""
        ds_instructions = self.dataset_instructions.get(dataset_config.name, {})
        label_map = ds_instructions.get('label_map')
        if label_map and true_label in label_map:
            return label_map[true_label]
        if label_map:
            str_key = str(true_label)
            for key, value in label_map.items():
                if str(key) == str_key:
                    return value
        return true_label

    def _should_use_hf_chat_template(self, model_config: ModelConfig) -> bool:
        """Check whether this model should use tokenizer.apply_chat_template."""
        chat_template_mode = str(
            model_config.additional_params.get('chat_template_mode', 'manual')
        ).lower()
        return model_config.provider == 'huggingface_local' and chat_template_mode in {'hf', 'huggingface', 'apply_chat_template'}
        
    def load_dataset(self, dataset_config: DatasetConfig):
        """Load a dataset based on configuration"""
        if dataset_config.type == "promptbench":
            if not PROMPTBENCH_AVAILABLE:
                raise ImportError("PromptBench is not installed")
            
            # Load PromptBench dataset
            dataset_name = dataset_config.additional_params.get('dataset_name')
            # PromptBench dataset loading
            dataset = pb.DatasetLoader.load_dataset(dataset_name)
            
            if dataset_config.num_samples:
                dataset = dataset[:dataset_config.num_samples]
            
            return dataset
            
        elif dataset_config.type == "custom":
            # Load custom dataset
            loader = DatasetLoader()
            dataset = loader.load_custom_dataset(
                name=dataset_config.name,
                file_path=dataset_config.additional_params.get('file_path'),
                task_type=dataset_config.task_type,
                format=dataset_config.additional_params.get('format'),
                fields=dataset_config.additional_params.get('fields', {}),
                num_samples=dataset_config.num_samples,
                shuffle=self.config.global_dataset_settings.get('shuffle', True),
                seed=self.config.global_dataset_settings.get('seed', 42)
            )
            return dataset.get_samples(dataset_config.num_samples)
            
        elif dataset_config.type == "huggingface":
            # Load HuggingFace dataset
            from datasets import load_dataset
            
            dataset_name = dataset_config.additional_params.get('dataset_name')
            subset = dataset_config.additional_params.get('subset')
            split = dataset_config.additional_params.get('split', 'test')
            
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            if dataset_config.num_samples:
                dataset = dataset.select(range(min(dataset_config.num_samples, len(dataset))))
            
            return dataset
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_config.type}")

    def _build_local_model_runtime_config(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Flatten model config so LocalModelHandler receives runtime fields directly."""
        return {
            'name': model_config.name,
            'provider': model_config.provider,
            'model_id': model_config.model_id,
            'max_tokens': model_config.max_tokens,
            'temperature': model_config.temperature,
            'enabled': model_config.enabled,
            **model_config.additional_params,
        }
    
    def initialize_model(self, model_config: ModelConfig):
        """Initialize a model based on configuration"""
        provider = model_config.provider
        
        # Check if model is already loaded (for local models)
        if model_config.name in self.loaded_models:
            self.logger.log_info(f"Using cached model: {model_config.name}")
            return self.loaded_models[model_config.name]
        
        # Handle local models
        if provider in ['huggingface_local', 'gguf', 'vllm']:
            self.logger.log_info(f"Initializing local model: {model_config.name}")
            handler = LocalModelHandler(
                model_config=self._build_local_model_runtime_config(model_config),
                global_settings=self.config.global_model_settings
            )
            handler.load_model()
            self.loaded_models[model_config.name] = handler
            return handler
        
        # Handle PromptBench / API models
        if not PROMPTBENCH_AVAILABLE:
            # Return a mock model for testing without PromptBench
            return MockLLMModel(model_config)
        
        # Initialize model through PromptBench
        model = pb.LLMModel(
            model=model_config.model_id,
            max_new_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            **model_config.additional_params
        )
        
        return model
    
    def run_single_experiment(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        prompt_technique: Optional[PromptTechnique] = None,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a single experiment (one model on one dataset with optional prompting technique)"""
        prompt_name = prompt_technique.name if prompt_technique else "baseline"
        exp_id = experiment_id or f"{model_config.name}_{dataset_config.name}_{prompt_name}"
        
        self.logger.log_info(
            f"Starting experiment [{exp_id}]: {model_config.name} on {dataset_config.name} with {prompt_name}"
        )
        
        results = {
            'experiment_id': exp_id,
            'model': model_config.name,
            'dataset': dataset_config.name,
            'task_type': dataset_config.task_type,
            'prompting_technique': prompt_name,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'predictions': [],
            'metrics': {}
        }
        
        try:
            # Load dataset
            self.logger.log_info(f"Loading dataset: {dataset_config.name}")
            with tqdm(total=1, desc=f"Loading {dataset_config.name}", leave=False) as pbar:
                dataset = self.load_dataset(dataset_config)
                pbar.update(1)
            results['num_samples'] = len(dataset)
            self.logger.log_info(f"✓ Loaded {len(dataset)} samples")
            
            # Initialize model
            self.logger.log_info(f"Initializing model: {model_config.name}")
            model = self.initialize_model(model_config)
            
            # Run predictions
            self.logger.log_info(f"Running predictions with prompting: {prompt_name}")
            predictions = self._run_predictions(
                model, 
                dataset, 
                dataset_config,
                model_config,
                prompt_technique
            )
            results['predictions'] = predictions
            
            # Calculate metrics
            self.logger.log_info("Calculating metrics...")
            metrics = self._calculate_metrics(predictions, dataset_config.task_type)
            results['metrics'] = metrics
            
            results['status'] = 'completed'
            self.logger.log_info(
                f"Experiment completed. Accuracy: {metrics.get('accuracy', 'N/A')}"
            )
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
            self.logger.log_error(f"Experiment failed: {str(e)}")
        
        results['end_time'] = datetime.now().isoformat()
        
        # Save results
        self.logger.save_experiment_results(results)
        
        return results
    
    def _run_predictions(
        self,
        model,
        dataset,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        prompt_technique: Optional[PromptTechnique] = None
    ) -> List[Dict[str, Any]]:
        """Run model predictions on dataset with optional prompting technique"""
        predictions = []
        
        # Get few-shot examples if needed
        few_shot_examples = None
        if prompt_technique and self.prompt_manager:
            if prompt_technique.technique == 'few_shot' or 'few_shot' in prompt_technique.name.lower():
                # Use first few examples as few-shot examples (excluding them from test set)
                num_examples = prompt_technique.params.get('num_examples', 3)
                if len(dataset) > num_examples:
                    few_shot_examples = dataset[:num_examples]
                    dataset = dataset[num_examples:]  # Skip examples in test
        
        # Add progress bar for prediction loop
        pbar = tqdm(enumerate(dataset), total=len(dataset), 
                   desc=f"{model_config.name} on {dataset_config.name}",
                   unit="sample")
        use_hf_chat_template = self._should_use_hf_chat_template(model_config)
        
        # This is a simplified version - actual implementation would depend on
        # dataset structure and task type
        for i, example in pbar:
            try:
                input_text, true_label = self._extract_input_and_label(example, dataset_config)
                true_label = self._normalize_true_label(true_label, dataset_config)

                # Apply prompting technique if available
                prompt_text = input_text
                if prompt_technique and self.prompt_manager:
                    prompt_text = self.prompt_manager.apply_technique(
                        technique=prompt_technique,
                        input_text=input_text,
                        dataset_name=dataset_config.name,
                        task_type=dataset_config.task_type,
                        examples=few_shot_examples,
                        output_mode='content' if use_hf_chat_template else 'manual'
                    )

                model_input = prompt_text
                if isinstance(model, LocalModelHandler):
                    model_input = model.format_prompt(
                        prompt_text,
                        use_hf_chat_template=use_hf_chat_template
                    )
                
                # Get model prediction
                prediction_result = self._get_model_prediction(
                    model,
                    prompt_text,
                    model_config,
                    task_type=dataset_config.task_type,
                    use_hf_chat_template=use_hf_chat_template
                )
                
                # Handle both string predictions and dict (with raw/extracted)
                if isinstance(prediction_result, dict):
                    raw_prediction = prediction_result.get('raw', '')
                    extracted_prediction = prediction_result.get('extracted', '')
                else:
                    # Fallback for non-local models
                    raw_prediction = prediction_result
                    extracted_prediction = prediction_result
                
                # Log prediction details for first few samples
                if i < 3:  # Log first 3 predictions
                    self.logger.log_info(f"Sample {i}:")
                    self.logger.log_info(f"  Model input: '{model_input}'")
                    self.logger.log_info(f"  Raw prediction: '{raw_prediction}'")
                    self.logger.log_info(f"  Extracted: '{extracted_prediction}'")
                    self.logger.log_info(f"  True label: '{true_label}'")
                
                predictions.append({
                    'index': i,
                    'input': model_input,
                    'prompt_text': prompt_text,
                    'used_hf_chat_template': use_hf_chat_template,
                    'raw_prediction': raw_prediction,
                    'prediction': extracted_prediction,
                    'true_label': true_label
                })
                
                # Update progress bar with current accuracy
                if len(predictions) > 0 and i % 10 == 0:
                    valid_preds = [p for p in predictions if 'error' not in p]
                    if len(valid_preds) > 0:
                        pbar.set_postfix({'completed': len(valid_preds)})
                
            except Exception as e:
                self.logger.log_error(f"Error processing sample {i}: {str(e)}")
                predictions.append({
                    'index': i,
                    'error': str(e)
                })
        
        pbar.close()
        return predictions
    
    def _get_model_prediction(
        self,
        model,
        input_text: str,
        model_config: ModelConfig,
        task_type: str = None,
        use_hf_chat_template: bool = False
    ):
        """Get prediction from model (handles different model types)
        
        Returns:
            dict or str: For local models, returns {'raw': str, 'extracted': str}
                        For other models, returns str
        """
        # Handle local models
        if isinstance(model, LocalModelHandler):
            raw_response = model.generate(
                input_text,
                task_type=task_type,
                use_hf_chat_template=use_hf_chat_template
            )
            # Extract answer based on task type
            return model.extract_answer(raw_response, task_type)
        
        # Handle PromptBench models
        if hasattr(model, 'predict'):
            return model.predict(input_text)
        
        # Handle mock models
        if hasattr(model, 'config'):
            return f"Mock prediction from {model.config.name}"
        
        return "Mock prediction"
    
    def _calculate_metrics(
        self,
        predictions: List[Dict[str, Any]],
        task_type: str
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Extract valid predictions
        valid_predictions = [
            p for p in predictions 
            if 'error' not in p and 'prediction' in p and 'true_label' in p
        ]
        
        if not valid_predictions:
            return {'error': 'No valid predictions'}
        
        # Calculate metrics based on task type
        if task_type == "classification":
            return self.metrics_calculator.classification_metrics(valid_predictions)
        elif task_type == "qa":
            return self.metrics_calculator.qa_metrics(valid_predictions)
        elif task_type == "generation":
            return self.metrics_calculator.generation_metrics(valid_predictions)
        else:
            return self.metrics_calculator.basic_metrics(valid_predictions)
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all enabled experiments - uses triplet mode if experiments.yaml exists, otherwise auto-combination mode"""
        # Check if using experiment triplet mode
        if self.config.use_experiment_mode():
            return self._run_experiment_triplets()
        else:
            return self._run_auto_combinations()
    
    def _run_experiment_triplets(self) -> Dict[str, Any]:
        """Run experiments specified as explicit triplets in experiments.yaml"""
        enabled_experiments = self.config.get_enabled_experiments()
        
        self.logger.log_info(f"Starting benchmark suite (TRIPLET MODE): {self.experiment_name}")
        self.logger.log_info(f"Experiments: {len(enabled_experiments)}")
        
        all_results = {
            'experiment_name': self.experiment_name,
            'mode': 'triplet',
            'start_time': datetime.now().isoformat(),
            'config': {
                'experiments': [e.id for e in enabled_experiments]
            },
            'experiments': []
        }
        
        # Progress bar for overall experiments
        experiment_pbar = tqdm(total=len(enabled_experiments), desc="Overall Progress", 
                              position=0, leave=True, unit="exp")
        
        # Run each specified experiment
        for exp_triplet in enabled_experiments:
            # Lookup model and dataset configurations
            model_config = self.config.get_model_by_name(exp_triplet.model)
            dataset_config = self.config.get_dataset_by_name(exp_triplet.dataset)
            
            if not model_config:
                self.logger.log_warning(f"Model '{exp_triplet.model}' not found, skipping experiment {exp_triplet.id}")
                experiment_pbar.update(1)
                continue
            
            if not dataset_config:
                self.logger.log_warning(f"Dataset '{exp_triplet.dataset}' not found, skipping experiment {exp_triplet.id}")
                experiment_pbar.update(1)
                continue
            
            # Lookup prompting technique
            technique = None
            if exp_triplet.prompting_technique and self.prompt_manager:
                technique = self.prompt_manager.get_technique_by_name(exp_triplet.prompting_technique)
                if not technique:
                    self.logger.log_warning(f"Technique '{exp_triplet.prompting_technique}' not found, running without prompting")
            
            # Apply experiment-level overrides
            if exp_triplet.num_samples is not None:
                # Create a copy of dataset_config with overridden num_samples
                dataset_config = copy(dataset_config)
                dataset_config.num_samples = exp_triplet.num_samples
                self.logger.log_info(f"Overriding num_samples to {exp_triplet.num_samples} for experiment {exp_triplet.id}")
            
            if exp_triplet.max_tokens is not None:
                # Create a copy of model_config with overridden max_tokens
                model_config = copy(model_config)
                model_config.max_tokens = exp_triplet.max_tokens
                self.logger.log_info(f"Overriding max_tokens to {exp_triplet.max_tokens} for experiment {exp_triplet.id}")
            
            # Run experiment
            experiment_pbar.set_description(f"[{exp_triplet.id}] {exp_triplet.model}/{exp_triplet.dataset}")
            result = self.run_single_experiment(
                model_config,
                dataset_config,
                technique,
                experiment_id=exp_triplet.id
            )
            all_results['experiments'].append(result)
            experiment_pbar.update(1)
        
        experiment_pbar.close()
        all_results['end_time'] = datetime.now().isoformat()
        
        # Clean up loaded models
        self._cleanup_models()
        
        # Save summary
        summary_path = self.output_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.log_info(f"All experiments completed. Summary saved to {summary_path}")
        
        return all_results
    
    def _run_auto_combinations(self) -> Dict[str, Any]:
        """Run all combinations of enabled models and datasets (legacy mode)"""
    def _run_auto_combinations(self) -> Dict[str, Any]:
        """Run all combinations of enabled models and datasets (legacy mode)"""
        enabled_models = self.config.get_enabled_models()
        enabled_datasets = self.config.get_enabled_datasets()
        
        self.logger.log_info(f"Starting benchmark suite (AUTO-COMBINATION MODE): {self.experiment_name}")
        self.logger.log_info(f"Models: {len(enabled_models)}, Datasets: {len(enabled_datasets)}")
        
        all_results = {
            'experiment_name': self.experiment_name,
            'mode': 'auto_combination',
            'start_time': datetime.now().isoformat(),
            'config': {
                'models': [m.name for m in enabled_models],
                'datasets': [d.name for d in enabled_datasets]
            },
            'experiments': []
        }
        
        # Calculate total experiments to run
        total_experiments = 0
        for model_config in enabled_models:
            for dataset_config in enabled_datasets:
                if self.enable_prompting and self.prompt_manager:
                    techniques = self.prompt_manager.get_techniques_for_dataset(dataset_config.name)
                    total_experiments += len(techniques) if techniques else 1
                else:
                    total_experiments += 1
        
        # Progress bar for overall experiments
        experiment_pbar = tqdm(total=total_experiments, desc="Overall Progress", 
                              position=0, leave=True, unit="exp")
        
        # Run each combination
        for model_config in enabled_models:
            for dataset_config in enabled_datasets:
                # Get prompting techniques for this dataset
                techniques = []
                if self.enable_prompting and self.prompt_manager:
                    techniques = self.prompt_manager.get_techniques_for_dataset(dataset_config.name)
                
                # If no techniques configured, run without prompting
                if not techniques:
                    experiment_pbar.set_description(f"{model_config.name}/{dataset_config.name}/baseline")
                    result = self.run_single_experiment(model_config, dataset_config, None)
                    all_results['experiments'].append(result)
                    experiment_pbar.update(1)
                else:
                    # Run experiment for each prompting technique
                    for technique in techniques:
                        experiment_pbar.set_description(f"{model_config.name}/{dataset_config.name}/{technique.name}")
                        result = self.run_single_experiment(
                            model_config, 
                            dataset_config,
                            technique
                        )
                        all_results['experiments'].append(result)
                        experiment_pbar.update(1)
        
        experiment_pbar.close()
        all_results['end_time'] = datetime.now().isoformat()
        
        # Clean up loaded models
        self._cleanup_models()
        
        # Save summary
        summary_path = self.output_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.log_info(f"All experiments completed. Summary saved to {summary_path}")
        
        return all_results
    
    def _cleanup_models(self):
        """Clean up loaded models to free memory"""
        for model_name, model in self.loaded_models.items():
            if isinstance(model, LocalModelHandler):
                self.logger.log_info(f"Unloading model: {model_name}")
                model.unload_model()
        
        self.loaded_models.clear()


class MockLLMModel:
    """Mock model for testing without actual LLM"""
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def predict(self, text: str) -> str:
        return f"Mock prediction from {self.config.name}"


def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarking experiments")
    parser.add_argument(
        '--config-dir',
        default='./config',
        help='Directory containing configuration files'
    )
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Directory for output results'
    )
    parser.add_argument(
        '--experiment-name',
        help='Name for this experiment run'
    )
    parser.add_argument(
        '--model',
        help='Run only specific model (by name)'
    )
    parser.add_argument(
        '--dataset',
        help='Run only specific dataset (by name)'
    )
    parser.add_argument(
        '--enable-prompting',
        action='store_true',
        default=True,
        help='Enable prompting techniques (default: True)'
    )
    parser.add_argument(
        '--no-prompting',
        action='store_true',
        help='Disable prompting techniques'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig(config_dir=args.config_dir)
    config.load_configs()
    
    # Print summary
    print(config.summary())
    
    # Filter by specific model/dataset if requested
    if args.model:
        config.models = [m for m in config.models if m.name == args.model]
    if args.dataset:
        config.datasets = [d for d in config.datasets if d.name == args.dataset]
    
    # Determine prompting enable status
    enable_prompting = args.enable_prompting and not args.no_prompting
    
    # Initialize and run
    runner = BenchmarkRunner(
        config=config,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        enable_prompting=enable_prompting
    )
    
    # Print prompting summary if enabled
    if enable_prompting and runner.prompt_manager:
        print("\n" + runner.prompt_manager.summary())
    
    results = runner.run_all_experiments()
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(results['experiments'])}")
    
    successful = sum(1 for e in results['experiments'] if e['status'] == 'completed')
    failed = sum(1 for e in results['experiments'] if e['status'] == 'failed')
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print("\nResults:")
        for exp in results['experiments']:
            if exp['status'] == 'completed':
                metrics = exp.get('metrics', {})
                accuracy = metrics.get('accuracy', 'N/A')
                prompt_tech = exp.get('prompting_technique', 'baseline')
                print(f"  {exp['model']} on {exp['dataset']} [{prompt_tech}]: {accuracy}")


if __name__ == "__main__":
    main()
