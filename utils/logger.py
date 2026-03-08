"""
Experiment Logging Utilities
Handles logging of experiment progress and results
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """Logger for experiment tracking and results"""
    
    def __init__(
        self,
        output_dir: str = "./results",
        experiment_name: str = "experiment",
        log_level: int = logging.INFO
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        
        # Setup file logger
        log_file = self.output_dir / f"{experiment_name}.log"
        
        # Create logger
        self.logger = logging.getLogger(f"Experiment_{experiment_name}")
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_info(f"Initialized experiment logger: {experiment_name}")
        
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def save_experiment_results(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ):
        """Save experiment results to JSON file"""
        if filename is None:
            model_name = results.get('model', 'unknown')
            dataset_name = results.get('dataset', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{model_name}_{dataset_name}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        self.log_info(f"Results saved to {output_path}")
        
        return output_path
    
    def save_metrics_summary(
        self,
        metrics: Dict[str, Any],
        filename: Optional[str] = None
    ):
        """Save metrics summary"""
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        self.log_info(f"Metrics saved to {output_path}")
        
        return output_path


class TensorBoardLogger:
    """TensorBoard logger for experiment tracking"""
    
    def __init__(self, log_dir: str = "./results/tensorboard"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_value_dict: Dict[str, float], step: int):
        """Log multiple scalar values"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_value_dict, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.enabled:
            self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """Log hyperparameters and metrics"""
        if self.enabled:
            self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Close the writer"""
        if self.enabled:
            self.writer.close()


if __name__ == "__main__":
    # Test logger
    logger = ExperimentLogger(
        output_dir="./results",
        experiment_name="test_experiment"
    )
    
    logger.log_info("Testing logger")
    
    # Test saving results
    test_results = {
        'model': 'test_model',
        'dataset': 'test_dataset',
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.82
        }
    }
    
    logger.save_experiment_results(test_results)
    print("Logger test completed")
