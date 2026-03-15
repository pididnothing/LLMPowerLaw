"""
Metrics Calculation Utilities
Calculates various evaluation metrics for LLM benchmarking
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import re
from fractions import Fraction


class MetricsCalculator:
    """Calculate evaluation metrics for different task types"""
    
    @staticmethod
    def classification_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate classification metrics
        
        Args:
            predictions: List of predictions with 'prediction' and 'true_label' keys
            
        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {}
        
        # Extract predictions and labels
        pred_labels = [str(p['prediction']).strip().lower() for p in predictions]
        true_labels = [str(p['true_label']).strip().lower() for p in predictions]
        
        # Accuracy
        correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
        accuracy = correct / len(predictions)
        
        # Get unique labels
        unique_labels = sorted(set(true_labels + pred_labels))
        
        # Calculate per-class metrics
        class_metrics = {}
        for label in unique_labels:
            tp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t == label)
            fp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t != label)
            fn = sum(1 for p, t in zip(pred_labels, true_labels) if p != label and t == label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Macro averages
        macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'num_samples': len(predictions),
            'num_correct': correct,
            'class_metrics': class_metrics
        }
    
    @staticmethod
    def qa_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate QA metrics (exact match, F1)
        
        Args:
            predictions: List of predictions with 'prediction' and 'true_label' keys
            
        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {}
        
        exact_matches = []
        f1_scores = []
        
        for pred_dict in predictions:
            prediction = str(pred_dict['prediction']).strip().lower()
            true_answer = str(pred_dict['true_label']).strip().lower()
            
            # Exact match
            exact_match = int(prediction == true_answer)
            exact_matches.append(exact_match)
            
            # Token F1
            pred_tokens = prediction.split()
            true_tokens = true_answer.split()
            
            if len(pred_tokens) == 0 or len(true_tokens) == 0:
                f1_scores.append(0.0)
                continue
            
            common = Counter(pred_tokens) & Counter(true_tokens)
            num_common = sum(common.values())
            
            if num_common == 0:
                f1_scores.append(0.0)
                continue
            
            precision = num_common / len(pred_tokens)
            recall = num_common / len(true_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        
        return {
            'exact_match': np.mean(exact_matches),
            'f1': np.mean(f1_scores),
            'num_samples': len(predictions)
        }
    
    @staticmethod
    def generation_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate generation metrics (BLEU-inspired simple score)
        
        Args:
            predictions: List of predictions with 'prediction' and 'true_label' keys
            
        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {}
        
        scores = []
        
        for pred_dict in predictions:
            prediction = str(pred_dict['prediction']).strip().lower()
            reference = str(pred_dict['true_label']).strip().lower()
            
            # Simple n-gram overlap score
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                scores.append(0.0)
                continue
            
            # Unigram precision
            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())
            precision = num_common / len(pred_tokens)
            
            # Recall
            recall = num_common / len(ref_tokens)
            
            # F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            scores.append(f1)
        
        return {
            'average_score': np.mean(scores),
            'num_samples': len(predictions)
        }

    @staticmethod
    def _extract_numeric_value(text: str) -> Optional[float]:
        """Extract a final numeric value from text when possible."""
        if text is None:
            return None

        s = str(text).strip().lower()
        if not s:
            return None

        s = s.replace(',', '')
        matches = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', s)
        if not matches:
            return None

        token = matches[-1]
        try:
            if '/' in token and '.' not in token:
                return float(Fraction(token))
            return float(token)
        except Exception:
            return None

    @staticmethod
    def reasoning_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for reasoning tasks (e.g., GSM8K)."""
        if not predictions:
            return {}

        exact_matches = []
        numeric_exact_matches = []
        abs_errors = []
        parseable_pairs = 0

        for pred_dict in predictions:
            prediction = str(pred_dict['prediction']).strip().lower()
            true_answer = str(pred_dict['true_label']).strip().lower()

            exact_matches.append(int(prediction == true_answer))

            pred_num = MetricsCalculator._extract_numeric_value(prediction)
            true_num = MetricsCalculator._extract_numeric_value(true_answer)
            if pred_num is not None and true_num is not None:
                parseable_pairs += 1
                numeric_exact = int(abs(pred_num - true_num) < 1e-6)
                numeric_exact_matches.append(numeric_exact)
                abs_errors.append(abs(pred_num - true_num))

        parse_rate = parseable_pairs / len(predictions)

        return {
            'exact_match': float(np.mean(exact_matches)) if exact_matches else 0.0,
            'numeric_exact_match': float(np.mean(numeric_exact_matches)) if numeric_exact_matches else 0.0,
            'numeric_parse_rate': parse_rate,
            'mean_absolute_error': float(np.mean(abs_errors)) if abs_errors else None,
            'num_samples': len(predictions)
        }
    
    @staticmethod
    def basic_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate basic metrics when task type is unknown
        
        Args:
            predictions: List of predictions
            
        Returns:
            Dictionary of basic statistics
        """
        return {
            'num_samples': len(predictions),
            'num_predictions': sum(1 for p in predictions if 'prediction' in p),
            'num_errors': sum(1 for p in predictions if 'error' in p)
        }
    
    @staticmethod
    def calculate_confidence_interval(
        values: List[float],
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence interval for a list of values
        
        Args:
            values: List of numeric values
            confidence: Confidence level (default 0.95)
            
        Returns:
            Dictionary with mean, std, and confidence interval
        """
        if not values:
            return {}
        
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0
        
        # Using normal approximation
        from scipy import stats
        ci = stats.t.interval(
            confidence,
            len(values) - 1,
            loc=mean,
            scale=std / np.sqrt(len(values))
        )
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'confidence': confidence
        }


def compare_models(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple model results
    
    Args:
        results: List of experiment results
        
    Returns:
        Comparison summary
    """
    comparison = {
        'models': [],
        'datasets': []
    }
    
    # Group by model
    models = {}
    for result in results:
        model_name = result.get('model')
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(result)
    
    # Calculate aggregate metrics for each model
    for model_name, model_results in models.items():
        accuracies = [
            r.get('metrics', {}).get('accuracy', 0)
            for r in model_results
            if 'metrics' in r
        ]
        
        comparison['models'].append({
            'name': model_name,
            'num_experiments': len(model_results),
            'avg_accuracy': np.mean(accuracies) if accuracies else 0,
            'std_accuracy': np.std(accuracies) if accuracies else 0
        })
    
    return comparison


if __name__ == "__main__":
    # Test metrics calculator
    calc = MetricsCalculator()
    
    # Test classification metrics
    test_predictions = [
        {'prediction': 'positive', 'true_label': 'positive'},
        {'prediction': 'negative', 'true_label': 'negative'},
        {'prediction': 'positive', 'true_label': 'negative'},
        {'prediction': 'neutral', 'true_label': 'neutral'},
    ]
    
    metrics = calc.classification_metrics(test_predictions)
    print("Classification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    
    # Test QA metrics
    qa_predictions = [
        {'prediction': 'Paris', 'true_label': 'Paris'},
        {'prediction': 'London', 'true_label': 'Paris'},
    ]
    
    qa_metrics = calc.qa_metrics(qa_predictions)
    print("\nQA Metrics:")
    print(f"  Exact Match: {qa_metrics['exact_match']:.3f}")
    print(f"  F1: {qa_metrics['f1']:.3f}")
