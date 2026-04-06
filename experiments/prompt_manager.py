"""
Prompt Manager for Different Prompting Techniques
Handles both PromptBench built-in and custom prompting strategies
"""

import yaml
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from string import Template

from utils.generate_prompt_templates import build_inner_prompt, load_configs


@dataclass
class PromptTechnique:
    """Configuration for a prompting technique"""
    name: str
    type: str  # promptbench or custom
    description: str
    enabled: bool
    technique: Optional[str] = None  # For PromptBench techniques
    template: Optional[str] = None  # For custom techniques
    params: Dict[str, Any] = None
    fields: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.fields is None:
            self.fields = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTechnique':
        """Create PromptTechnique from dictionary"""
        return cls(
            name=data['name'],
            type=data.get('type', 'custom'),
            description=data.get('description', ''),
            enabled=data.get('enabled', True),
            technique=data.get('technique'),
            template=data.get('template'),
            params=data.get('params', {}),
            fields=data.get('fields', {})
        )


class PromptManager:
    """Manager for different prompting techniques"""
    
    def __init__(self, config_path: str = "./config/prompting_techniques.yaml"):
        self.config_path = Path(config_path)
        self.techniques: List[PromptTechnique] = []
        self.technique_combinations: List[Dict[str, Any]] = []
        self.global_settings: Dict[str, Any] = {}
        self.example_cache: Dict[str, List] = {}
        self.template_source_configs: Optional[Dict[str, Any]] = None
        
        if self.config_path.exists():
            self.load_config()
    
    def load_config(self):
        """Load prompting techniques configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.techniques = [
            PromptTechnique.from_dict(tech_data)
            for tech_data in config.get('prompting_techniques', [])
        ]
        
        self.technique_combinations = config.get('technique_combinations', [])
        self.global_settings = config.get('global_settings', {})

        # Also load generated prompt templates if available
        generated_path = self.config_path.parent / "prompting_techniques_generated.yaml"
        if generated_path.exists():
            with open(generated_path, 'r', encoding='utf-8') as f:
                gen_config = yaml.safe_load(f) or {}
            existing_names = {t.name for t in self.techniques}
            for tech_data in gen_config.get('prompting_techniques', []):
                if tech_data.get('name') not in existing_names:
                    self.techniques.append(PromptTechnique.from_dict(tech_data))
    
    def get_enabled_techniques(self) -> List[PromptTechnique]:
        """Get list of enabled techniques"""
        return [t for t in self.techniques if t.enabled]
    
    def get_technique_by_name(self, name: str) -> Optional[PromptTechnique]:
        """Get technique by name"""
        for technique in self.techniques:
            if technique.name == name:
                return technique
        return None
    
    def get_techniques_for_dataset(self, dataset_name: str) -> List[PromptTechnique]:
        """Get techniques configured for a specific dataset"""
        mapping = self.global_settings.get('dataset_technique_mapping', {})
        
        if dataset_name in mapping:
            technique_names = mapping[dataset_name]
            return [
                self.get_technique_by_name(name)
                for name in technique_names
                if self.get_technique_by_name(name) is not None
            ]
        
        # Return all enabled techniques if no specific mapping
        if self.global_settings.get('apply_to_all', False):
            return self.get_enabled_techniques()
        
        # Return default technique
        default_name = self.global_settings.get('default_technique', 'zero_shot')
        default = self.get_technique_by_name(default_name)
        return [default] if default else []
    
    def apply_technique(
        self,
        technique: PromptTechnique,
        input_text: str,
        dataset_name: str = None,
        task_type: str = None,
        examples: List[Dict[str, Any]] = None,
        output_mode: str = 'manual',
        **kwargs
    ) -> str:
        """
        Apply a prompting technique to input text
        
        Args:
            technique: The prompting technique to apply
            input_text: The input text/question
            dataset_name: Name of the dataset (optional)
            task_type: Type of task (optional)
            examples: Few-shot examples (optional)
            output_mode: 'manual' for full template text, 'content' for chat-body only
            **kwargs: Additional keyword arguments for template
            
        Returns:
            Formatted prompt string
        """
        if technique.type == 'promptbench':
            return self._apply_promptbench_technique(
                technique, input_text, examples, task_type
            )
        elif technique.type == 'custom':
            return self._apply_custom_technique(
                technique, input_text, task_type, examples, output_mode=output_mode, **kwargs
            )
        else:
            return input_text
    
    def _apply_promptbench_technique(
        self,
        technique: PromptTechnique,
        input_text: str,
        examples: List[Dict[str, Any]] = None,
        task_type: str = None
    ) -> str:
        """Apply PromptBench built-in technique"""
        tech_name = technique.technique
        params = technique.params
        
        if tech_name == 'zero_shot':
            # Simple zero-shot: just return the input
            return input_text
        
        elif tech_name == 'few_shot':
            # Few-shot: prepend examples
            if not examples:
                return input_text
            
            num_examples = params.get('num_examples', 3)
            separator = params.get('example_separator', '\n\n')
            
            # Select examples
            selected_examples = self._select_few_shot_examples(
                examples, num_examples
            )
            
            # Format examples
            example_text = separator.join([
                self._format_example(ex) for ex in selected_examples
            ])
            
            return f"{example_text}{separator}{input_text}"
        
        elif tech_name == 'cot':
            # Chain-of-thought: add reasoning trigger
            cot_trigger = params.get('cot_trigger', "Let's think step by step.")
            return f"{input_text}\n\n{cot_trigger}"
        
        elif tech_name == 'role':
            # Role prompting: prepend role description
            role = params.get('role', 'You are a helpful AI assistant.')
            return f"{role}\n\n{input_text}"
        
        elif tech_name == 'emotion':
            # Emotion prompting: add emotional phrase
            emotion = params.get('emotion_phrase', 'This is very important.')
            return f"{input_text}\n\n{emotion}"
        
        elif tech_name == 'expert':
            # Expert prompting: invoke expert persona
            expert_type = params.get('expert_type', 'expert')
            return f"As a {expert_type}, {input_text}"
        
        else:
            # Unknown technique, return input as-is
            return input_text
    
    def _apply_custom_technique(
        self,
        technique: PromptTechnique,
        input_text: str,
        task_type: str = None,
        examples: List[Dict[str, Any]] = None,
        output_mode: str = 'manual',
        **kwargs
    ) -> str:
        """Apply custom prompting technique"""
        if output_mode == 'content':
            content_template = self._get_content_template(technique)
            if content_template:
                return self._render_template_string(
                    content_template,
                    input_text,
                    task_type=task_type,
                    examples=examples,
                    fields=technique.fields,
                    **kwargs
                )

        if not technique.template:
            return input_text

        return self._render_template_string(
            technique.template,
            input_text,
            task_type=task_type,
            examples=examples,
            fields=technique.fields,
            **kwargs
        )

    def _render_template_string(
        self,
        template_string: str,
        input_text: str,
        task_type: str = None,
        examples: List[Dict[str, Any]] = None,
        fields: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Render a template string with runtime values."""
        template_vars = {
            'input': input_text,
            'task': task_type or 'task',
            **(fields or {}),
            **kwargs
        }

        # Handle few-shot examples in custom templates
        if examples and ('$examples' in template_string or '${examples}' in template_string):
            num_examples = (fields or {}).get('num_examples', 3)
            selected_examples = self._select_few_shot_examples(examples, num_examples)
            example_text = '\n\n'.join([
                self._format_example(ex) for ex in selected_examples
            ])
            template_vars['examples'] = example_text

        # Format template
        try:
            template = Template(template_string)
            formatted_prompt = template.safe_substitute(template_vars)
            return formatted_prompt
        except Exception as e:
            print(f"Error formatting template: {e}")
            return input_text

    def _get_content_template(self, technique: PromptTechnique) -> Optional[str]:
        """Return a chat-body-only prompt template for HF chat templating."""
        content_template = technique.fields.get('content_template')
        if content_template:
            return content_template

        dataset_key = technique.fields.get('dataset')
        technique_key = technique.fields.get('technique')
        if not dataset_key or not technique_key:
            return None

        source_cfg = self._load_template_source_configs()
        try:
            return build_inner_prompt(
                technique_key=technique_key,
                dataset_key=dataset_key,
                technique_templates=source_cfg['technique_templates'],
                dataset_instructions=source_cfg['dataset_instructions'],
                few_shot_samples=source_cfg['few_shot_samples'],
                domain_experts=source_cfg['domain_experts'],
            )
        except Exception:
            return None

    def _load_template_source_configs(self) -> Dict[str, Any]:
        """Load generator source configs used to reconstruct chat-body prompts."""
        if self.template_source_configs is None:
            self.template_source_configs = load_configs(self.config_path.parent)
        return self.template_source_configs
    
    def _select_few_shot_examples(
        self,
        examples: List[Dict[str, Any]],
        num_examples: int
    ) -> List[Dict[str, Any]]:
        """Select few-shot examples based on strategy"""
        if not examples:
            return []
        
        strategy = self.global_settings.get('few_shot_strategy', 'random')
        
        # Ensure we don't request more examples than available
        num_examples = min(num_examples, len(examples))
        
        if strategy == 'random':
            return random.sample(examples, num_examples)
        
        elif strategy == 'first':
            return examples[:num_examples]
        
        elif strategy == 'diverse':
            # TODO: Implement diverse selection based on labels
            return random.sample(examples, num_examples)
        
        else:
            return examples[:num_examples]
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a single example for few-shot prompting"""
        # Try to extract input and output from example
        input_text = example.get('input') or example.get('text') or example.get('question') or example.get('content')
        output_text = example.get('output') or example.get('label') or example.get('answer')
        
        # Handle PromptBench QQP format specifically
        if isinstance(input_text, dict) and 'content' in input_text:
            input_text = input_text['content']
        
        if input_text and output_text is not None:  # Allow 0 as output
            # Convert output to string
            output_str = str(output_text)
            return f"{input_text}\nLabel: {output_str}"
        elif input_text:
            return str(input_text)
        else:
            # Last resort: try to find any meaningful text
            for key in ['sentence', 'premise', 'hypothesis']:
                if key in example:
                    return str(example[key])
            # If still nothing, return empty string instead of dict representation
            return ""
    
    def apply_technique_combination(
        self,
        combination_name: str,
        input_text: str,
        **kwargs
    ) -> str:
        """Apply multiple techniques in combination"""
        combination = None
        for comb in self.technique_combinations:
            if comb['name'] == combination_name:
                combination = comb
                break
        
        if not combination:
            return input_text
        
        # Apply each technique in sequence
        result = input_text
        for tech_name in combination['techniques']:
            technique = self.get_technique_by_name(tech_name)
            if technique:
                result = self.apply_technique(technique, result, **kwargs)
        
        return result
    
    def summary(self) -> str:
        """Generate a summary of prompting configuration"""
        enabled_techniques = self.get_enabled_techniques()
        
        summary = []
        summary.append("=" * 60)
        summary.append("PROMPTING TECHNIQUES CONFIGURATION")
        summary.append("=" * 60)
        
        summary.append(f"\nEnabled Techniques ({len(enabled_techniques)}):")
        for tech in enabled_techniques:
            tech_type = f"[{tech.type}]"
            summary.append(f"  - {tech.name} {tech_type}")
            summary.append(f"    {tech.description}")
        
        # Dataset mappings
        mappings = self.global_settings.get('dataset_technique_mapping', {})
        if mappings:
            summary.append(f"\nDataset-Technique Mappings:")
            for dataset, techniques in mappings.items():
                summary.append(f"  - {dataset}: {', '.join(techniques)}")
        
        # Combinations
        enabled_combos = [c for c in self.technique_combinations if c.get('enabled')]
        if enabled_combos:
            summary.append(f"\nEnabled Combinations ({len(enabled_combos)}):")
            for combo in enabled_combos:
                summary.append(f"  - {combo['name']}: {' + '.join(combo['techniques'])}")
        
        summary.append("=" * 60)
        return "\n".join(summary)


def demonstrate_prompting():
    """Demonstrate different prompting techniques"""
    manager = PromptManager()
    
    # Example input
    input_text = "The movie was absolutely fantastic and I loved every minute of it."
    
    print("ORIGINAL INPUT:")
    print(input_text)
    print("\n" + "="*60 + "\n")
    
    # Example few-shot examples
    examples = [
        {"input": "This film is great!", "output": "positive"},
        {"input": "I hated this movie.", "output": "negative"},
        {"input": "It was okay.", "output": "neutral"},
    ]
    
    # Try each enabled technique
    for technique in manager.get_enabled_techniques():
        print(f"TECHNIQUE: {technique.name}")
        print(f"Type: {technique.type}")
        print(f"Description: {technique.description}")
        print("-" * 60)
        
        prompt = manager.apply_technique(
            technique,
            input_text,
            task_type="sentiment_classification",
            examples=examples
        )
        
        print(prompt)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Test the prompt manager
    manager = PromptManager()
    print(manager.summary())
    print("\n")
    
    # Demonstrate prompting
    demonstrate_prompting()
