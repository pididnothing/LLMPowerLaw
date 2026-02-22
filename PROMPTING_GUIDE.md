# Prompting Techniques Guide

This guide explains how to use different prompting techniques with your LLM benchmarking experiments.

## Table of Contents

- [Overview](#overview)
- [Built-in PromptBench Techniques](#built-in-promptbench-techniques)
- [Custom Prompting Techniques](#custom-prompting-techniques)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Overview

Prompting techniques can significantly affect LLM performance. This framework supports:

1. **PromptBench Built-in Techniques**: Pre-implemented prompting strategies
2. **Custom Techniques**: Your own prompt templates
3. **Technique Combinations**: Apply multiple techniques together

## Built-in PromptBench Techniques

### Zero-Shot

Direct question without examples.

```yaml
- name: zero_shot
  type: promptbench
  technique: zero_shot
  enabled: true
```

**Example:**

```
Input: The movie was amazing!
```

### Few-Shot

Provide examples before the question.

```yaml
- name: few_shot
  type: promptbench
  technique: few_shot
  enabled: true
  params:
    num_examples: 3
```

**Example:**

```
Input: This film is great!
Output: positive

Input: I hated this movie.
Output: negative

Input: The movie was amazing!
```

### Chain-of-Thought (CoT)

Add reasoning trigger to encourage step-by-step thinking.

```yaml
- name: cot
  type: promptbench
  technique: cot
  enabled: true
  params:
    cot_trigger: "Let's think step by step."
```

**Example:**

```
Input: The movie was amazing!

Let's think step by step.
```

### Role Prompting

Assign a role to the model.

```yaml
- name: role_prompting
  type: promptbench
  technique: role
  enabled: true
  params:
    role: "You are an expert sentiment analyst."
```

**Example:**

```
You are an expert sentiment analyst.

The movie was amazing!
```

### Other Built-in Techniques

- **Emotion Prompting**: Add emotional appeal
- **Expert Prompting**: Invoke expert persona

## Custom Prompting Techniques

Create your own prompt templates using placeholders.

### Basic Custom Template

```yaml
- name: custom_instructive
  type: custom
  enabled: true
  template: |
    Instructions: {instructions}

    Task: {task}

    Input: {input}

    Output:
  fields:
    instructions: "Carefully analyze the input."
    task: "classification"
```

### Available Placeholders

- `{input}`: The input text/question
- `{task}`: Task type from dataset config
- `{examples}`: Few-shot examples (if provided)
- Any custom field you define in `fields:`

### Example Custom Templates

#### Structured Output

```yaml
- name: custom_structured
  type: custom
  template: |
    ### Task Description ###
    {task_description}

    ### Input ###
    {input}

    ### Expected Output Format ###
    {output_format}

    ### Your Response ###
  fields:
    task_description: "Analyze the sentiment"
    output_format: "Single word: positive, negative, or neutral"
```

#### Persona-Based

```yaml
- name: persona_prompting
  type: custom
  template: |
    You are a {persona} with expertise in {domain}.

    Task: {input}

    Provide a {output_style} response.
  fields:
    persona: "professional analyst"
    domain: "text analysis"
    output_style: "concise and accurate"
```

## Configuration

### Global Settings

```yaml
global_settings:
  # Default technique if none specified
  default_technique: zero_shot

  # Whether to apply prompting techniques to all datasets
  apply_to_all: false

  # Per-dataset technique mapping
  dataset_technique_mapping:
    sst2: [zero_shot, few_shot, cot]
    mnli: [zero_shot, few_shot]
    custom_data: [zero_shot, custom_instructive]

  # Few-shot example selection
  few_shot_strategy: random # random, similar, diverse

  # Cache prompts
  cache_prompts: true

  # Maximum prompt length (tokens)
  max_prompt_length: 2048
```

### Per-Dataset Configuration

Map specific techniques to datasets:

```yaml
dataset_technique_mapping:
  sst2: [zero_shot, few_shot, cot] # Run 3 experiments
  mnli: [zero_shot] # Run 1 experiment
  custom_task: [custom_instructive] # Run 1 experiment
```

## Usage Examples

### Run with Prompting Techniques

```bash
# Run all enabled techniques
python experiments/run_benchmark.py

# Disable prompting techniques
python experiments/run_benchmark.py --no-prompting

# Run specific model and dataset
python experiments/run_benchmark.py --model gpt-3.5-turbo --dataset sst2
```

### Programmatic Usage

```python
from experiments import BenchmarkRunner, ExperimentConfig
from experiments.prompt_manager import PromptManager

# Load configurations
config = ExperimentConfig()
config.load_configs()

# Initialize runner with prompting enabled
runner = BenchmarkRunner(
    config=config,
    enable_prompting=True
)

# Run experiments
results = runner.run_all_experiments()
```

### Test Prompting Techniques

```python
from experiments.prompt_manager import PromptManager

# Initialize manager
manager = PromptManager()

# Get a technique
technique = manager.get_technique_by_name('few_shot')

# Apply to input
prompt = manager.apply_technique(
    technique=technique,
    input_text="This movie was great!",
    task_type="sentiment_analysis",
    examples=[
        {"input": "Loved it!", "output": "positive"},
        {"input": "Hated it!", "output": "negative"}
    ]
)

print(prompt)
```

## Best Practices

### 1. Start Simple

Begin with zero-shot, then try few-shot and CoT:

```yaml
dataset_technique_mapping:
  my_dataset: [zero_shot, few_shot, cot]
```

### 2. Few-Shot Guidelines

- Use 3-5 examples for most tasks
- Ensure examples cover all classes
- Use diverse, representative examples

### 3. Custom Template Tips

- Keep templates clear and structured
- Test templates manually first
- Use descriptive field names
- Consider token limits

### 4. Performance Considerations

- More complex prompts = more tokens = higher cost
- Cache prompts when possible
- Test on small samples first

### 5. Task-Specific Techniques

**Classification:**

- Zero-shot, Few-shot, Role prompting

**Question Answering:**

- CoT, Few-shot, Expert prompting

**Generation:**

- Role prompting, Persona prompting, Custom structured

## Technique Combinations

Combine multiple techniques:

```yaml
technique_combinations:
  - name: role_plus_cot
    techniques:
      - role_prompting
      - cot
    enabled: true
```

Apply in code:

```python
prompt = manager.apply_technique_combination(
    combination_name='role_plus_cot',
    input_text="The movie was amazing!",
    task_type="sentiment"
)
```

## Results Analysis

Results include prompting technique info:

```json
{
  "model": "gpt-3.5-turbo",
  "dataset": "sst2",
  "prompting_technique": "few_shot",
  "metrics": {
    "accuracy": 0.87
  }
}
```

Compare techniques:

```python
import pandas as pd

# Load results
results_df = pd.read_json('results/experiment_summary.json')

# Group by technique
technique_comparison = results_df.groupby('prompting_technique')['accuracy'].mean()
print(technique_comparison)
```

## Troubleshooting

**Prompts too long:**

- Reduce `num_examples` in few-shot
- Shorten custom templates
- Increase `max_prompt_length`

**Inconsistent results:**

- Fix random seed in config
- Use temperature=0.0
- Try multiple runs

**Poor performance:**

- Test different techniques
- Adjust template wording
- Check example quality

## References

- PromptBench Documentation
- Chain-of-Thought Prompting: Wei et al. (2022)
- Few-Shot Learning: Brown et al. (2020)

---

For more information, see the [README](README.md) or [Local Models Guide](LOCAL_MODELS.md).
