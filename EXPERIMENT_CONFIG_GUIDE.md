# Experiment Configuration Guide

## Overview

The `config/experiments.yaml` file allows you to define specific experiment triplets (model + dataset + prompting technique) with fine-grained control over each experiment's parameters.

## New Features: Per-Experiment Overrides

### Sample Size Override (`num_samples`)

Control how many samples to use for a specific experiment, overriding the dataset's default setting.

```yaml
- id: quick_test
  model: phi-3-mini-4bit
  dataset: sst2
  prompting_technique: classification_simple
  num_samples: 10 # Test on only 10 samples instead of full dataset
  enabled: true
```

**Use Cases:**

- **Quick validation:** Test infrastructure with `num_samples: 10`
- **Development testing:** Use `num_samples: 100` for faster iteration
- **Gradual scaling:** Test on 10, 100, 1000 samples before full dataset
- **Cost control:** Limit samples when using API-based models

### Max Tokens Override (`max_tokens`)

Control the maximum tokens generated per sample, overriding the model's default setting.

```yaml
- id: efficient_classification
  model: llama-3-8b
  dataset: sentiment_analysis
  prompting_technique: classification_simple
  max_tokens: 5 # Only need a few tokens for label output
  enabled: true
```

**Use Cases:**

- **Classification tasks:** Use `max_tokens: 5-10` for single-word outputs
- **QA tasks:** Use `max_tokens: 50-100` for short answers
- **Reasoning tasks:** Use `max_tokens: 500-1000` for chain-of-thought
- **Cost optimization:** Reduce tokens to lower API costs

## Complete Example

```yaml
experiments:
  # Quick validation test
  - id: phi3_validation
    model: phi-3-mini-4bit
    dataset: custom_classification
    prompting_technique: classification_simple
    description: "Quick validation with minimal resources"
    enabled: true
    num_samples: 10 # Only test 10 samples
    max_tokens: 20 # Limit generation

  # Development testing
  - id: phi3_dev_test
    model: phi-3-mini-4bit
    dataset: sst2
    prompting_technique: classification_simple
    description: "Development testing on subset"
    enabled: true
    num_samples: 100 # Test on 100 samples
    max_tokens: 10 # Classification needs few tokens

  # Full evaluation
  - id: phi3_full_eval
    model: phi-3-mini-4bit
    dataset: sst2
    prompting_technique: classification_simple
    description: "Full dataset evaluation"
    enabled: true
    # num_samples not specified = use full dataset
    # max_tokens not specified = use model default

  # Reasoning task
  - id: phi3_reasoning
    model: phi-3-mini-4bit
    dataset: gsm8k
    prompting_technique: cot
    description: "Math reasoning with chain-of-thought"
    enabled: true
    num_samples: 50 # Test subset first
    max_tokens: 500 # Allow detailed reasoning
```

## Override Priority

Configuration values are applied in this order (later overrides earlier):

1. **Model/Dataset config files** (`models.yaml`, `datasets.yaml`)
2. **Global experiment settings** (`experiments.yaml` → `global_settings`)
3. **Per-experiment overrides** (`experiments.yaml` → individual experiment)

Example:

```yaml
# models.yaml
- name: phi-3-mini-4bit
  max_tokens: 512 # Default for all experiments

# experiments.yaml
- id: my_experiment
  model: phi-3-mini-4bit
  max_tokens: 50 # Overrides the 512 default for THIS experiment only
```

## Common Patterns

### 1. Progressive Testing

```yaml
# Test with increasing sample sizes
- id: test_10
  num_samples: 10
  enabled: true

- id: test_100
  num_samples: 100
  enabled: false # Enable after test_10 passes

- id: test_full
  # num_samples: null = full dataset
  enabled: false # Enable after test_100 passes
```

### 2. Token Optimization by Task Type

```yaml
# Classification: minimal tokens
- id: sentiment_classification
  max_tokens: 5

# QA: short answers
- id: question_answering
  max_tokens: 100

# Generation: longer output
- id: text_generation
  max_tokens: 500

# Reasoning: extensive output
- id: math_reasoning
  max_tokens: 1000
```

### 3. Cost-Performance Trade-offs

```yaml
# Fast iteration (cheaper, less accurate)
- id: fast_iteration
  num_samples: 50
  max_tokens: 20
  enabled: true

# Full evaluation (expensive, more accurate)
- id: full_evaluation
  # Use defaults - full dataset, full tokens
  enabled: false
```

## Best Practices

1. **Start Small:** Always test with `num_samples: 10` first to validate your setup
2. **Be Specific:** Use descriptive experiment IDs that indicate the override
3. **Document Why:** Add descriptions explaining why you chose specific values
4. **Gradual Scaling:** Test 10 → 100 → 1000 samples before full evaluation
5. **Task-Appropriate Tokens:** Match max_tokens to the task (classification vs generation)

## Running Experiments

```bash
# Run all enabled experiments with overrides
python experiments/run_benchmark.py

# Run specific experiment
python experiments/run_benchmark.py --experiment-id phi3_validation

# Check configuration before running
python experiments/run_benchmark.py --dry-run
```

## Troubleshooting

**Q: My override isn't being applied?**

- Check experiment is `enabled: true`
- Verify model/dataset names match config files
- Look for override confirmation in logs: "Overriding num_samples to X"

**Q: Should I override at experiment level or model level?**

- **Model level:** If all experiments with that model need the same setting
- **Experiment level:** If only specific experiments need different settings

**Q: What happens if I set num_samples larger than the dataset?**

- The system will use min(num_samples, dataset_size)
- No error, just uses all available samples

## Example Output

When overrides are applied, you'll see log messages:

```
2026-02-24 10:15:30 - INFO - Overriding num_samples to 10 for experiment phi3_validation
2026-02-24 10:15:30 - INFO - Overriding max_tokens to 20 for experiment phi3_validation
2026-02-24 10:15:31 - INFO - Starting experiment: phi-3-mini-4bit on custom_classification
```

## See Also

- [experiments.yaml](config/experiments.yaml) - Example configurations
- [DIAGNOSIS_AND_FIXES.md](DIAGNOSIS_AND_FIXES.md) - Troubleshooting guide
- [README.md](README.md) - Main documentation
