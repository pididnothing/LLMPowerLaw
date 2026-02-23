# LLM Power Law - Multi-Model Benchmarking Framework

A comprehensive framework for conducting LLM performance experiments using **PromptBench** with support for multiple custom benchmarking datasets, models, **prompting techniques**, and **local model execution**.

## ✨ New Features

### 🎯 Prompting Techniques Support

- **Built-in PromptBench techniques**: Zero-shot, Few-shot, Chain-of-Thought, Role prompting, and more
- **Custom prompting templates**: Create your own prompting strategies
- **Automatic comparison**: Test multiple prompting techniques across models and datasets
- **See [Prompting Guide](PROMPTING_GUIDE.md)** for details

### 💻 Local Model Execution

- **Run models locally** with full privacy and no API costs
- **Multiple formats**: HuggingFace, GGUF/llama.cpp, vLLM
- **Memory optimization**: 8-bit/4-bit quantization support
- **PromptBench compatible**: Full support for local LLM execution
- **See [Local Models Guide](LOCAL_MODELS.md)** for details

## Features (API + Local)

│ ├── datasets.yaml # Dataset configurations
│ └── prompting_techniques.yaml # Prompting strategies
├── datasets/ # Dataset handling
│ ├── **init**.py
│ ├── custom_loader.py # Custom dataset loader
│ └── data/ # Your custom dataset files
├── experiments/ # Experiment execution
│ ├── **init**.py
│ ├── experiment_config.py # Configuration manager
│ ├── prompt_manager.py # Prompting techniques handler
│ ├── local_model_handler.py # Local model loader
│ └── run_benchmark.py # Main benchmark runner
├── utils/ # Utilities
│ ├── **init**.py
│ ├── logger.py # Logging utilities
│ └── metrics.py # Metrics calculation
├── results/ # Experiment results
├── notebooks/ # Jupyter notebooks for analysis
├── requirements.txt # Python dependencies
├── .env.example # Environment variables template
├── README.md # This file
├── PROMPTING_GUIDE.md # Prompting techniques guide
├── LOCAL_MODELS.md # Local models guide
└── QUICKSTART.md # Quick start guidconfigurations
├── datasets/ # Dataset handling
│ ├── **init**.py
│ ├── custom_loader.py # Custom dataset loader
│ └── data/ # Your custom dataset files
├── experiments/ # Experiment execution
│ ├── **init**.py
│ ├── experiment_config.py # Configuration manager
│ └── run_benchmark.py # Main benchmark runner
├── utils/ # Utilities
│ ├── **init**.py
│ ├── logger.py # Logging utilities
│ └── metrics.py # Metrics calculation
├── results/ # Experiment results
├── notebooks/ # Jupyter notebooks for analysis
├── requirements.txt # Python dependencies
├── .env.example # Environment variables template
└── README.md # This file

````

## 🎯 Prompting Techniques

Test how different prompting strategies affect model performance:

### Built-in PromptBench Techniques
- **Zero-Shot**: Direct questions without examples
- **Few-Shot**: Learn from examples
- **Chain-of-Thought (CoT)**: Step-by-step reasoning
- **Role Prompting**: Assign expert personas
- **And more...**

### Custom Techniques
Create your own prompting templates:

```yaml
- name: custom_instructive
  type: custom
  template: |
    Instructions: {instructions}
    Task: {task}
    Input: {input}
    Output:
````

**📚 See [PROMPTING_GUIDE.md](PROMPTING_GUIDE.md) for complete documentation**

## 💻 Local Model Support

### Run LLMs Locally - PromptBench Compatible!

**Yes! PromptBench fully supports local LLM execution.** This framework extends that support with optimizations:

#### Supported Formats

- **HuggingFace**: Direct from HF Hub or local files
- **GGUF**: Quantized models for llama.cpp
- **vLLM**: High-performance serving

#### Memory Optimization

- **8-bit quantization**: ~50% memory reduction
- **4-bit quantization**: ~75% memory reduction
- **GGUF quantized**: Run 7B models on 4GB RAM

#### Example Configuration

```yaml
- name: llama-2-7b-local
  provider: huggingface_local
  model_id: meta-llama/Llama-2-7b-chat-hf
  load_in_8bit: true # Reduce memory usage
  device: auto # Auto-detect GPU/CPU
  enabled: true
```

**📚 See [LOCAL_MODELS.md](LOCAL_MODELS.md) for complete documentation**

## Installation

> **⚠️ Windows Users (Python 3.13)**: If you encounter `sentencepiece` build errors, use the automated installation script instead:
>
> ```bash
> python install_windows.py
> # OR
> install_windows.bat
> ```
>
> This handles a known compatibility issue with `sentencepiece` on Windows with Python 3.13.

### 1. Clone or Navigate to the Project

```bash
cd d:\Projects\LLMPowerLaw
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

**Option A: Automated (Recommended for Windows)**

```bash
# Python script with detailed output
python install_windows.py

# OR batch file for quick installation
install_windows.bat
```

**Option B: Manual Installation**

```bash
# Standard installation
pip install -r requirements.txt

# If you get sentencepiece errors on Windows:
pip install sentencepiece>=0.2.0
pip install promptbench --no-deps
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# HUGGINGFACE_TOKEN=your_token_here
```

## Quick Start

### 1. Configure Models

Edit `config/models.yaml` to enable/disable models:

```yaml
models:
  - name: gpt-3.5-turbo
    provider: openai
    model_id: gpt-3.5-turbo
    max_tokens: 512
    temperature: 0.0
    enabled: true # Set to true to enable
```

### 2. Configure Datasets

Edit `config/datasets.yaml` to enable/disable datasets:

```yaml
datasets:
  - name: sst2
    type: promptbench
    dataset_name: sst-2
    task_type: classification
    num_samples: 1000
    enabled: true # Set to true to enable
```

### 3. Create Example Datasets (Optional)

```bash
python -m datasets.custom_loader
```

This creates example custom datasets in `datasets/data/`.

### 4. Run Experiments

```bash
# Run all enabled experiments
python experiments/run_benchmark.py

# Run specific model
python experiments/run_benchmark.py --model gpt-3.5-turbo

# Run specific dataset
python experiments/run_benchmark.py --dataset sst2

# Custom output directory
python experiments/run_benchmark.py --output-dir ./my_results

# Named experiment
python experiments/run_benchmark.py --experiment-name my_experiment_v1
```

### 5. Run with Prompting Techniques

```bash
# Run with enabled prompting techniques
python experiments/run_benchmark.py

# Disable prompting (baseline only)
python experiments/run_benchmark.py --no-prompting
```

Results will include prompting technique used:

```
gpt-3.5-turbo on sst2 [zero_shot]: 0.85
gpt-3.5-turbo on sst2 [few_shot]: 0.88
gpt-3.5-turbo on sst2 [cot]: 0.87
```

### 6. Run with Local Models

```bash
# First, enable a local model in config/models.yaml
# Then run:
python experiments/run_benchmark.py --model llama-2-7b-local

# Or run all local models
python experiments/run_benchmark.py
```

### 7. View Results

Results are saved in the `results/` directory:

- Individual experiment results: `{experiment_name}_{model}_{dataset}_{timestamp}.json`
- Experiment summary: `{experiment_name}_summary.json`
- Logs: `{experiment_name}.log`

## 🚀 Advanced Usage

### Using Prompting Techniques

Configure in `config/prompting_techniques.yaml`:

```yaml
prompting_techniques:
  - name: few_shot
    type: promptbench
    technique: few_shot
    enabled: true
    params:
      num_examples: 3

  - name: custom_cot
    type: custom
    enabled: true
    template: |
      Question: {input}
      Let's solve this step by step:
```

Map to datasets in `config/prompting_techniques.yaml`:

```yaml
global_settings:
  dataset_technique_mapping:
    sst2: [zero_shot, few_shot, cot] # Test 3 techniques
    mnli: [zero_shot] # Test 1 technique
    custom: [custom_cot, few_shot] # Test 2 techniques
```

**📚 Full guide:** [PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)

### Using Local Models

Configure in `config/models.yaml`:

```yaml
models:
  # For 16GB+ RAM with 8GB VRAM
  - name: mistral-7b-local
    provider: huggingface_local
    model_id: mistralai/Mistral-7B-Instruct-v0.2
    load_in_8bit: true # Reduce memory
    device: auto
    enabled: true

  # For lower memory (GGUF)
  - name: llama-2-7b-gguf
    provider: gguf
    model_id: TheBloke/Llama-2-7B-Chat-GGUF
    model_file: llama-2-7b-chat.Q4_K_M.gguf
    n_gpu_layers: 32 # Offload to GPU
    enabled: true
```

**📚 Full guide:** [LOCAL_MODELS.md](LOCAL_MODELS.md)

### 5. View Results

## Usage Guide

### Adding Custom Datasets

#### Format 1: JSONL (Recommended for large datasets)

```jsonl
{"input": "This is great!", "label": "positive"}
{"input": "This is terrible.", "label": "negative"}
```

#### Format 2: JSON

```json
{
  "data": [
    { "prompt": "Question here?", "answer": "Answer here" },
    { "prompt": "Another question?", "answer": "Another answer" }
  ]
}
```

#### Format 3: CSV

```csv
question,answer,context
What is ML?,Machine learning...,ML is a field...
```

#### Configure in `datasets.yaml`

```yaml
- name: my_custom_dataset
  type: custom
  file_path: "./datasets/data/my_data.jsonl"
  task_type: classification
  format: jsonl
  fields:
    text: "input" # Map your field names
    label: "label"
  num_samples: 500
  enabled: true
```

### Adding New Models

#### OpenAI Model

```yaml
- name: gpt-4
  provider: openai
  model_id: gpt-4
  max_tokens: 512
  temperature: 0.0
  enabled: true
```

#### HuggingFace Model

```yaml
- name: llama-2-7b
  provider: huggingface
  model_id: meta-llama/Llama-2-7b-chat-hf
  max_tokens: 512
  temperature: 0.0
  device: auto
  load_in_8bit: false
  enabled: true
```

### Programmatic Usage

```python
from experiments import ExperimentConfig, BenchmarkRunner

# Load configuration
config = ExperimentConfig(config_dir="./config")
config.load_configs()

# Print summary
print(config.summary())

# Run experiments
runner = BenchmarkRunner(
    config=config,
    output_dir="./results",
    experiment_name="my_experiment"
)

results = runner.run_all_experiments()
```

### Custom Metrics

The framework supports multiple task types with appropriate metrics:

- **Classification**: Accuracy, Precision, Recall, F1 (macro/per-class)
- **Question Answering**: Exact Match, Token F1
- **Generation**: N-gram overlap scores

Add custom metrics in `utils/metrics.py`:

```python
class MetricsCalculator:
    @staticmethod
    def my_custom_metric(predictions):
        # Your implementation
        return {'metric_name': value}
```

## Advanced Features

### Batch Size and Retries

Configure in `config/models.yaml`:

```yaml
global_settings:
  default_batch_size: 8
  use_cache: true
  cache_dir: "./cache"
  max_retries: 3
  retry_delay: 2
```

### TensorBoard Logging

View experiments in TensorBoard:

```bash
tensorboard --logdir results/tensorboard
```

### Analysis Notebook

Use the provided Jupyter notebook for analysis:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Supported Task Types

- `classification`: Text classification tasks
- `qa`: Question answering tasks
- `generation`: Text generation tasks
- `reasoning`: Reasoning tasks (e.g., math, logic)

## Configuration Reference

### Model Configuration Fields

- `name`: Unique identifier for the model
- `provider`: Model provider (openai, anthropic, huggingface, google)
- `model_id`: Provider-specific model identifier
- `max_tokens`: Maximum tokens in response
- `temperature`: Sampling temperature
- `enabled`: Whether to include in experiments

### Dataset Configuration Fields

- `name`: Unique identifier for the dataset
- `type`: Dataset source (promptbench, custom, huggingface)
- `task_type`: Type of task (classification, qa, generation, reasoning)
- `num_samples`: Number of samples to use (null for all)
- `enabled`: Whether to include in experiments
- `fields`: Field mapping for custom datasets

## Results Format

Each experiment produces a JSON file with:

```json
{
  "model": "gpt-3.5-turbo",
  "dataset": "sst2",
  "task_type": "classification",
  "status": "completed",
  "start_time": "2026-02-21T10:00:00",
  "end_time": "2026-02-21T10:05:00",
  "num_samples": 1000,
  "metrics": {
    "accuracy": 0.85,
    "macro_f1": 0.84,
    "macro_precision": 0.83,
    "macro_recall": 0.85
  },
  "predictions": [...]
}
```

## Troubleshooting

### API Key Issues

```bash
# Check if environment variables are loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### PromptBench Not Found

```bash
# Install PromptBench
pip install promptbench
```

### Memory Issues with Large Models

For large HuggingFace models, enable 8-bit loading:

```yaml
- name: llama-2-7b
  provider: huggingface
  model_id: meta-llama/Llama-2-7b-chat-hf
  load_in_8bit: true # Reduces memory usage
  device: auto
```

## Contributing

Feel free to extend this framework:

1. Add new model providers in `experiments/run_benchmark.py`
2. Add new metrics in `utils/metrics.py`
3. Add new dataset loaders in `datasets/custom_loader.py`
4. Share your configurations in `config/`

## 📚 Documentation

- **[README.md](README.md)** - This file, main documentation
- **[PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)** - Complete guide to prompting techniques
- **[LOCAL_MODELS.md](LOCAL_MODELS.md)** - Guide to running models locally
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference for common commands
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - High-level project summary

### Configuration Files

- `config/models.yaml` - Model configurations
- `config/datasets.yaml` - Dataset configurations
- `config/prompting_techniques.yaml` - Prompting strategies

## FAQ

### Can I run models locally with PromptBench?

**Yes!** PromptBench fully supports local LLM execution through HuggingFace Transformers. This framework adds optimizations like quantization, GGUF support, and vLLM integration. See [LOCAL_MODELS.md](LOCAL_MODELS.md).

### How do I compare different prompting techniques?

Configure techniques in `config/prompting_techniques.yaml` and map them to datasets. The framework will automatically run all combinations. See [PROMPTING_GUIDE.md](PROMPTING_GUIDE.md).

### What if I have limited RAM/VRAM?

Use quantized models or GGUF format. Examples:

- 8GB RAM: Use Phi-3-mini with 4-bit quantization
- 16GB RAM + 8GB VRAM: Use Mistral-7B with 8-bit quantization
- See [LOCAL_MODELS.md](LOCAL_MODELS.md) for details

### How do I add a custom prompting technique?

Add it to `config/prompting_techniques.yaml`:

```yaml
- name: my_custom
  type: custom
  template: |
    {your_template_here}
```

### Can I use my own datasets?

Yes! Support for JSON, JSONL, and CSV formats. See "Adding Custom Datasets" section above.

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this framework, please cite PromptBench:

```bibtex
@article{zhu2023promptbench,
  title={PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts},
  author={Zhu, Kaijie and others},
  journal={arXiv preprint arXiv:2306.04528},
  year={2023}
}
```

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review configuration files for typos
3. Check logs in the `results/` directory
4. Ensure API keys are properly set

---

**Happy Benchmarking! 🚀**
