# 🚀 LLM Power Law - Project Overview

Your complete LLM benchmarking framework has been successfully set up!

## 📁 Project Structure

```
LLMPowerLaw/
├── 📄 README.md                      # Comprehensive documentation
├── 📄 QUICKSTART.md                  # Quick start instructions
├── 📄 requirements.txt               # Python dependencies
├── 📄 setup.py                       # Setup verification script
├── 📄 .env.example                   # Environment variables template
├── 📄 .gitignore                     # Git ignore rules
│
├── 📂 config/                        # Configuration files
│   ├── models.yaml                   # Model configurations
│   └── datasets.yaml                 # Dataset configurations
│
├── 📂 datasets/                      # Dataset management
│   ├── __init__.py
│   ├── custom_loader.py              # Custom dataset loader
│   └── data/                         # Your dataset files (created after setup)
│
├── 📂 experiments/                   # Experiment execution
│   ├── __init__.py
│   ├── experiment_config.py          # Configuration manager
│   └── run_benchmark.py              # Main benchmark runner
│
├── 📂 utils/                         # Utility functions
│   ├── __init__.py
│   ├── logger.py                     # Logging utilities
│   └── metrics.py                    # Metrics calculation
│
├── 📂 notebooks/                     # Jupyter notebooks
│   └── analysis.ipynb                # Results analysis notebook
│
└── 📂 results/                       # Experiment results
    └── .gitkeep
```

## 🎯 Key Features

### ✅ Multi-Model Support

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet
- **HuggingFace**: Llama 2, Mistral, Llama 3, and any HF model
- **Google**: Gemini Pro

### ✅ Flexible Dataset Integration

- **PromptBench** built-in datasets (SST-2, MNLI, QQP, etc.)
- **Custom datasets** (JSON, JSONL, CSV formats)
- **HuggingFace datasets** (SQuAD, GSM8K, etc.)

### ✅ Comprehensive Evaluation

- Classification metrics (Accuracy, Precision, Recall, F1)
- QA metrics (Exact Match, Token F1)
- Generation metrics (N-gram overlap)
- Per-class metrics and confusion analysis

### ✅ Easy Configuration

- YAML-based configuration for models and datasets
- Environment variable management for API keys
- Enable/disable specific models and datasets easily

### ✅ Result Management

- JSON-based result storage
- Detailed logging
- TensorBoard integration
- Jupyter notebook for visualization

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy the example environment file
copy .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# HUGGINGFACE_TOKEN=hf_...
```

### Step 3: Verify Setup

```bash
python setup.py
```

This will:

- Check all dependencies
- Verify configuration files
- Create example datasets
- Test configuration loading

### Step 4: Configure Your Experiments

**Enable Models** (config/models.yaml):

```yaml
models:
  - name: gpt-3.5-turbo
    enabled: true # Set to true
```

**Enable Datasets** (config/datasets.yaml):

```yaml
datasets:
  - name: sst2
    enabled: true # Set to true
```

### Step 5: Run Experiments

```bash
# Run all enabled experiments
python experiments/run_benchmark.py

# Run specific combinations
python experiments/run_benchmark.py --model gpt-3.5-turbo --dataset sst2

# Custom experiment name
python experiments/run_benchmark.py --experiment-name my_test_v1
```

### Step 6: Analyze Results

```bash
# Open Jupyter notebook for analysis
jupyter notebook notebooks/analysis.ipynb

# Or view the JSON results directly
ls results/
```

## 📊 Example Workflow

### 1. Testing with Custom Dataset

Create a custom dataset file `datasets/data/my_task.jsonl`:

```jsonl
{"input": "This is great!", "label": "positive"}
{"input": "Not good at all.", "label": "negative"}
```

Add to `config/datasets.yaml`:

```yaml
- name: my_task
  type: custom
  file_path: "./datasets/data/my_task.jsonl"
  task_type: classification
  format: jsonl
  fields:
    text: "input"
    label: "label"
  enabled: true
```

### 2. Comparing Multiple Models

Enable multiple models in `config/models.yaml`:

```yaml
models:
  - name: gpt-3.5-turbo
    enabled: true
  - name: gpt-4
    enabled: true
  - name: claude-3-sonnet
    enabled: true
```

Run experiments:

```bash
python experiments/run_benchmark.py --dataset my_task
```

### 3. Analyzing Results

Results are saved in `results/` directory:

- Individual results: `experiment_model_dataset_timestamp.json`
- Summary: `experiment_summary.json`
- Logs: `experiment.log`

Load in notebook:

```python
import json
with open('results/experiment_summary.json') as f:
    results = json.load(f)
```

## 📚 Documentation

- **README.md**: Complete documentation with all features
- **QUICKSTART.md**: Quick reference guide
- **Inline comments**: All code is well-documented
- **Example notebooks**: Provided for analysis

## 🔧 Customization

### Add New Model Provider

Edit `experiments/run_benchmark.py`:

```python
def initialize_model(self, model_config):
    if model_config.provider == 'your_provider':
        # Your initialization code
        pass
```

### Add Custom Metrics

Edit `utils/metrics.py`:

```python
@staticmethod
def your_custom_metric(predictions):
    # Your metric calculation
    return {'metric_name': value}
```

### Add New Dataset Type

Edit `datasets/custom_loader.py`:

```python
@staticmethod
def load_your_format(file_path):
    # Your loading code
    return data
```

## 📈 Expected Output

After running experiments, you'll have:

```
results/
├── experiment_20260221_100000.log
├── experiment_20260221_100000_summary.json
├── experiment_gpt-3.5-turbo_sst2_20260221_100000.json
├── experiment_gpt-4_sst2_20260221_100001.json
└── analysis_summary.csv
```

Each result file contains:

- Model and dataset info
- Execution time
- All predictions
- Comprehensive metrics
- Error information (if any)

## 🎓 Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run setup**: `python setup.py`
3. **Add API keys**: Edit `.env` file
4. **Configure experiments**: Enable models and datasets
5. **Run benchmarks**: `python experiments/run_benchmark.py`
6. **Analyze results**: Use the Jupyter notebook

## 💡 Tips

- Start with small datasets to test your setup
- Use `num_samples` to limit dataset size during testing
- Enable `use_cache` to speed up repeated runs
- Check logs in `results/` directory for debugging
- Use the analysis notebook for visualizations

## 🆘 Troubleshooting

**Import Errors**: Run `pip install -r requirements.txt`

**API Key Issues**: Check `.env` file and ensure keys are correct

**PromptBench Errors**: Optional, you can use custom datasets without it

**Memory Issues**: Use `load_in_8bit: true` for large models

## 📞 Support

Check these resources:

1. README.md for detailed documentation
2. QUICKSTART.md for quick reference
3. Example configurations in config/
4. Logs in results/ directory

---

**Your LLM benchmarking framework is ready! 🎉**

Start by running: `python setup.py`
