# 🚀 LLM Power Law - Project Overview

A comprehensive LLM benchmarking framework with progress monitoring, memory optimization, and flexible prompting techniques.

## 🎯 Quick Start Options

1. **Google Colab** (Recommended): Open [`LLM_PowerLaw_Colab.ipynb`](LLM_PowerLaw_Colab.ipynb) - zero setup, free GPU
2. **Local Setup**: Follow [docs/SETUP.md](docs/SETUP.md) for your platform

## 📁 Project Structure

```
LLMPowerLaw/
├── 📄 README.md                      # Main documentation
├── 📄 LLM_PowerLaw_Colab.ipynb      # 🚀 Start here for Colab!
├── 📄 requirements.txt               # Python dependencies
│
├── 📂 config/                        # Configuration (YAML files)
│   ├── models.yaml                   # Model configurations
│   ├── datasets.yaml                 # Dataset configurations
│   └── prompting_techniques.yaml     # Prompting strategies
│
├── 📂 experiments/                   # Core benchmarking code
│   ├── run_benchmark.py              # Main runner (with progress bars!)
│   ├── local_model_handler.py        # Local model loading
│   ├── prompt_manager.py             # Prompting techniques
│   └── experiment_config.py          # Configuration manager
│
├── 📂 data_loaders/                  # Dataset loaders
│   ├── custom_loader.py              # Custom dataset support
│   └── data/                         # Custom dataset files
│
├── 📂 utils/                         # Utilities
│   ├── logger.py                     # Logging with UTF-8 support
│   └── metrics.py                    # Comprehensive metrics
│
├── 📂 results/                       # Experiment outputs
│   └── (JSON files, logs, summaries)
│
├── 📂 docs/                          # Extended documentation
│   ├── SETUP.md                      # Complete setup guide
│   ├── DIAGNOSIS_AND_FIXES.md        # Troubleshooting
│   ├── WINDOWS_INSTALL_FIX.md        # Windows-specific issues
│   └── old/                          # Archived documentation
│
└── 📂 notebooks/                     # Analysis notebooks
    └── analysis.ipynb                # Results visualization
```

## 🎯 Key Features

### ✅ Progress Monitoring (NEW!)

- **Real-time progress bars** for all operations (tqdm library)
- Model loading progress (tokenizer + model)
- Dataset loading progress
- Per-sample prediction progress
- Overall experiment tracking

### ✅ Memory Optimization (NEW!)

- **4-bit quantization**: ~75% memory reduction (run 7B on 4GB VRAM!)
- **8-bit quantization**: ~50% memory reduction
- **GGUF support**: CPU-friendly quantized models
- **Auto device mapping**: Intelligent GPU/CPU utilization

### ✅ Prompting Techniques (NEW!)

- **PromptBench techniques**: Zero-shot, few-shot, chain-of-thought, role prompting
- **Custom templates**: Create your own prompting strategies
- **Dataset-specific mapping**: Different techniques per dataset
- **Automatic evaluation**: Compare prompting approaches

### ✅ Multi-Model Support

- **Local Models**: HuggingFace Transformers, GGUF, vLLM
- **API Models**: OpenAI (GPT-4, GPT-3.5), Anthropic (Claude), Google (Gemini)
- **Quantized Models**: Run large models on consumer hardware
- **Flexible providers**: Easy to add custom providers

### ✅ Flexible Dataset Integration

- **PromptBench datasets**: SST-2, MNLI, QQP, SQuAD, etc.
- **Custom datasets**: JSONL, JSON, CSV formats
- **HuggingFace datasets**: Direct integration
- **Test datasets**: 5-sample quick tests for verification

### ✅ Comprehensive Evaluation

- **Classification**: Accuracy, Precision, Recall, F1 (macro/per-class)
- **QA**: Exact Match, Token F1
- **Generation**: N-gram overlap scores
- **Detailed outputs**: Full predictions + metrics saved

## 🚀 Getting Started

### Option 1: Google Colab (Easiest!)

1. Open [`LLM_PowerLaw_Colab.ipynb`](LLM_PowerLaw_Colab.ipynb) in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells step by step
4. Download results from the Files panel

**Perfect for**: Zero-setup testing, free GPU (~15GB VRAM), no local installation

### Option 2: Local Installation

**See [docs/SETUP.md](docs/SETUP.md) for complete instructions**

Quick version:

```bash
# Install dependencies
pip install -r requirements.txt
pip install bitsandbytes  # For 4-bit quantization

# Run test (5 samples)
python experiments/run_benchmark.py
```

## 📊 Example Workflow

## 📊 Example Workflows

### 1. Quick Sanity Check (2-3 minutes)

```bash
# Uses pre-configured test dataset (5 samples)
python experiments/run_benchmark.py
```

Output:

```
Loading model components: 100%|████████| 2/2 [00:45<00:00]
gemma-2b-4bit on custom_classification_test: 100%|████| 5/5 [00:12<00:00]
✅ Accuracy: 0.80
```

### 2. Compare Prompting Techniques

Edit `config/prompting_techniques.yaml`:

```yaml
- name: zero_shot
  enabled: true
- name: few_shot
  enabled: true
  params:
    num_examples: 3
- name: chain_of_thought
  enabled: true
```

Results show technique comparison:

```
gemma-2b-4bit on sst2 [zero_shot]: Accuracy 0.82
gemma-2b-4bit on sst2 [few_shot]: Accuracy 0.87
gemma-2b-4bit on sst2 [chain_of_thought]: Accuracy 0.85
```

### 3. Scale to Larger Dataset

Edit `config/datasets.yaml`:

```yaml
- name: sst2
  num_samples: 200 # Increase from 5
  enabled: true
```

### 4. Run Memory-Efficient Large Model

Edit `config/models.yaml`:

```yaml
- name: llama-2-7b-4bit
  provider: huggingface_local
  model_id: meta-llama/Llama-2-7b-chat-hf
  load_in_4bit: true # 75% memory reduction!
  enabled: true
```

Can run 7B model on just 4GB VRAM!

### 5. Custom Dataset Evaluation

### 5. Custom Dataset Evaluation

Create `data_loaders/data/my_task.jsonl`:

```jsonl
{"input": "This is great!", "label": "positive"}
{"input": "Not good at all.", "label": "negative"}
```

Add to `config/datasets.yaml`:

```yaml
- name: my_task
  type: custom
  file_path: "./data_loaders/data/my_task.jsonl"
  task_type: classification
  format: jsonl
  fields:
    text: "input"
    label: "label"
  enabled: true
```

Run: `python experiments/run_benchmark.py`

## 📈 Results Structure

After running experiments:

```
results/
├── experiment_20260308_summary.json          # All results summary
├── experiment_gemma-2b_sst2_20260308.json   # Individual result
├── experiment_20260308.log                   # Detailed logs
└── (more result files...)
```

Each result file contains:

- Model and dataset configuration
- Start/end timestamps
- All predictions with inputs
- Comprehensive metrics (accuracy, F1, precision, recall)
- Error information (if any)

## 📚 Documentation Guide

| Document                                                     | Purpose                                 |
| ------------------------------------------------------------ | --------------------------------------- |
| [README.md](README.md)                                       | Main documentation & quick start        |
| [docs/SETUP.md](docs/SETUP.md)                               | Complete installation for all platforms |
| [LLM_PowerLaw_Colab.ipynb](LLM_PowerLaw_Colab.ipynb)         | Google Colab notebook                   |
| [QUICKSTART.md](QUICKSTART.md)                               | Quick reference commands                |
| [QUICKSTART_4GB_VRAM.md](QUICKSTART_4GB_VRAM.md)             | 4GB VRAM setup                          |
| [MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md) | Quantization deep dive                  |
| [PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)                     | Prompting techniques                    |
| [LOCAL_MODELS.md](LOCAL_MODELS.md)                           | Local model execution                   |
| [docs/DIAGNOSIS_AND_FIXES.md](docs/DIAGNOSIS_AND_FIXES.md)   | Troubleshooting                         |

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
def custom_metric(predictions):
    # Your metric calculation
    return {'metric_name': value}
```

### Create Custom Prompting Technique

Edit `config/prompting_techniques.yaml`:

```yaml
- name: my_custom_technique
  type: custom
  enabled: true
  template: |
    Context: {context}
    Question: {input}

    Let's think step by step:
```

## 💡 Tips & Best Practices

- **Start small**: Use test datasets (5 samples) to verify setup
- **Use quantization**: Enable 4-bit for large models on limited VRAM
- **Monitor progress**: Watch the progress bars to estimate completion time
- **Check logs**: `results/*.log` files contain detailed execution info
- **Compare techniques**: Enable multiple prompting techniques to find best approach
- **Iterate quickly**: Test with 5-100 samples before scaling to full datasets

## 🆘 Common Issues

| Issue                   | Solution                                                      |
| ----------------------- | ------------------------------------------------------------- |
| **Out of Memory**       | Enable `load_in_4bit: true` or reduce `num_samples`           |
| **Slow inference**      | Expected with quantization (~2-3x slower than full precision) |
| **Unicode errors**      | Fixed in latest version (UTF-8 encoding added)                |
| **sentencepiece error** | Windows: `pip install sentencepiece --only-binary :all:`      |
| **Import errors**       | `pip install -r requirements.txt`                             |

**Full troubleshooting**: [docs/DIAGNOSIS_AND_FIXES.md](docs/DIAGNOSIS_AND_FIXES.md)

## 🎓 Next Steps

1. **Run test**: `python experiments/run_benchmark.py` (2-3 min)
2. **Review results**: Check `results/` directory
3. **Enable more models**: Edit `config/models.yaml`
4. **Scale datasets**: Increase `num_samples` in `config/datasets.yaml`
5. **Try prompting**: Enable techniques in `config/prompting_techniques.yaml`
6. **Analyze**: Use `notebooks/analysis.ipynb` for visualization

## 🤝 Contributing

Areas to enhance:

- New model providers (Cohere, AI21, etc.)
- Additional metrics (BLEU, ROUGE, etc.)
- Custom dataset loaders
- Configuration presets
- Analysis tools

---

**Ready to benchmark LLMs! 🚀**

Start with: **[LLM_PowerLaw_Colab.ipynb](LLM_PowerLaw_Colab.ipynb)** or **[docs/SETUP.md](docs/SETUP.md)**
