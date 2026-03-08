# LLM Power Law - Multi-Model Benchmarking Framework

A comprehensive framework for benchmarking LLMs with support for local models, quantization, prompting techniques, and custom datasets.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

**Easiest way to get started - no installation required!**

1. Open [`LLM_PowerLaw_Colab.ipynb`](LLM_PowerLaw_Colab.ipynb) in Google Colab
2. Enable GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells
4. Download your results

**Provides**: Free ~15GB VRAM GPU, zero setup, works immediately

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/LLMPowerLaw.git
cd LLMPowerLaw

# Install dependencies
pip install -r requirements.txt
pip install bitsandbytes  # For 4-bit quantization (optional)

# Run test (5 samples, ~2-3 minutes)
python experiments/run_benchmark.py
```

**Full instructions**: [docs/SETUP.md](docs/SETUP.md)

## ✨ Key Features

- 🚀 **Progress Bars**: Real-time monitoring of all operations
- 🧠 **Memory Optimization**: Run 7B-70B models with 4-bit/8-bit quantization
- 🎯 **Prompting Techniques**: Zero-shot, few-shot, chain-of-thought, custom templates
- 💻 **Multi-Provider Support**: HuggingFace, OpenAI, Anthropic, GGUF, vLLM
- 📊 **Multiple Datasets**: PromptBench datasets + custom JSONL/JSON/CSV
- 🔧 **Flexible Configuration**: Simple YAML files for models, datasets, prompts

## 📁 Project Structure

```
LLMPowerLaw/
├── LLM_PowerLaw_Colab.ipynb      # 👈 Start here for Colab
├── config/                        # Configuration files
│   ├── models.yaml               # Model setup
│   ├── datasets.yaml             # Dataset setup
│   └── prompting_techniques.yaml # Prompting strategies
├── experiments/                   # Core benchmarking code
│   ├── run_benchmark.py          # Main runner
│   ├── local_model_handler.py    # Local model loading
│   └── prompt_manager.py         # Prompting techniques
├── data_loaders/                 # Dataset loaders
├── utils/                        # Logging & metrics
├── results/                      # Output directory
└── docs/                         # Documentation
    └── SETUP.md                  # Complete setup guide
```

## 🎯 Hardware-Specific Guides

| Hardware                | Recommended Models | Quick Start Guide                                            |
| ----------------------- | ------------------ | ------------------------------------------------------------ |
| **Google Colab** (15GB) | 7B-13B with 4-bit  | [LLM_PowerLaw_Colab.ipynb](LLM_PowerLaw_Colab.ipynb)         |
| **4GB VRAM**            | 2B-3B with 4-bit   | [QUICKSTART_4GB_VRAM.md](QUICKSTART_4GB_VRAM.md)             |
| **8GB VRAM**            | 7B with 4-bit      | [MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md) |
| **12GB+ VRAM**          | 13B-70B with 4-bit | [LARGE_MODEL_QUICKSTART.md](LARGE_MODEL_QUICKSTART.md)       |
| **CPU Only**            | GGUF quantized     | [LOCAL_MODELS.md](LOCAL_MODELS.md)                           |

## 🔥 Quick Examples

### Run Test (5 samples, ~2 minutes)

```bash
python experiments/run_benchmark.py
```

### Enable More Models

```yaml
# Edit config/models.yaml
- name: llama-2-7b-4bit
  enabled: true # Change false → true
```

### Scale Up Dataset

```yaml
# Edit config/datasets.yaml
- name: sst2
  num_samples: 200 # Increase from 5
  enabled: true
```

### Run Specific Model/Dataset

```bash
python experiments/run_benchmark.py --model gemma-2b-4bit --dataset sst2
```

## 📊 Example Output

```
Loading model components: 100%|████████| 2/2 [00:45<00:00]
✓ Model loaded successfully on device: cuda:0
Loading sst2: 100%|████████| 1/1 [00:02<00:00]
✓ Loaded 100 samples
gemma-2b-4bit on sst2: 100%|████████| 100/100 [03:25<00:00, 2.1s/sample]
Overall Progress: 100%|████████| 1/1 [03:30<00:00]

Results saved to: results/experiment_20260308_summary.json
✅ Accuracy: 0.87
```

## 📚 Documentation

### Getting Started

- **[docs/SETUP.md](docs/SETUP.md)** - Complete installation guide (Colab, Windows, Linux)
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference commands
- **[LLM_PowerLaw_Colab.ipynb](LLM_PowerLaw_Colab.ipynb)** - Interactive Colab notebook

### Hardware-Specific

- **[QUICKSTART_4GB_VRAM.md](QUICKSTART_4GB_VRAM.md)** - For 4GB VRAM GPUs
- **[MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md)** - Quantization deep dive
- **[LARGE_MODEL_QUICKSTART.md](LARGE_MODEL_QUICKSTART.md)** - For 12GB+ VRAM

### Feature Guides

- **[PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)** - Prompting techniques explained
- **[LOCAL_MODELS.md](LOCAL_MODELS.md)** - Running models locally
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Architecture overview

### Configuration

- [`config/models.yaml`](config/models.yaml) - Model configurations
- [`config/datasets.yaml`](config/datasets.yaml) - Dataset configurations
- [`config/prompting_techniques.yaml`](config/prompting_techniques.yaml) - Prompting strategies

### Troubleshooting

- **[docs/DIAGNOSIS_AND_FIXES.md](docs/DIAGNOSIS_AND_FIXES.md)** - Common issues & solutions
- **[docs/WINDOWS_INSTALL_FIX.md](docs/WINDOWS_INSTALL_FIX.md)** - Windows-specific fixes

## 🎯 Common Use Cases

### Quick Sanity Check

```bash
# Uses test dataset (5 samples) - 2-3 minutes
python experiments/run_benchmark.py
```

### Compare Multiple Models

```yaml
# In config/models.yaml - enable multiple models
- name: gemma-2b-4bit
  enabled: true
- name: phi-3-mini-4bit
  enabled: true
- name: llama-2-7b-4bit
  enabled: true
```

### Test Prompting Techniques

```yaml
# In config/prompting_techniques.yaml
- name: zero_shot
  enabled: true
- name: few_shot
  enabled: true
- name: chain_of_thought
  enabled: true
```

### Custom Dataset

```yaml
# In config/datasets.yaml
- name: my_dataset
  type: custom
  file_path: "./data_loaders/data/my_data.jsonl"
  task_type: classification
  format: jsonl
  enabled: true
```

## 🔧 Configuration Quick Reference

Enable/disable models:

```yaml
# config/models.yaml
- name: model_name
  enabled: true # or false
```

Adjust sample size:

```yaml
# config/datasets.yaml
- name: dataset_name
  num_samples: 100 # or any number
```

Enable quantization:

```yaml
# config/models.yaml
- name: model_name
  load_in_4bit: true # 75% memory reduction
```

## 🆘 Troubleshooting

| Issue                   | Quick Fix                                           |
| ----------------------- | --------------------------------------------------- |
| **Out of Memory**       | Enable `load_in_4bit: true` or reduce `num_samples` |
| **Slow inference**      | Expected with quantization (~2-3x slower)           |
| **sentencepiece error** | `pip install sentencepiece --only-binary :all:`     |
| **Unicode error**       | Fixed in latest version                             |
| **Import errors**       | `pip install -r requirements.txt`                   |

**Full guide**: [docs/DIAGNOSIS_AND_FIXES.md](docs/DIAGNOSIS_AND_FIXES.md)

## 🤝 Contributing

Contributions welcome! Areas to enhance:

- New model providers
- Additional metrics
- Custom dataset loaders
- Configuration presets

## 📄 License

Provided as-is for research and educational purposes.

## 🙏 Citation

If you use this framework, please cite PromptBench:

```bibtex
@article{zhu2023promptbench,
  title={PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts},
  author={Zhu, Kaijie and others},
  journal={arXiv preprint arXiv:2306.04528},
  year={2023}
}
```

---

**Questions?** Check [docs/](docs/) or open an issue | **Quick start?** Open [LLM_PowerLaw_Colab.ipynb](LLM_PowerLaw_Colab.ipynb) 🚀
