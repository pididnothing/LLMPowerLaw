# Setup Guide

Quick setup instructions for different environments.

## 🚀 Google Colab (Recommended for Quick Start)

**Easiest option** - No installation required, free GPU included!

1. Open [`LLM_PowerLaw_Colab.ipynb`](../LLM_PowerLaw_Colab.ipynb) in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU (T4)
3. Run all cells in order
4. Download results when done

**Hardware**: Colab provides ~15GB VRAM - enough for 7B-13B models with 4-bit quantization!

---

## 💻 Local Setup (Windows)

### Prerequisites

- Python 3.8-3.12
- NVIDIA GPU with CUDA support (optional but recommended)
- 8GB+ RAM (16GB recommended)

### Installation

```powershell
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/LLMPowerLaw.git
cd LLMPowerLaw

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install other packages
pip install sentencepiece --only-binary :all:
pip install -r requirements.txt

# 5. Install quantization support (for 4-bit models)
pip install bitsandbytes

# 6. Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Troubleshooting

**sentencepiece build error**:

```powershell
pip cache purge
pip install sentencepiece --only-binary :all:
```

**CUDA not available**:

- Install CUDA toolkit from NVIDIA
- Or use CPU mode (slower): Set `device: cpu` in `config/models.yaml`

---

## 🐧 Local Setup (Linux/Mac)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/LLMPowerLaw.git
cd LLMPowerLaw

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install bitsandbytes

# 4. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## ⚙️ Configuration

### Quick Test (Verify Setup)

1. Models are pre-configured in `config/models.yaml`
2. Test dataset is enabled by default (5 samples)
3. Run: `python experiments/run_benchmark.py`
4. Should complete in 2-3 minutes

### Hardware-Specific Recommendations

**4GB VRAM (GTX 1650, RTX 3050)**:

- Use: TinyLlama (1.1B), Gemma-2B-4bit, Phi-3-mini-4bit
- Already configured and enabled by default!

**8GB VRAM (RTX 3060, 2060)**:

- Add: Llama-2-7B-4bit
- Edit `config/models.yaml`: Set `enabled: true` for `llama-2-7b-4bit`

**12GB+ VRAM (RTX 3080, 4070)**:

- Add: Llama-2-13B-4bit
- Edit `config/models.yaml`: Set `enabled: true` for `llama-2-13b-4bit`

**15GB+ VRAM (Colab T4, RTX 4080)**:

- Run multiple large models simultaneously
- Or test 13B+ models with full precision

### Scaling Up After Test

1. **Small experiment** (5-10 minutes):

   ```yaml
   # In config/datasets.yaml
   - name: sst2
     num_samples: 50
     enabled: true
   ```

2. **Full experiment** (30-60 minutes):
   ```yaml
   - name: sst2
     num_samples: 200
     enabled: true
   - name: mnli
     num_samples: 200
     enabled: true
   ```

---

## 📊 Running Experiments

### Basic Usage

```bash
# Run all enabled models on all enabled datasets
python experiments/run_benchmark.py

# Run specific model only
python experiments/run_benchmark.py --model tinyllama-test

# Run specific dataset only
python experiments/run_benchmark.py --dataset sst2
```

### Progress Monitoring

The framework includes progress bars for:

- ✅ Model loading (tokenizer + model)
- ✅ Dataset loading
- ✅ Per-sample predictions with stats
- ✅ Overall experiment progress

### Results

Results are saved to `results/` folder:

- Individual experiment files: `experiment_TIMESTAMP_MODEL_DATASET.json`
- Summary file: `experiment_TIMESTAMP_summary.json`

View results:

```bash
# List all results
ls results/

# View summary (Windows)
cat results/*_summary.json | ConvertFrom-Json

# View summary (Linux/Mac)
cat results/*_summary.json | jq
```

---

## 🔧 Advanced Configuration

### Enable Prompting Techniques

Edit `config/prompting_techniques.yaml` to test different prompting strategies:

- Zero-shot
- Few-shot
- Chain-of-Thought
- Custom templates

See [PROMPTING_GUIDE.md](../PROMPTING_GUIDE.md) for details.

### Add Custom Datasets

1. Create dataset file in `data_loaders/data/`
2. Configure in `config/datasets.yaml`
3. Supported formats: JSONL, JSON, CSV

Example:

```yaml
- name: my_dataset
  type: custom
  file_path: "./data_loaders/data/my_data.jsonl"
  task_type: classification
  format: jsonl
  fields:
    text: "input"
    label: "label"
  num_samples: 100
  enabled: true
```

### Use API Models (OpenAI, Anthropic)

1. Create `.env` file:

   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

2. Enable API models in `config/models.yaml`:

   ```yaml
   - name: gpt-4
     enabled: true
   - name: claude-3-opus
     enabled: true
   ```

3. Run as normal - API keys are loaded automatically

---

## 📚 Additional Documentation

- **[MEMORY_OPTIMIZATION_GUIDE.md](../MEMORY_OPTIMIZATION_GUIDE.md)** - Deep dive into quantization
- **[PROMPTING_GUIDE.md](../PROMPTING_GUIDE.md)** - Prompting techniques guide
- **[LOCAL_MODELS.md](../LOCAL_MODELS.md)** - Local model execution details
- **[PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)** - Architecture overview

---

## ❓ Common Issues

### Out of Memory

- Enable 4-bit quantization: `load_in_4bit: true`
- Reduce samples: `num_samples: 50`
- Try smaller model: Use TinyLlama or Gemma-2B

### Slow Performance

- Expected with quantization (2-3x slower)
- Reduce sample size for testing
- Use GPU instead of CPU

### Import Errors

- Verify all packages installed: `pip list`
- Reinstall: `pip install -r requirements.txt --force-reinstall`
- Check Python version: `python --version` (should be 3.8-3.12)

### Unicode/Encoding Errors

- Files use UTF-8 encoding (fixed in latest version)
- If issues persist: Update to latest code

---

## 🆘 Getting Help

1. Check [docs/DIAGNOSIS_AND_FIXES.md](DIAGNOSIS_AND_FIXES.md)
2. Search existing issues on GitHub
3. Create new issue with:
   - Error message
   - System info (`python --version`, `pip list`)
   - Config files used
