# Configuration Summary - 4GB VRAM Setup

## ✅ Current Configuration (Ready to Test!)

### Enabled Models (3)

1. **tinyllama-test** - 1.1B params, ~1GB VRAM
2. **gemma-2b-4bit** - 2B params, ~1GB VRAM
3. **phi-3-mini-4bit** - 3.8B params, ~2GB VRAM

### Enabled Datasets (1)

1. **custom_classification_test** - 5 samples (quick test)

### What This Will Do

- Load 3 models sequentially
- Test each on 5 samples = 15 total predictions
- Complete in ~2-3 minutes
- Verify everything works before full experiments

## 🚀 Run Your First Test

```bash
# From d:\Projects\LLMPowerLaw
python experiments/run_benchmark.py
```

**Expected Duration**: 2-3 minutes
**Expected Output**: Progress bars + results saved to `results/` folder

## 📋 Step-by-Step Test Checklist

### ☐ Phase 1: Quick Verification (Now!)

```bash
python experiments/run_benchmark.py
```

- [ ] TinyLlama loads successfully
- [ ] Gemma-2B loads successfully
- [ ] Phi-3-mini loads successfully
- [ ] All predictions complete
- [ ] Results saved to `results/` folder

**✅ If all pass, proceed to Phase 2**

### ☐ Phase 2: Small Experiment (After Phase 1)

Edit `config/datasets.yaml`:

```yaml
- name: custom_classification_test
  enabled: false # ← Disable test

- name: sst2
  num_samples: 50 # ← Start with 50
  enabled: true # ← Enable this
```

Run:

```bash
python experiments/run_benchmark.py
```

Expected: 5-10 minutes, 3 models × 50 samples

### ☐ Phase 3: Full Experiment (After Phase 2)

Edit `config/datasets.yaml`:

```yaml
- name: sst2
  num_samples: 200 # ← Increase to 200
  enabled: true

- name: mnli
  num_samples: 200 # ← Add second dataset
  enabled: true
```

Run:

```bash
python experiments/run_benchmark.py
```

Expected: 30-60 minutes, 3 models × 2 datasets × 200 samples

## 🎛️ Optional: Enable Llama-2-7B (Best Quality)

**Only after successful Phase 1-3 tests!**

Edit `config/models.yaml`:

```yaml
- name: llama-2-7b-4bit
  enabled: true # ← Enable this
```

⚠️ **Warning**:

- Uses all 4GB VRAM (tight fit)
- Close browser and other GPU apps
- Slower inference (~15-25 tokens/sec)
- But excellent quality!

## 📊 Progress Bar Examples

### Model Loading

```
Loading model components: 100%|████████| 2/2 [00:30<00:00]
✓ Model loaded successfully on device: cuda:0
```

### Dataset & Predictions

```
Loading custom_classification_test: 100%|████████| 1/1 [00:01<00:00]
✓ Loaded 5 samples
tinyllama-test on custom_classification_test: 100%|████████| 5/5 [00:15<00:00]
```

### Overall Progress

```
Overall Progress: 100%|████████| 3/3 [02:45<00:00] (tinyllama/gemma/phi-3)
```

## 📁 Where to Find Results

```bash
# List all results
ls results/

# View latest experiment summary
cat results/*_summary.json

# Each model-dataset pair gets its own file:
# - experiment_TIMESTAMP_MODEL_DATASET_TIMESTAMP.json
# - experiment_TIMESTAMP_summary.json
```

## 🔍 Verify Everything Works

After Phase 1 test completes, check:

```bash
# Should see 3 experiment files + 1 summary
ls results/ | Select-String "$(Get-Date -Format 'yyyyMMdd')"
```

Each result file should contain:

- `status: "completed"`
- `metrics`: accuracy, precision, etc.
- `predictions`: array of all predictions

## 🛠️ Configuration Files Reference

### Models: `config/models.yaml`

Currently enabled for 4GB VRAM:

- Line ~51: `tinyllama-test`
- Line ~70: `gemma-2b-4bit`
- Line ~85: `phi-3-mini-4bit`

### Datasets: `config/datasets.yaml`

Currently enabled:

- Line ~31: `custom_classification_test` (5 samples)

### Available for enabling:

- `sst2`: Sentiment classification
- `mnli`: Natural language inference
- `qqp`: Question pair matching

## ⚡ Quick Commands

```bash
# Run experiments
python experiments/run_benchmark.py

# Run specific model only
python experiments/run_benchmark.py --model tinyllama-test

# Run specific dataset only
python experiments/run_benchmark.py --dataset custom_classification_test

# Check GPU usage (separate window)
nvidia-smi -l 1

# View latest results
cat results/*_summary.json | ConvertFrom-Json | Format-List
```

## 📚 Documentation

- **[QUICKSTART_4GB_VRAM.md](QUICKSTART_4GB_VRAM.md)** - Detailed guide for your setup
- **[MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md)** - How quantization works
- **[PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)** - Test different prompting techniques
- **[LOCAL_MODELS.md](LOCAL_MODELS.md)** - General local model info

## 🎯 Your Hardware Limits

| Resource   | Available | Per Model | Max Models  |
| ---------- | --------- | --------- | ----------- |
| GPU VRAM   | 4GB       | 1-2GB     | 1-2 at once |
| System RAM | 16GB      | ~4GB      | 3-4 at once |

**Note**: Models are loaded sequentially, not simultaneously. Each model loads → runs → unloads before the next.

## 🚨 Common Issues & Quick Fixes

### CUDA Out of Memory

```yaml
# Disable larger models, keep only:
- tinyllama-test: enabled
- gemma-2b-4bit: enabled
- phi-3-mini-4bit: disabled
```

### Too Slow

```yaml
# Use only tinyllama-test
- tinyllama-test: enabled
- gemma-2b-4bit: disabled
- phi-3-mini-4bit: disabled
```

### Model Download Taking Too Long

- First run downloads models (2-14GB each)
- Go make coffee ☕
- Subsequent runs use cache (fast)

---

## ✅ Ready to Start!

```bash
python experiments/run_benchmark.py
```

Watch the progress bars and verify everything works! 🎉
