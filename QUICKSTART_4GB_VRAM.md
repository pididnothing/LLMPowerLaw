# Quick Start Guide for 4GB VRAM Setup

This guide is tailored for your hardware: **4GB VRAM + 16GB RAM**

## ⚡ 30-Second Test Run

Verify everything works before committing to long experiments:

```bash
# 1. Activate your virtual environment
.venv\Scripts\Activate.ps1

# 2. Run quick test (takes ~1-2 minutes)
python experiments/run_benchmark.py
```

**What happens:**

- ✅ Loads TinyLlama (1.1B model) - ~30 seconds
- ✅ Tests on 5 samples from custom dataset
- ✅ Shows progress bars for everything
- ✅ Saves results to `results/` folder

## 🎯 Your Configured Models (All fit in 4GB VRAM)

Currently enabled in `config/models.yaml`:

| Model               | Size | VRAM | Quality   | Speed  | Status                      |
| ------------------- | ---- | ---- | --------- | ------ | --------------------------- |
| **tinyllama-test**  | 1.1B | ~1GB | Good      | Fast   | ✅ ENABLED                  |
| **gemma-2b-4bit**   | 2B   | ~1GB | Excellent | Fast   | ✅ ENABLED                  |
| **phi-3-mini-4bit** | 3.8B | ~2GB | Very Good | Medium | ✅ ENABLED                  |
| llama-2-7b-4bit     | 7B   | ~4GB | Excellent | Slow   | ⚠️ Disabled (uses all VRAM) |

**Recommendation**: Start with the 3 enabled models. Enable `llama-2-7b-4bit` only if you need maximum quality and can close other GPU applications.

## 🧪 Test Run → Full Experiment Workflow

### Phase 1: Quick Verification (1-2 minutes)

**Current Status**: ✅ Already configured!

- Model: `tinyllama-test` (enabled)
- Dataset: `custom_classification_test` (5 samples, enabled)

**Run it:**

```bash
python experiments/run_benchmark.py
```

**Expected output:**

```
Loading model components: 100%|████████| 2/2 [00:30<00:00]
✓ Model loaded successfully on device: cuda:0
Loading custom_classification_test: 100%|████████| 1/1 [00:01<00:00]
✓ Loaded 5 samples
tinyllama-test on custom_classification_test: 100%|████████| 5/5 [00:15<00:00]
Overall Progress: 100%|████████| 1/1 [00:45<00:00]
```

**✅ If this works, you're ready for Phase 2!**

### Phase 2: Small Experiments (5-10 minutes)

**Edit `config/datasets.yaml`:**

```yaml
# Disable test, enable small experiments
- name: custom_classification_test
  enabled: false # ← Change to false

- name: sst2
  num_samples: 50 # ← Change from 100 to 50 for first run
  enabled: true # ← Change to true
```

**Run it:**

```bash
python experiments/run_benchmark.py
```

**Now you'll test:**

- 3 models × 50 samples = 150 predictions
- Takes ~5-10 minutes
- Progress bars show each model's progress

### Phase 3: Full Experiments (30-60 minutes)

**Edit `config/datasets.yaml`:**

```yaml
- name: sst2
  num_samples: 200 # ← Increase sample size
  enabled: true

- name: mnli
  num_samples: 200 # ← Enable second dataset
  enabled: true
```

**Run it:**

```bash
python experiments/run_benchmark.py
```

**Now you'll test:**

- 3 models × 2 datasets × 200 samples = 1200 predictions
- Takes ~30-60 minutes
- Can run overnight if needed
- Results auto-saved as each experiment completes

## 📊 Monitoring Your Experiments

### Watch Progress

The progress bars show:

- Overall experiment progress
- Per-sample progress for each model
- Estimated time remaining

### Check GPU Usage (Optional)

Open another PowerShell window:

```powershell
# Watch GPU memory usage
nvidia-smi -l 1
```

You should see:

- **TinyLlama**: ~1GB / 4GB
- **Gemma-2B-4bit**: ~1GB / 4GB
- **Phi-3-mini-4bit**: ~2GB / 4GB
- **Llama-2-7B-4bit**: ~3.8GB / 4GB (tight!)

### Check Results

```powershell
# List all result files
ls results/

# View latest summary
cat results/*_summary.json | ConvertFrom-Json | ConvertTo-Json
```

## ⚙️ Quick Configuration Changes

### Enable Llama-2-7B (Best Quality)

**Edit `config/models.yaml`:**

```yaml
- name: llama-2-7b-4bit
  enabled: true # ← Change to true
```

⚠️ **Warning**: Uses all 4GB VRAM. Close browser/other GPU apps first!

### Add More Datasets

**Edit `config/datasets.yaml`:**

```yaml
- name: mnli
  enabled: true # ← Natural language inference

- name: qqp
  enabled: true # ← Question pair matching
```

### Reduce Sample Size (Faster Testing)

**Edit any dataset:**

```yaml
- name: sst2
  num_samples: 20 # ← Reduce from 100
```

## 🐛 Troubleshooting

### "CUDA Out of Memory"

**Solution 1**: Disable larger model

```yaml
- name: phi-3-mini-4bit
  enabled: false # Use only tinyllama and gemma-2b
```

**Solution 2**: Close other GPU applications

- Close browser tabs
- Close Discord, Slack, etc.
- Check Task Manager → Performance → GPU

**Solution 3**: Force CPU (slow but works)

```yaml
- name: tinyllama-test
  device: cpu # Add this line
```

### "Model download is slow"

First run downloads models:

- TinyLlama: ~2GB
- Gemma-2B: ~5GB
- Phi-3-mini: ~8GB
- Llama-2-7B: ~14GB

Subsequent runs use cached files (fast).

### "No progress, script seems frozen"

The model is loading! First load takes time:

- TinyLlama: 30-60 seconds
- Gemma-2B: 1-2 minutes
- Phi-3-mini: 2-3 minutes
- Llama-2-7B: 3-5 minutes

You'll see the progress bar once loading starts.

## 📈 Expected Performance

### With Your Hardware (4GB VRAM + 16GB RAM)

| Model           | Load Time | Prediction Speed | Quality   |
| --------------- | --------- | ---------------- | --------- |
| TinyLlama       | ~30s      | 40-60 tokens/sec | Good      |
| Gemma-2B-4bit   | ~1m       | 30-50 tokens/sec | Excellent |
| Phi-3-mini-4bit | ~2m       | 20-35 tokens/sec | Very Good |
| Llama-2-7B-4bit | ~3m       | 15-25 tokens/sec | Excellent |

### Time Estimates

**For 100 samples:**

- TinyLlama: ~5 minutes
- Gemma-2B: ~8 minutes
- Phi-3-mini: ~12 minutes
- All 3 together: ~25 minutes

**For 500 samples:**

- TinyLlama: ~25 minutes
- Gemma-2B: ~40 minutes
- Phi-3-mini: ~60 minutes
- All 3 together: ~2 hours

## 🎓 Next Steps

1. **✅ Run the 30-second test** (configured and ready!)
2. **📊 Try 50-sample experiment** (5-10 minutes)
3. **📈 Scale up to 200 samples** (30-60 minutes)
4. **🔬 Enable more datasets** (mnli, qqp, etc.)
5. **🧠 Try prompting techniques** (see PROMPTING_GUIDE.md)

## 📚 More Resources

- **[MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md)** - Deep dive into quantization
- **[PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)** - Test different prompting techniques
- **[LOCAL_MODELS.md](LOCAL_MODELS.md)** - General local model information

---

**Ready?** Just run `python experiments/run_benchmark.py` and watch the progress bars! 🚀
