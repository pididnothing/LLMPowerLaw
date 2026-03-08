# Quick Start: Running Large Models

Want to run Llama 3 8B, Llama 2 13B, or even larger models? Here's everything you need.

## TL;DR - Just Show Me How

### Step 1: Edit `config/models.yaml`

Find a large model configuration and set `enabled: true`:

```yaml
- name: llama-2-13b-4bit
  provider: huggingface_local
  model_id: meta-llama/Llama-2-13b-chat-hf
  load_in_4bit: true # ← This is the magic
  enabled: true # ← Enable this model
```

### Step 2: Pick a Small Test Dataset

Edit `config/datasets.yaml` - start with just 50 samples:

```yaml
- name: sst2
  type: promptbench
  task_type: classification
  num_samples: 50 # ← Small for testing
  enabled: true
```

### Step 3: Run

```bash
python experiments/run_benchmark.py
```

### Step 4: Watch the Progress Bars! 🎉

You'll see:

- ✅ Model loading progress (tokenizer + model)
- ✅ Dataset loading progress
- ✅ Per-sample prediction progress with live counts
- ✅ Overall experiment progress

## Memory Requirements

| Model | Full Precision | 4-bit Quantized | Recommendation    |
| ----- | -------------- | --------------- | ----------------- |
| 7B    | 14GB           | ~4GB            | Any modern GPU    |
| 8B    | 16GB           | ~5GB            | 8GB+ VRAM         |
| 13B   | 26GB           | ~7GB            | 12GB+ VRAM        |
| 70B   | 140GB          | ~35GB           | 48GB+ VRAM or CPU |

## Available Large Models (Pre-configured)

All of these are ready to use in `config/models.yaml`:

### Llama Family

- `llama-2-7b-local` - 7B parameters (14GB → 4GB with 4-bit)
- `llama-2-13b-4bit` - 13B parameters (~7GB VRAM)
- `llama-3-8b-local` - Latest Llama 3 (16GB → 5GB with 4-bit)
- `llama-3-70b-4bit` - Huge but possible (~35GB, CPU+GPU)

### Others

- `mistral-7b-local` - Efficient 7B model
- `mixtral-8x7b-4bit` - Mixture of Experts (~25GB)
- `phi-3-mini-local` - Microsoft's compact model
- `qwen-14b-4bit` - Alibaba's 14B model

## Common Issues

### "CUDA Out of Memory"

**Solution 1**: Use 4-bit instead of 8-bit

```yaml
load_in_4bit: true
load_in_8bit: false
```

**Solution 2**: Reduce output tokens

```yaml
max_tokens: 256 # or even 128
```

**Solution 3**: Force CPU (slow but works)

```yaml
device: cpu
```

### "Model downloads are slow"

First run downloads the model (~5-35GB depending on model). Subsequent runs use cached version.

Set custom cache directory:

```bash
# In your terminal before running
export HF_HOME=/path/to/large/disk/cache
```

### "Inference is too slow"

Expected! Larger models with quantization are 2-3x slower:

- 7B full precision: 50-100 tokens/sec
- 7B 4-bit: 20-40 tokens/sec
- 13B 4-bit: 10-25 tokens/sec
- 70B 4-bit: 2-10 tokens/sec

**But** - For experiments, accuracy matters more than speed!

## Progress Bar Example

When you run an experiment, you'll see:

```
Loading model components: 100%|████████| 2/2 [00:45<00:00]
✓ Model loaded successfully on device: cuda:0
Loading sst2: 100%|████████| 1/1 [00:02<00:00]
✓ Loaded 50 samples
llama-2-13b-4bit on sst2: 100%|████████| 50/50 [02:15<00:00, 2.7s/sample]
Overall Progress: 100%|████████| 1/1 [02:15<00:00]
```

## Tips for Experiments

1. **Start small**: Test with 50 samples first
2. **Test locally**: Use TinyLlama (1.1B) to verify your setup works
3. **Then scale up**: Try 7B → 13B → 70B as needed
4. **Monitor resources**:
   - Windows: Task Manager → Performance
   - Linux: `watch nvidia-smi`
5. **Save money**: Local models = $0 inference cost!

## Install Missing Dependencies

If you see import errors:

```bash
# For quantization (required for large models)
pip install bitsandbytes

# For GGUF models (optional, CPU-optimized)
pip install llama-cpp-python

# For high-performance serving (optional, advanced)
pip install vllm
```

## Next Steps

1. ✅ Read [MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md) for deep dive
2. ✅ Check [LOCAL_MODELS.md](LOCAL_MODELS.md) for general local model info
3. ✅ See [PROMPTING_GUIDE.md](PROMPTING_GUIDE.md) to test different prompting techniques

---

**Ready?** Just set `enabled: true` on a large model, reduce `num_samples` to 50, and run! The progress bars will keep you sane. 🚀
