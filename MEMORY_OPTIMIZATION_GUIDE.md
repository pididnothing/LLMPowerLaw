# Memory Optimization Guide: Running Larger Models

This guide explains how to run larger language models on consumer hardware by trading inference time for reduced memory usage.

## Available Memory-Efficient Techniques

### 1. **Quantization** (Recommended) ⚡

Quantization reduces model precision from 32-bit floats to lower precision, dramatically reducing memory usage with minimal accuracy loss.

#### 4-bit Quantization

- **Memory Savings**: ~75% reduction (e.g., 13B model: 26GB → 6.5GB)
- **Speed**: Slower inference (~2-3x slower)
- **Accuracy**: ~1-3% degradation for most tasks
- **Best for**: Maximum memory savings when running 13B+ models

```yaml
load_in_4bit: true
load_in_8bit: false
torch_dtype: auto
```

#### 8-bit Quantization

- **Memory Savings**: ~50% reduction (e.g., 13B model: 26GB → 13GB)
- **Speed**: Moderate slowdown (~1.5-2x slower)
- **Accuracy**: <1% degradation
- **Best for**: Balance between memory and accuracy

```yaml
load_in_8bit: true
load_in_4bit: false
torch_dtype: auto
```

### 2. **CPU Offloading**

Automatically offload model layers to CPU when GPU memory is insufficient.

```yaml
device: auto # Enables automatic CPU offloading
max_memory:
  0: "20GB" # Limit GPU usage
  cpu: "100GB" # Allow CPU usage
```

### 3. **Float16/BFloat16**

Use lower precision floating point for faster computation with minimal quality loss.

```yaml
torch_dtype: float16 # or bfloat16 (better for newer GPUs)
```

## Example Configurations

### Running Llama 3 8B (Consumer GPU with 8GB VRAM)

```yaml
- name: llama-3-8b-4bit
  provider: huggingface_local
  model_id: meta-llama/Meta-Llama-3-8B-Instruct
  max_tokens: 512
  temperature: 0.0
  device: auto
  load_in_4bit: true # Essential for 8GB GPU
  torch_dtype: auto
  trust_remote_code: false
  enabled: true
```

### Running Llama 2 13B (16GB VRAM or CPU)

```yaml
- name: llama-2-13b-4bit
  provider: huggingface_local
  model_id: meta-llama/Llama-2-13b-chat-hf
  max_tokens: 512
  temperature: 0.0
  device: auto
  load_in_4bit: true # Required for 13B model
  torch_dtype: auto
  trust_remote_code: false
  enabled: true
```

### Running Mixtral 8x7B (High-end GPU 24GB+)

```yaml
- name: mixtral-8x7b-4bit
  provider: huggingface_local
  model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
  max_tokens: 512
  temperature: 0.0
  device: auto
  load_in_4bit: true # Essential even for 24GB GPU
  torch_dtype: auto
  trust_remote_code: false
  enabled: true
```

### Running Llama 3 70B (CPU Only - Very Slow!)

```yaml
- name: llama-3-70b-4bit-cpu
  provider: huggingface_local
  model_id: meta-llama/Meta-Llama-3-70B-Instruct
  max_tokens: 256 # Reduce tokens for speed
  temperature: 0.0
  device: cpu # Force CPU
  load_in_4bit: true
  torch_dtype: auto
  trust_remote_code: false
  enabled: true
```

## Installation Requirements

For quantization support, install bitsandbytes:

```bash
# Windows/CUDA
pip install bitsandbytes

# Linux/CUDA
pip install bitsandbytes

# For CPU-only (slower)
pip install bitsandbytes-cpu
```

## Performance Expectations

| Model Size | Configuration | Memory | Speed (tokens/sec) |
| ---------- | ------------- | ------ | ------------------ |
| 7B         | Full (FP16)   | 14GB   | 50-100             |
| 7B         | 4-bit         | 4GB    | 20-40              |
| 13B        | Full (FP16)   | 26GB   | 30-60              |
| 13B        | 4-bit         | 7GB    | 10-25              |
| 70B        | 4-bit         | 35GB   | 2-10               |
| 70B        | 4-bit (CPU)   | 35GB   | 0.5-2              |

_Actual performance varies by hardware, batch size, and sequence length._

## Best Practices for Experiments

1. **Start Small**: Test with 7B models first to verify setup
2. **Enable Progress Bars**: Monitor long-running experiments
3. **Reduce Samples**: Use `num_samples: 100` for initial tests
4. **Watch Memory**: Use `nvidia-smi` or Task Manager to monitor usage
5. **Batch Processing**: Process smaller batches to avoid OOM errors
6. **Save Often**: Results are saved incrementally per experiment

## Troubleshooting

### Out of Memory (OOM)

```yaml
# If you get OOM errors, try:
load_in_4bit: true # Enable if not already
torch_dtype: float16 # Force float16
max_tokens: 256 # Reduce output length
```

### Slow Performance

```yaml
# If inference is too slow:
load_in_8bit: true # Use 8-bit instead of 4-bit
load_in_4bit: false
num_samples: 50 # Reduce dataset size
max_tokens: 128 # Reduce generation length
```

### Model Download Issues

```yaml
# For cached models:
local_model_path: "path/to/local/model"

# For offline mode (requires pre-downloaded models):
# In experiment_config.py global settings:
local_models:
  offline_mode: true
  hf_cache_dir: "path/to/cache"
```

## Running Your First Large Model Experiment

1. **Edit config/models.yaml**:

```yaml
- name: llama-3-8b-experiments
  provider: huggingface_local
  model_id: meta-llama/Meta-Llama-3-8B-Instruct
  max_tokens: 512
  temperature: 0.0
  device: auto
  load_in_4bit: true
  torch_dtype: auto
  trust_remote_code: false
  enabled: true # ← Set to true
```

2. **Edit config/datasets.yaml** (reduce samples for testing):

```yaml
- name: sst2
  type: promptbench
  task_type: classification
  num_samples: 50 # Start with 50 samples
  enabled: true
```

3. **Run the benchmark**:

```bash
python experiments/run_benchmark.py
```

4. **Monitor progress** (new progress bars will show):
   - Model loading stages
   - Dataset processing
   - Prediction generation per sample
   - Overall experiment progress

## Advanced: GGUF Models for CPU

For pure CPU inference, consider GGUF format (optimized for CPU):

```yaml
- name: llama-3-8b-gguf
  provider: gguf
  model_id: TheBloke/Llama-3-8B-GGUF
  model_file: llama-3-8b.Q4_K_M.gguf
  max_tokens: 512
  temperature: 0.0
  n_ctx: 2048
  n_gpu_layers: 0 # 0 = CPU only, >0 = offload to GPU
  enabled: true
```

Install llama-cpp-python:

```bash
# CPU only
pip install llama-cpp-python

# With CUDA support (faster)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

---

**Summary**: Set `load_in_4bit: true` for any model >7B parameters, monitor with progress bars, and reduce `num_samples` for initial testing. Inference will be 2-3x slower but use ~75% less memory! 🎯
