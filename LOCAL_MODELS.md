# Local Models Guide

This guide explains how to run LLM models locally on your machine, supporting various model formats and optimization techniques.

## Table of Contents

- [Overview](#overview)
- [Supported Formats](#supported-formats)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Memory Optimization](#memory-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

Running models locally provides:

- **Privacy**: Data never leaves your machine
- **Cost savings**: No API fees
- **Customization**: Full control over model parameters
- **Offline operation**: No internet required (after download)

### PromptBench Local Model Support

**Yes, PromptBench supports locally run LLMs!** Through integration with HuggingFace Transformers, you can run any compatible model locally. This framework extends that support with additional optimizations and formats.

## Supported Formats

### 1. HuggingFace Models (Recommended)

Full-precision or quantized models from HuggingFace Hub.

**Supported models:**

- Llama 2 (7B, 13B, 70B)
- Llama 3 (8B, 70B)
- Mistral (7B, 8x7B MoE)
- Phi-3 (mini, small, medium)
- Gemma (2B, 7B)
- And thousands more...

### 2. GGUF Format

Quantized models for llama.cpp (CPU/GPU).

**Advantages:**

- Smaller file sizes
- Fast CPU inference
- Lower memory usage

### 3. vLLM

High-performance serving for production.

**Advantages:**

- Fastest inference
- Batch processing
- PagedAttention optimization

## Installation

### Basic Requirements

```bash
pip install torch transformers accelerate huggingface-hub
```

### For Quantization (GPU)

```bash
# 8-bit and 4-bit quantization
pip install bitsandbytes

# Requires NVIDIA GPU with CUDA
```

### For GGUF Models

```bash
# llama.cpp Python bindings
pip install llama-cpp-python

# With GPU support (optional)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### For vLLM (Production)

```bash
# High-performance serving
pip install vllm

# Requires NVIDIA GPU with CUDA
```

## Configuration

### HuggingFace Local Models

Edit `config/models.yaml`:

```yaml
models:
  - name: llama-2-7b-local
    provider: huggingface_local
    model_id: meta-llama/Llama-2-7b-chat-hf
    max_tokens: 512
    temperature: 0.0
    device: auto # auto, cuda, cpu
    load_in_8bit: false # Enable for 8-bit quantization
    load_in_4bit: false # Enable for 4-bit quantization
    torch_dtype: auto # auto, float16, bfloat16
    trust_remote_code: false
    local_model_path: null # Optional: path to local files
    enabled: true
```

### Device Options

- `auto`: Automatically detect best device
- `cuda`: Use NVIDIA GPU
- `cpu`: Use CPU only
- `mps`: Use Apple Silicon GPU (Mac)

### Memory Optimization Options

#### 8-bit Quantization

Reduces memory by ~50%:

```yaml
load_in_8bit: true
```

#### 4-bit Quantization

Reduces memory by ~75%:

```yaml
load_in_4bit: true
```

#### Custom Data Type

```yaml
torch_dtype: float16  # Half precision (2x smaller)
torch_dtype: bfloat16  # Better for training
torch_dtype: float32  # Full precision
```

### GGUF Models

```yaml
- name: llama-2-7b-gguf
  provider: gguf
  model_id: TheBloke/Llama-2-7B-Chat-GGUF
  model_file: llama-2-7b-chat.Q4_K_M.gguf
  max_tokens: 512
  temperature: 0.0
  n_ctx: 2048 # Context window
  n_gpu_layers: 0 # GPU layers (0 = CPU only)
  enabled: true
```

### vLLM Models

```yaml
- name: llama-2-7b-vllm
  provider: vllm
  model_id: meta-llama/Llama-2-7b-chat-hf
  max_tokens: 512
  temperature: 0.0
  tensor_parallel_size: 1 # Number of GPUs
  gpu_memory_utilization: 0.9
  enabled: true
```

### Global Settings

```yaml
global_settings:
  local_models:
    default_device: auto
    enable_8bit_by_default: false
    enable_4bit_by_default: false
    hf_cache_dir: null # Use HF default
    offline_mode: false
    gpu_memory_fraction: 0.9
    use_flash_attention: false
    compile_model: false
```

## Usage

### Run Local Models

```bash
# Enable a local model in config/models.yaml, then:
python experiments/run_benchmark.py

# Run specific local model
python experiments/run_benchmark.py --model llama-2-7b-local

# Test with small dataset
python experiments/run_benchmark.py --model llama-2-7b-local --dataset sst2
```

### Programmatic Usage

```python
from experiments import BenchmarkRunner, ExperimentConfig

# Load configuration
config = ExperimentConfig()
config.load_configs()

# Filter to local models only
config.models = [m for m in config.models if '_local' in m.name]

# Run experiments
runner = BenchmarkRunner(config=config)
results = runner.run_all_experiments()
```

### Direct Model Usage

```python
from experiments.local_model_handler import LocalModelHandler

# Configuration
model_config = {
    'provider': 'huggingface_local',
    'model_id': 'meta-llama/Llama-2-7b-chat-hf',
    'max_tokens': 512,
    'temperature': 0.0,
    'device': 'auto',
    'load_in_8bit': True
}

global_settings = {
    'local_models': {}
}

# Initialize and load
handler = LocalModelHandler(model_config, global_settings)
handler.load_model()

# Generate
response = handler.generate("What is machine learning?")
print(response)

# Cleanup
handler.unload_model()
```

## Memory Optimization

### Model Size Guidelines

| Model | Full (FP32) | FP16    | 8-bit  | 4-bit   |
| ----- | ----------- | ------- | ------ | ------- |
| 7B    | ~28 GB      | ~14 GB  | ~7 GB  | ~3.5 GB |
| 13B   | ~52 GB      | ~26 GB  | ~13 GB | ~6.5 GB |
| 70B   | ~280 GB     | ~140 GB | ~70 GB | ~35 GB  |

### Recommendations by Hardware

#### 8GB RAM / No GPU

```yaml
# Use GGUF quantized models
provider: gguf
model_file: model.Q4_K_M.gguf # 4-bit quantization
n_gpu_layers: 0 # CPU only
```

#### 16GB RAM / 8GB VRAM

```yaml
# Use 4-bit quantization
provider: huggingface_local
load_in_4bit: true
device: auto
```

#### 32GB+ RAM / 16GB+ VRAM

```yaml
# Use 8-bit or FP16
provider: huggingface_local
load_in_8bit: true
torch_dtype: float16
device: auto
```

#### 80GB+ VRAM (A100/H100)

```yaml
# Full precision or vLLM
provider: vllm
tensor_parallel_size: 1
```

### Example Configurations

#### Budget Setup (8GB RAM)

```yaml
- name: phi-3-mini-local
  provider: huggingface_local
  model_id: microsoft/Phi-3-mini-4k-instruct
  load_in_4bit: true
  device: cpu
```

#### Mid-Range (16GB RAM, 8GB VRAM)

```yaml
- name: mistral-7b-local
  provider: huggingface_local
  model_id: mistralai/Mistral-7B-Instruct-v0.2
  load_in_8bit: true
  device: auto
```

#### High-End (32GB+ RAM, 24GB+ VRAM)

```yaml
- name: llama-2-13b-local
  provider: huggingface_local
  model_id: meta-llama/Llama-2-13b-chat-hf
  torch_dtype: float16
  device: cuda
```

## Performance Tips

### 1. Use Flash Attention

For faster inference (requires flash-attn):

```yaml
global_settings:
  local_models:
    use_flash_attention: true
```

### 2. Compile Model

For PyTorch 2.0+ (one-time compilation cost):

```yaml
global_settings:
  local_models:
    compile_model: true
```

### 3. Batch Processing

Process multiple samples at once (for HF models):

```yaml
global_settings:
  default_batch_size: 8 # Adjust based on memory
```

### 4. Offload Strategy

For large models, offload some layers to CPU:

```python
max_memory = {0: "10GiB", "cpu": "30GiB"}
```

## Checking System Compatibility

```bash
# Check available dependencies
python experiments/local_model_handler.py
```

Output shows:

```
Available Libraries:
  ✓ transformers
  ✓ torch
  ✗ bitsandbytes  # Install if needed
  ✗ llama-cpp-python
  ✗ vllm
```

## Troubleshooting

### Out of Memory (OOM)

1. Enable quantization:

```yaml
load_in_8bit: true # or load_in_4bit: true
```

2. Reduce context window:

```yaml
max_tokens: 256 # Lower token limit
```

3. Use GGUF format:

```yaml
provider: gguf
n_gpu_layers: 0 # CPU inference
```

### Slow Inference

1. Check device:

```python
print(model.device)  # Should be cuda if GPU available
```

2. Enable GPU layers (GGUF):

```yaml
n_gpu_layers: 32 # Offload layers to GPU
```

3. Use vLLM for production:

```yaml
provider: vllm # Much faster
```

### Model Download Fails

1. Check HuggingFace token:

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

2. Use offline mode with pre-downloaded models:

```yaml
local_model_path: /path/to/model
offline_mode: true
```

### ImportError: bitsandbytes

For quantization on Windows:

```bash
# Use WSL or Docker, or install pre-built wheels
pip install bitsandbytes-windows
```

### CUDA Out of Memory

1. Reduce GPU memory usage:

```yaml
gpu_memory_utilization: 0.7 # Lower from 0.9
```

2. Use CPU offloading:

```yaml
device: auto # Splits load between GPU/CPU
```

## Model Recommendations

### Best Overall (7B class)

- **Mistral-7B-Instruct**: Excellent performance, efficient
- **Llama-3-8B-Instruct**: Latest Meta model, very capable

### Best for Low Memory

- **Phi-3-mini (3.8B)**: Surprisingly good, fits on 4GB
- **Gemma-2B-it**: Google's small but capable model

### Best for Quality (if you have VRAM)

- **Llama-2-13B**: Better than 7B, manageable size
- **Llama-3-70B** (4-bit): SOTA quality on consumer hardware

## Example Workflow

```bash
# 1. Check dependencies
python experiments/local_model_handler.py

# 2. Download model (first time only)
# Done automatically on first run, or:
huggingface-cli download meta-llama/Llama-2-7b-chat-hf

# 3. Enable model in config
# Edit config/models.yaml, set enabled: true

# 4. Set quantization based on your RAM
# load_in_8bit: true or load_in_4bit: true

# 5. Run experiments
python experiments/run_benchmark.py --model llama-2-7b-local

# 6. Monitor resource usage
# Use nvidia-smi (GPU) or htop (CPU)
```

## Resources

- [HuggingFace Models](https://huggingface.co/models)
- [GGUF Models](https://huggingface.co/TheBloke)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

For more information, see [README](README.md) or [Prompting Guide](PROMPTING_GUIDE.md).
