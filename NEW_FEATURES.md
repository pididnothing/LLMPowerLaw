# New Features Summary

## 🎉 What's New

Your LLM benchmarking framework has been enhanced with two major features:

### 1. 🎯 Prompting Techniques Support

### 2. 💻 Local Model Execution

## 🎯 Prompting Techniques

### What is it?

Test how different prompting strategies affect model performance. Compare zero-shot vs few-shot vs chain-of-thought and more!

### Built-in Techniques

- **Zero-Shot**: Direct questions
- **Few-Shot**: Learn from examples (3-5 examples)
- **Chain-of-Thought (CoT)**: Step-by-step reasoning
- **Role Prompting**: Assign expert personas
- **Emotion Prompting**: Add emotional appeal
- **Expert Prompting**: Invoke domain expertise

### Custom Techniques

Create your own prompting templates with placeholders:

```yaml
- name: my_custom_prompt
  type: custom
  template: |
    Task: {task}
    Input: {input}
    Instructions: Think carefully and provide a detailed answer.
```

### How to Use

1. **Configure techniques** in `config/prompting_techniques.yaml`
2. **Map to datasets**:
   ```yaml
   dataset_technique_mapping:
     sst2: [zero_shot, few_shot, cot] # Test 3 techniques
   ```
3. **Run experiments**:
   ```bash
   python experiments/run_benchmark.py
   ```

### Results

Each experiment shows which technique was used:

```
gpt-3.5-turbo on sst2 [zero_shot]: 0.85
gpt-3.5-turbo on sst2 [few_shot]: 0.88  ← Few-shot improved!
gpt-3.5-turbo on sst2 [cot]: 0.87
```

### Documentation

📚 **Complete Guide:** [PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)

---

## 💻 Local Model Execution

### What is it?

Run LLM models on your own machine with full privacy and no API costs!

### ✅ PromptBench Compatibility

**Yes! PromptBench fully supports local LLM execution** through HuggingFace Transformers. This framework extends that support with:

- Memory optimization (8-bit/4-bit quantization)
- Multiple formats (HF, GGUF, vLLM)
- Easy configuration
- Automatic GPU/CPU detection

### Supported Formats

#### 1. HuggingFace Local

Run any HF model directly:

```yaml
- name: llama-2-7b-local
  provider: huggingface_local
  model_id: meta-llama/Llama-2-7b-chat-hf
  load_in_8bit: true # Reduce memory by 50%
  device: auto
```

#### 2. GGUF (llama.cpp)

Quantized models for efficient CPU/GPU inference:

```yaml
- name: llama-2-7b-gguf
  provider: gguf
  model_id: TheBloke/Llama-2-7B-Chat-GGUF
  model_file: llama-2-7b-chat.Q4_K_M.gguf
  n_gpu_layers: 32 # GPU layers
```

#### 3. vLLM

High-performance serving:

```yaml
- name: llama-2-7b-vllm
  provider: vllm
  model_id: meta-llama/Llama-2-7b-chat-hf
  gpu_memory_utilization: 0.9
```

### Memory Optimization

| Technique   | Memory Reduction | Example (7B model) |
| ----------- | ---------------- | ------------------ |
| Full (FP32) | -                | ~28 GB             |
| FP16        | 50%              | ~14 GB             |
| 8-bit       | 75%              | ~7 GB              |
| 4-bit       | 87.5%            | ~3.5 GB            |
| GGUF Q4     | 87.5%            | ~3.5 GB            |

### Hardware Requirements

- **8GB RAM**: Phi-3-mini (4-bit), GGUF models
- **16GB RAM + 8GB VRAM**: Mistral-7B (8-bit), Llama-2-7B (8-bit)
- **32GB RAM + 16GB VRAM**: Llama-2-13B (FP16), Llama-2-7B (full)
- **80GB VRAM**: Llama-2-70B (FP16)

### How to Use

1. **Install dependencies**:

   ```bash
   # Basic (required)
   pip install torch transformers accelerate

   # For quantization (optional)
   pip install bitsandbytes

   # For GGUF (optional)
   pip install llama-cpp-python
   ```

2. **Configure model** in `config/models.yaml`:

   ```yaml
   - name: mistral-7b-local
     provider: huggingface_local
     model_id: mistralai/Mistral-7B-Instruct-v0.2
     load_in_8bit: true
     enabled: true
   ```

3. **Run experiments**:
   ```bash
   python experiments/run_benchmark.py --model mistral-7b-local
   ```

### Supported Models

- Llama 2 (7B, 13B, 70B)
- Llama 3 (8B, 70B)
- Mistral (7B, 8x7B)
- Phi-3 (mini, small, medium)
- Gemma (2B, 7B)
- Any HuggingFace causal LM model

### Documentation

📚 **Complete Guide:** [LOCAL_MODELS.md](LOCAL_MODELS.md)

---

## 🔧 New Files Created

### Configuration

- `config/prompting_techniques.yaml` - Prompting strategies configuration

### Core Modules

- `experiments/prompt_manager.py` - Prompting techniques handler
- `experiments/local_model_handler.py` - Local model loader and inference

### Documentation

- `PROMPTING_GUIDE.md` - Complete guide to prompting techniques
- `LOCAL_MODELS.md` - Complete guide to local model execution
- `NEW_FEATURES.md` - This file

### Updated Files

- `config/models.yaml` - Added local model configurations
- `experiments/run_benchmark.py` - Integrated prompting and local models
- `requirements.txt` - Added optional dependencies
- `README.md` - Updated with new features
- `experiments/__init__.py` - Added new exports

---

## 🚀 Quick Start Examples

### Example 1: Compare Prompting Techniques

```bash
# 1. Enable techniques in config/prompting_techniques.yaml
# 2. Map to dataset:
#    dataset_technique_mapping:
#      sst2: [zero_shot, few_shot, cot]
# 3. Run:
python experiments/run_benchmark.py --model gpt-3.5-turbo --dataset sst2
```

Results will show 3 experiments with different techniques.

### Example 2: Run Local Model

```bash
# 1. Enable model in config/models.yaml:
#    - name: mistral-7b-local
#      provider: huggingface_local
#      load_in_8bit: true
#      enabled: true
# 2. Run:
python experiments/run_benchmark.py --model mistral-7b-local
```

Model runs on your machine, no API calls!

### Example 3: Local Model + Prompting Techniques

```bash
# Combine both features!
# Configure local model + prompting techniques, then:
python experiments/run_benchmark.py --model mistral-7b-local --dataset sst2
```

Tests local model with multiple prompting strategies.

---

## 💡 Usage Tips

### Prompting Techniques

1. Start with zero-shot, few-shot, and CoT
2. Test on small datasets first (num_samples: 100)
3. Create custom templates for specific tasks
4. Compare results to find best technique

### Local Models

1. Check memory requirements before downloading
2. Start with 8-bit quantization
3. Use GGUF for CPU-only machines
4. Monitor GPU memory with `nvidia-smi`
5. Enable offline_mode after first download

### Combined Usage

1. Test prompting techniques on API models first (faster)
2. Apply best techniques to local models
3. Use local models for large-scale experiments (cost savings)

---

## 📊 Example Results

### Prompting Technique Comparison

```
Model: gpt-3.5-turbo, Dataset: sst2

Baseline (no prompting):     0.84
Zero-shot:                    0.85
Few-shot (3 examples):        0.88 ← +4% improvement!
Chain-of-Thought:             0.87
Role prompting:               0.86
Custom instructive:           0.89 ← Best!
```

### Local vs API Model

```
Task: Classification on SST-2 (1000 samples)

gpt-3.5-turbo (API):          0.87  |  Cost: $0.50  |  Time: 2 min
mistral-7b-local (8-bit):     0.84  |  Cost: $0.00  |  Time: 5 min
```

---

## 🎓 Learning Resources

### Prompting Techniques

- Wei et al. (2022) - Chain-of-Thought Prompting
- Brown et al. (2020) - Language Models are Few-Shot Learners
- PromptBench Paper (2023)

### Local Models

- HuggingFace Transformers Documentation
- bitsandbytes Quantization Guide
- llama.cpp Documentation
- vLLM Documentation

---

## 🤝 Getting Help

1. **Documentation**: Check the guides in this folder
2. **Examples**: See configuration files in `config/`
3. **Errors**: Review logs in `results/` directory
4. **Dependencies**: Run `python experiments/local_model_handler.py` to check

---

## ✨ What's Next?

### Experiment Ideas

1. Compare all prompting techniques across multiple models
2. Test local models vs API models on your task
3. Find memory-optimal configuration for your hardware
4. Create custom prompting templates for your domain

### Advanced Use Cases

1. Run large-scale experiments with local models (zero cost!)
2. Test robustness across prompting techniques
3. Optimize prompts for specific models
4. Build your own prompting technique library

---

**Happy Experimenting! 🚀**

For detailed documentation:

- [PROMPTING_GUIDE.md](PROMPTING_GUIDE.md)
- [LOCAL_MODELS.md](LOCAL_MODELS.md)
- [README.md](README.md)
