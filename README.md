# LLM Power Law — Scaling-Law Benchmark Framework

Benchmark open-source LLMs across multiple prompting techniques to plot **model size vs accuracy** and empirically derive scaling laws for prompt-augmented LLMs.

All models are free, ungated, and run locally (Colab T4 or your own GPU). No API keys needed.

---

## Quick Start

1. Open **[LLM_PowerLaw_Colab.ipynb](LLM_PowerLaw_Colab.ipynb)** in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Use **Mode A** or **Mode A+** (see below) to enable experiments.
4. Run the benchmark cells and download results.

For local setup:

```bash
pip install -r requirements.txt
pip install bitsandbytes          # optional, needed for 4-bit models
python experiments/run_benchmark.py
```

---

## How Experiments Work

Every experiment is a **triplet**: `(model, dataset, prompting_technique)`.
All 195 triplets are defined in `config/experiments.yaml` (disabled by default).
The notebook provides two ways to enable the ones you want.

---

## Mode A — Pick Individual Experiments

Edit the `experiments_to_enable` list in the "Mode A" cell with specific experiment IDs:

```python
experiments_to_enable = [
    "p1_phi3_sst2_zero",
    "p1_tinyllama_sst2_few",
]
```

Use this when you want exact control over which triplets run.

---

## Mode A+ — Batch Enable by Cross-Product (Recommended)

Set three lists — every `(model × dataset × technique)` combination that exists in `experiments.yaml` is enabled automatically.

```python
MODELS     = ["smollm-135m", "qwen2.5-0.5b"]   # or ["*"] for all
DATASETS   = ["sst2", "arc_challenge"]           # or ["*"] for all
TECHNIQUES = ["zero_shot", "chain_of_thought"]   # or ["*"] for all
```

Glob patterns work: `"qwen*"` matches all Qwen models, `"*_shot*"` matches zero_shot + few_shot + few_shot_cot.

### Available Keys

**Models** (sorted by parameter count):

| Key                   | Model                    | Params |
| --------------------- | ------------------------ | ------ |
| `smollm-135m`         | SmolLM-135M-Instruct     | 0.135B |
| `smollm-360m`         | SmolLM-360M-Instruct     | 0.36B  |
| `qwen2.5-0.5b`        | Qwen2.5-0.5B-Instruct    | 0.5B   |
| `tinyllama-test`      | TinyLlama-1.1B-Chat      | 1.1B   |
| `qwen2.5-1.5b`        | Qwen2.5-1.5B-Instruct    | 1.5B   |
| `smollm-1.7b`         | SmolLM-1.7B-Instruct     | 1.7B   |
| `qwen2.5-3b`          | Qwen2.5-3B-Instruct      | 3B     |
| `phi-3-mini-4bit`     | Phi-3-mini-4k-instruct   | 3.8B   |
| `mistral-7b-instruct` | Mistral-7B-Instruct-v0.3 | 7.2B   |
| `qwen2.5-7b`          | Qwen2.5-7B-Instruct      | 7.6B   |

**Datasets:**

| Key             | Benchmark     | Task                            |
| --------------- | ------------- | ------------------------------- |
| `sst2`          | SST-2 (GLUE)  | Sentiment (positive / negative) |
| `arc_challenge` | ARC-Challenge | Science MCQ (A/B/C/D)           |
| `hellaswag`     | HellaSwag     | Commonsense MCQ (A/B/C/D)       |
| `gsm8k`         | GSM8K         | Math reasoning (numeric)        |
| `mmlu`          | MMLU          | Knowledge MCQ (A/B/C/D)         |

**Techniques:**

| Key                | Description                     |
| ------------------ | ------------------------------- |
| `zero_shot`        | Direct question, no examples    |
| `few_shot`         | Includes demonstration examples |
| `chain_of_thought` | Step-by-step reasoning          |
| `few_shot_cot`     | Examples + reasoning            |
| `role_expert`      | Domain-expert persona           |

---

## Project Structure

```
LLMPowerLaw/
├── LLM_PowerLaw_Colab.ipynb          # Main notebook (start here)
├── config/
│   ├── models.yaml                    # 10 scaling-law models + test models
│   ├── datasets.yaml                  # 5 benchmarks + test datasets
│   ├── experiments.yaml               # 195 experiment triplets
│   ├── prompting_techniques.yaml      # Base technique definitions
│   ├── prompting_techniques_generated.yaml  # 125 model-family templates
│   └── dataset_instructions.yaml      # Per-dataset instructions & label maps
├── experiments/
│   ├── run_benchmark.py               # Main benchmark runner
│   ├── experiment_config.py           # Config dataclasses
│   ├── prompt_manager.py              # Loads & applies prompting techniques
│   └── local_model_handler.py         # HuggingFace model loading & inference
├── utils/
│   ├── metrics.py                     # Accuracy, F1, exact-match
│   ├── logger.py                      # Experiment logging
│   └── generate_prompt_templates.py   # Regenerate 125 templates from sources
├── data_loaders/                      # Custom dataset loader
├── results/                           # Experiment output (JSON)
└── docs/                              # Setup & troubleshooting
```

---

## Configuration Quick Reference

Enable/disable a model:

```yaml
# config/models.yaml
- name: phi-3-mini-4bit
  enabled: true # or false
  load_in_4bit: true # 4-bit quantization (~75% memory saving)
```

Change sample count:

```yaml
# config/datasets.yaml
- name: sst2
  num_samples: 200
```

Per-experiment overrides in `experiments.yaml`:

```yaml
- id: p1_phi3_sst2_zero
  model: phi-3-mini-4bit
  dataset: sst2
  prompting_technique: sst2_zero_shot_phi3
  num_samples: 50 # override dataset default
  max_tokens: 10 # override model default
  enabled: true
```

Tune generation behavior (reasoning budget, sampling, truncation):

```yaml
# config/models.yaml
global_settings:
  generation_controls:
    default_preset: balanced # concise | balanced | reasoning | classification
    max_new_tokens: 256
    classification_max_new_tokens: 20
    temperature: 0.15
    top_p: 0.9
    repetition_penalty: 1.03
    max_input_tokens: 3072
```

Quick runtime override (no file edits required):

```bash
python experiments/run_benchmark.py \
  --model phi-3-mini-4bit \
  --dataset arc_challenge \
  --gen-preset reasoning \
  --gen-max-new-tokens 768 \
  --gen-temperature 0.2 \
  --gen-top-p 0.95
```

Notes:

- Reasoning does **not** have to be verbose, but multi-step tasks often need more token budget (`max_new_tokens`) to complete coherent reasoning.
- For strict label tasks, use `classification` preset or a low `classification_max_new_tokens` cap to avoid rambling outputs.

---

## Manually Edit Model Input Generation (YAML Workflow)

If you want to experiment with how prompts are built and sent to models, edit the YAMLs below.

### 1) Choose manual template mode vs HF chat templating

In `config/models.yaml`:

- `chat_template_mode: hf` -> use HuggingFace `tokenizer.apply_chat_template(...)` with message content.
- `chat_template_mode: manual` (or unset) -> use the full prompt string in the template directly.

This is useful when comparing model behavior between native tokenizer chat formatting and hand-written prompt wrappers.

### 2) Edit technique structure (global style)

In `config/prompt_technique_template.yaml`, edit technique skeletons such as:

- `few_shot`
- `chain_of_thought`
- `few_shot_cot`

This is where you change the generic layout (for example, `Input/Answer` vs `Question/Reasoning/Final Answer`).

### 3) Edit dataset-specific instructions and labels

In `config/dataset_instructions.yaml`, adjust:

- task instructions
- answer format constraints
- label map / label space

For classification tasks, this directly affects extraction and evaluation alignment.

### 4) Edit few-shot demonstrations and role personas

- `config/few_shot_samples.yaml` -> update in-context examples and reasoning traces
- `config/domain_experts.yaml` -> update role-expert persona text

### 5) Edit chat wrappers per model family

In `config/chat_templates.yaml`, edit system/user/assistant wrappers for each model family.

Use this when you want strict control over tokens like ChatML tags, instruction blocks, or family-specific formatting.

### 6) Regenerate experiment templates after source edits

After changing source YAMLs (steps 2-5), regenerate:

```bash
python -m utils.generate_prompt_templates --config-dir ./config --output-dir ./config
```

This updates:

- `config/generated_prompts.yaml`
- `config/model_template_map.yaml`
- `config/prompting_techniques_generated.yaml`

### 7) Quick direct-edit option (fast iteration)

For rapid experiments, you can directly edit `config/prompting_techniques_generated.yaml`.

- Pros: instant testing
- Cons: changes are overwritten next time you run the generator

Best practice: direct edit for quick tests, then move final changes back into source YAMLs and regenerate.

### 8) Target one experiment while iterating

In `config/experiments.yaml`, enable only one triplet (or a tiny set) and lower `num_samples` while testing prompt edits. This keeps turnaround fast and makes debugging easier.

---

## Hardware Guide

| Platform | VRAM  | Recommended models       | Quantization               |
| -------- | ----- | ------------------------ | -------------------------- |
| Colab T4 | 15 GB | All 10 models            | fp16 for ≤3B, 4-bit for 7B |
| 4 GB GPU | 4 GB  | smollm-135m → qwen2.5-3b | fp16 only                  |
| 8 GB GPU | 8 GB  | All up to 7B             | 4-bit for 7B               |
| CPU only | —     | smollm-135m, smollm-360m | float32 (slow)             |

---

## Troubleshooting

| Problem                              | Fix                                                                 |
| ------------------------------------ | ------------------------------------------------------------------- |
| `bitsandbytes` install error         | See [docs/INSTALL_BITSANDBYTES.md](docs/INSTALL_BITSANDBYTES.md)    |
| `sentencepiece` C++ error on Windows | See [docs/WINDOWS_INSTALL_FIX.md](docs/WINDOWS_INSTALL_FIX.md)      |
| Model outputs nonsense               | Check [docs/DIAGNOSIS_AND_FIXES.md](docs/DIAGNOSIS_AND_FIXES.md)    |
| 0% accuracy on SST-2                 | Label mapping issue — ensure you have the latest `run_benchmark.py` |

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
