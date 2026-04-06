# Debugging Guide for LLMPowerLaw Experiments

When reporting issues, provide these files depending on the error type:

## 1. **Configuration/YAML Errors** (e.g., missing 'type', invalid values)

Provide:

- `config/experiments.yaml` — experiment definitions
- `config/datasets.yaml` — dataset configs
- `config/models.yaml` — model configurations
- `config/prompting_techniques.yaml` — prompting technique definitions
- `config/prompting_techniques_generated.yaml` — auto-generated techniques

## 2. **Model Loading Errors** (e.g., rope_scaling, quantization, device issues)

Provide:

- Full error JSON from `results/experiment_TIMESTAMP_summary.json`
- Include the complete traceback
- Model config from `config/models.yaml` for the failing model
- Python version & package versions:
  ```bash
  python --version
  pip show transformers torch bitsandbytes
  ```

## 3. **Dataset Loading Errors**

Provide:

- `config/datasets.yaml` — dataset configuration
- Sample of the actual dataset file if custom (e.g., first 5 rows of `.jsonl`)
- Error message with dataset name

## 4. **Prompt/Technique Errors**

Provide:

- The prompting technique definition from `config/prompting_techniques.yaml` or `config/prompting_techniques_generated.yaml`
- The experiment ID using that technique
- Error traceback

## 5. **Generation/Output Errors** (model produces wrong output)

Provide:

- Sample predictions vs expected outputs
- The prompt template used
- Model config for that model
- Error traceback if applicable

---

## Known Issues & Solutions

### Issue: KeyError 'type' in rope_scaling (Phi-3 models)

**Status:** ✅ FIXED

- **Symptom:** `KeyError: 'type'` deep in transformers during Phi-3 initialization
- **Cause:** Incomplete rope_scaling config in Phi-3 model
- **Solution:** Auto-fixed in `local_model_handler.py`:
  1. Pre-loads config before model init
  2. Fixes rope_scaling if missing 'type'
  3. Defaults to 'eager' attention for Phi-3
- **Files Modified:** `experiments/local_model_handler.py`

### Issue: KeyError 'type' in config parsing

**Status:** ✅ FIXED

- **Symptom:** `KeyError: 'type'` when loading prompting techniques or datasets
- **Cause:** Config entries missing required 'type' field
- **Solution:** Updated defaults in config parsing:
  - `PromptTechnique.from_dict()` defaults to 'custom'
  - `DatasetConfig.from_dict()` defaults to 'huggingface'
- **Files Modified:**
  - `experiments/prompt_manager.py`
  - `experiments/experiment_config.py`

---

## Quick Diagnostic Commands

```bash
# Check Python & dependency versions
python --version && pip show transformers torch bitsandbytes

# Verify model can be loaded
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/Phi-3-mini-4k-instruct')"

# Test dataset loading
python -c "from datasets import load_dataset; print(load_dataset('mmlu', 'abstract_algebra', split='test'))"

# Check config syntax
python -c "import yaml; yaml.safe_load(open('config/experiments.yaml'))"

# Run single quick test experiment
python experiments/run_benchmark.py --experiment-name test --max-experiments 1
```

---

## Files Overview

| File                                         | Purpose                                  | When to Check                                  |
| -------------------------------------------- | ---------------------------------------- | ---------------------------------------------- |
| `config/experiments.yaml`                    | Defines experiment triplets              | Experiment not running, wrong triplet selected |
| `config/datasets.yaml`                       | Dataset configs & loading                | Dataset not found, wrong split/field names     |
| `config/models.yaml`                         | Model configs & parameters               | Model fails to load, wrong params              |
| `config/prompting_techniques.yaml`           | Prompting technique definitions          | Prompt formatting issues                       |
| `config/prompting_techniques_generated.yaml` | Auto-generated model-specific techniques | Prompt not matching model format               |
| `config/domain_experts.yaml`                 | Role personas for expert prompting       | Role-based prompt issues                       |
| `experiments/run_benchmark.py`               | Main orchestrator                        | Experiment flow issues                         |
| `experiments/experiment_config.py`           | Config parsing & validation              | Config loading errors                          |
| `experiments/prompt_manager.py`              | Prompt application & rendering           | Prompt generation errors                       |
| `experiments/local_model_handler.py`         | Model loading & inference                | Model loading/generation errors                |
| `data_loaders/custom_loader.py`              | Custom dataset loading                   | Custom dataset issues                          |
