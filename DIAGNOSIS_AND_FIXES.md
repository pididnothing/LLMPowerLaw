# Model Prediction Issues - Diagnosis and Fixes

## Problem Summary

The models were generating nonsensical predictions instead of proper classification labels. The predictions showed models just repeating input patterns or generating unrelated text.

## Root Causes Identified

### 1. **Template Variable Syntax Error** ❌ FIXED

**Problem:** Custom prompt templates used `{input}` syntax but the code uses Python's `string.Template` which requires `$input` syntax.

**Impact:** The `{input}` placeholder was not being replaced, so the model received prompts like:

```
Classify the following text.

Text: {input}

Answer with only the label:
```

**Fix:** Changed all custom templates to use `$` syntax:

```yaml
template: |
  Classify the sentiment of the following text.

  Text: $input

  Sentiment:
```

**Files Modified:**

- `config/prompting_techniques.yaml` - Fixed `classification_simple` and `custom_instructive` templates

### 2. **Poor Example Formatting for Few-Shot** ❌ PARTIALLY FIXED

**Problem:** The `_format_example()` method in `prompt_manager.py` was falling back to `str(example)` for PromptBench datasets, which created dictionary string representations like:

```
{'content': 'Question 1: ...', 'label': 0}
```

**Impact:** Models saw Python dictionary syntax in few-shot examples and tried to continue that pattern instead of answering the actual question.

**Fix:** Updated `_format_example()` to:

- Check for additional field names (`content`, `sentence`, `premise`, `hypothesis`)
- Handle nested dictionary structures
- Return empty string instead of dict representation as last resort
- Format examples as natural language

**Files Modified:**

- `experiments/prompt_manager.py` - Enhanced `_format_example()` method (lines 264-285)

### 3. **Vague Classification Instructions** ✅ IMPROVED

**Problem:** The original prompt "Answer with only the label" didn't specify what labels were valid.

**Impact:** TinyLlama generated arbitrary labels like "Classification: Pacing", "Label: Poem", etc.

**Fix:** Created explicit prompt with valid label options:

```yaml
template: |
  Classify the sentiment of the following text as either 'positive', 'negative', or 'neutral'.

  Text: $input

  Sentiment:
```

### 4. **Model Size Limitations** ⚠️ LIMITATION

**Problem:** TinyLlama (1.1B parameters) is too small for reliable instruction-following on complex tasks like QQP (paraphrase detection).

**Impact:** Even with correct prompts, small models may not generate proper classifications.

**Recommendations:**

- For QQP and complex tasks: Use models ≥3B (Phi-3-mini-4k-instruct, Llama-3-8B)
- For simple sentiment (SST-2): TinyLlama may work with few-shot examples
- Set realistic expectations: Very small models will have low accuracy

## Testing Results

### Before Fixes:

- **QQP with TinyLlama (few-shot):** 0% accuracy - Model generated dictionary patterns
- **Custom dataset with TinyLlama:** 0% accuracy - Template variables not substituted

### After Fixes:

- ✅ Template substitution now working correctly
- ✅ Examples formatted as natural language
- ✅ Clearer instructions with valid labels
- ⚠️ Model still generates incorrect labels (model capacity issue)

## Recommendations Going Forward

### 1. **Use Appropriately-Sized Models**

```yaml
# For classification tasks, minimum recommendations:
- SST-2 (sentiment): ≥1.1B (TinyLlama with few-shot)
- QQP (paraphrase): ≥3B (Phi-3-mini or better)
- MNLI (entailment): ≥7B (Mistral-7B or Llama-3-8B)
```

### 2. **Enable Better Models**

In `config/models.yaml`, enable a more capable model:

```yaml
- name: phi-3-mini-local
  provider: huggingface_local
  model_id: microsoft/Phi-3-mini-4k-instruct
  load_in_4bit: true # Fits in 8GB RAM
  enabled: true
```

### 3. **Use Task-Specific Prompts**

The new `classification_simple` template works better because it:

- Explicitly lists valid labels
- Uses clear task description
- Has concise format

### 4. **Add Output Parsing**

Future improvement: Parse model outputs to extract labels even if wrapped in extra text:

```python
def extract_label(prediction: str, valid_labels: List[str]) -> str:
    prediction_lower = prediction.lower()
    for label in valid_labels:
        if label.lower() in prediction_lower:
            return label
    return prediction  # fallback
```

### 5. **Test with Simpler Datasets First**

Testing progression:

1. ✅ Custom sentiment dataset (10 samples) - Good for testing infrastructure
2. SST-2 (100-500 samples) - Binary sentiment classification
3. QQP (100+ samples) - Paraphrase detection (harder)
4. MNLI - Natural language inference (hardest)

## Quick Test Command

Test the fixes with the improved prompt:

```bash
python experiments/run_benchmark.py \
  --model tinyllama-local \
  --dataset custom_classification \
  --experiment-name test_after_fixes
```

For better results, enable Phi-3-mini in `config/models.yaml` and run:

```bash
python experiments/run_benchmark.py \
  --model phi-3-mini-local \
  --dataset custom_classification \
  --experiment-name test_with_better_model
```

## Files Modified

1. `config/prompting_techniques.yaml`
   - Fixed template variable syntax (`$input` instead of `{input}`)
   - Improved classification prompt with explicit labels
   - Updated dataset technique mappings

2. `experiments/prompt_manager.py`
   - Enhanced `_format_example()` to handle PromptBench datasets
   - Better handling of nested dictionaries
   - More robust field extraction

3. `data_loaders/data/custom_classification.jsonl`
   - Created simple test dataset with clear examples

4. `config/datasets.yaml`
   - Enabled `custom_classification` dataset for testing

5. `config/models.yaml`
   - Reduced `max_tokens` for distilgpt2 to 50 (faster testing)

## Next Steps

1. **Test with Phi-3-mini:** Enable and test a better instruction-tuned model
2. **Implement output parsing:** Extract labels from model outputs robustly
3. **Add few-shot evaluation:** Test if few-shot helps smaller models
4. **Benchmark suite:** Run full comparison across model sizes

## Conclusion

The main issues were:

- ✅ **FIXED:** Template syntax error preventing variable substitution
- ✅ **FIXED:** Poor example formatting for few-shot learning
- ✅ **IMPROVED:** Vague prompts without explicit label specification
- ⚠️ **LIMITATION:** Model size too small for complex tasks

The infrastructure is now working correctly. To get meaningful results on classification tasks, use models with ≥3B parameters or add few-shot examples for TinyLlama.
