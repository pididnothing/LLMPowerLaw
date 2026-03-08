# ✅ Ready to Test - Just Run This!

Your system is **configured and ready** for a quick test run.

## What's Enabled

✅ **3 Models** (all fit in your 4GB VRAM):

- tinyllama-test (1.1B - fast)
- gemma-2b-4bit (2B - quality)
- phi-3-mini-4bit (3.8B - balanced)

✅ **1 Test Dataset**:

- custom_classification_test (5 samples only)

✅ **Total predictions**: 3 models × 5 samples = 15 predictions
✅ **Expected time**: 2-3 minutes

## Run the Test

```bash
python experiments/run_benchmark.py
```

## What You'll See

```
Loading model components: 100%|████████| 2/2 [00:30<00:00]
✓ Model loaded successfully on device: cuda:0

Loading custom_classification_test: 100%|████████| 1/1 [00:01<00:00]
✓ Loaded 5 samples

tinyllama-test on custom_classification_test: 100%|████████| 5/5 [00:15<00:00]
Overall Progress: 100%|████████| 1/3
```

This repeats for each of the 3 models.

## What Happens Next

1. **All 3 models load and run** (one at a time)
2. **Progress bars show status** (no guessing!)
3. **Results auto-saved** to `results/` folder
4. **Check results:**
   ```bash
   ls results/
   cat results/*_summary.json
   ```

## If Test Succeeds ✅

Proceed to larger experiments:

1. Open `config/datasets.yaml`
2. Change `custom_classification_test` → `enabled: false`
3. Change `sst2` → `enabled: true`, `num_samples: 50`
4. Run again: `python experiments/run_benchmark.py`

See **[CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md)** for step-by-step scaling.

## If Test Fails ❌

Common issues:

### "CUDA Out of Memory"

Edit `config/models.yaml`, disable phi-3-mini:

```yaml
- name: phi-3-mini-4bit
  enabled: false # Just run tinyllama + gemma-2b
```

### "Module not found: tqdm"

```bash
pip install tqdm
```

### "Module not found: bitsandbytes"

```bash
pip install bitsandbytes
```

### "Model downloading is slow"

First run downloads ~2-8GB per model. Grab coffee ☕, next runs will be fast!

---

**Ready? Just run:**

```bash
python experiments/run_benchmark.py
```

🎉 Watch the progress bars and verify it works!
