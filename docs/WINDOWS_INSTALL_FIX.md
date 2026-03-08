# Windows Installation Fix

## Issue: sentencepiece Build Error

You're seeing this error because `sentencepiece` needs C++ build tools to compile from source on Windows.

## ✅ Solution (Choose One)

### Option 1: Install Pre-built Wheel (Fastest)

```bash
# Clear any failed attempts
pip cache purge

# Install sentencepiece with pre-built wheel only
pip install sentencepiece --only-binary :all:

# Now install everything else
pip install -r requirements.txt
```

### Option 2: Skip Conflicting Package Temporarily

```bash
# Install everything except sentencepiece
pip install torch transformers openai anthropic pandas numpy datasets pyyaml python-dotenv scikit-learn scipy tqdm matplotlib seaborn tensorboard jupyter ipykernel

# Try sentencepiece separately
pip install sentencepiece==0.2.1
```

### Option 3: Install Without promptbench (If allowed)

```bash
# Install all dependencies except promptbench
pip install torch transformers openai anthropic pandas numpy datasets pyyaml python-dotenv scikit-learn scipy tqdm matplotlib seaborn tensorboard jupyter ipykernel sentencepiece

# Install promptbench without dependencies
pip install promptbench --no-deps
```

## After Installing: Add bitsandbytes

```bash
# Required for 4-bit quantization
pip install bitsandbytes
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import tqdm; print('✅ tqdm installed')"
python -c "import sentencepiece; print('✅ sentencepiece installed')"
```

## If You Keep Getting Errors

You can run the benchmark without sentencepiece for models that don't need it:

```bash
# Your configured models (TinyLlama, Gemma, Phi-3) work without sentencepiece
python experiments/run_benchmark.py
```

## Full Clean Install (Nuclear Option)

```bash
# Deactivate venv
deactivate

# Delete and recreate venv
Remove-Item -Recurse -Force .venv
python -m venv .venv

# Activate
.venv\Scripts\Activate.ps1

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install sentencepiece
pip install sentencepiece --only-binary :all:

# Install transformers and core deps
pip install transformers tqdm pandas numpy pyyaml python-dotenv

# Install bitsandbytes
pip install bitsandbytes

# Try the rest
pip install -r requirements.txt
```

---

**TL;DR**: Run these commands:

```bash
pip cache purge
pip install sentencepiece --only-binary :all:
pip install -r requirements.txt
pip install bitsandbytes
```
