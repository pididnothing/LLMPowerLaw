# Important: Install bitsandbytes for 4-bit Quantization

Your configured models use **4-bit quantization** to fit in 4GB VRAM. This requires the `bitsandbytes` library.

## Quick Install

```bash
pip install bitsandbytes
```

## Why You Need This

Without bitsandbytes:

- ❌ Models won't load in 4-bit mode
- ❌ Will try to use full precision (14GB+ VRAM)
- ❌ CUDA Out of Memory error

With bitsandbytes:

- ✅ 4-bit quantization works
- ✅ Models fit in 4GB VRAM
- ✅ ~75% memory savings

## Check If Already Installed

```bash
pip list | Select-String bitsandbytes
```

If you see output, it's already installed. If not, run the install command above.

## Full Setup (If Starting Fresh)

```bash
# Activate venv
.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# Install bitsandbytes (commented out in requirements.txt)
pip install bitsandbytes

# Verify
python -c "import bitsandbytes; print('✅ bitsandbytes installed')"
```

## Troubleshooting

### "Could not find CUDA"

Make sure you have CUDA-compatible PyTorch:

```bash
# Check current PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
# See: https://pytorch.org/get-started/locally/
```

### "DLL load failed" (Windows)

Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

**After installing, you're ready to run:**

```bash
python experiments/run_benchmark.py
```
