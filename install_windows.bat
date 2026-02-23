@echo off
REM Windows Installation Script for LLM Benchmarking Framework
REM Handles sentencepiece installation issues on Python 3.13

echo ============================================================
echo LLM Benchmarking Framework - Windows Installation
echo ============================================================
echo.

REM Step 1: Upgrade pip
echo [1/10] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)
echo.

REM Step 2: Install sentencepiece first (newer version with wheels)
echo [2/10] Installing sentencepiece...
python -m pip install "sentencepiece>=0.2.0"
if errorlevel 1 (
    echo ERROR: Failed to install sentencepiece
    pause
    exit /b 1
)
echo.

REM Step 3: Install promptbench without dependencies
echo [3/10] Installing promptbench...
python -m pip install promptbench --no-deps
if errorlevel 1 (
    echo ERROR: Failed to install promptbench
    pause
    exit /b 1
)
echo.

REM Step 4: Install PyTorch and Transformers
echo [4/10] Installing PyTorch and Transformers...
python -m pip install "torch>=2.0.0" "transformers>=4.30.0" "accelerate>=0.20.0"
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch/Transformers
    pause
    exit /b 1
)
echo.

REM Step 5: Install API clients
echo [5/10] Installing API clients...
python -m pip install "openai>=1.0.0" "anthropic>=0.18.0"
if errorlevel 1 (
    echo ERROR: Failed to install API clients
    pause
    exit /b 1
)
echo.

REM Step 6: Install data handling
echo [6/10] Installing data handling packages...
python -m pip install "pandas>=2.0.0" "numpy>=1.24.0" "datasets>=2.14.0" "huggingface-hub>=0.17.0"
if errorlevel 1 (
    echo ERROR: Failed to install data packages
    pause
    exit /b 1
)
echo.

REM Step 7: Install configuration packages
echo [7/10] Installing configuration packages...
python -m pip install "pyyaml>=6.0" "python-dotenv>=1.0.0"
if errorlevel 1 (
    echo ERROR: Failed to install config packages
    pause
    exit /b 1
)
echo.

REM Step 8: Install metrics
echo [8/10] Installing metrics packages...
python -m pip install "scikit-learn>=1.3.0" "scipy>=1.11.0"
if errorlevel 1 (
    echo ERROR: Failed to install metrics packages
    pause
    exit /b 1
)
echo.

REM Step 9: Install visualization
echo [9/10] Installing visualization packages...
python -m pip install "tqdm>=4.65.0" "matplotlib>=3.7.0" "seaborn>=0.12.0" "tensorboard>=2.13.0"
if errorlevel 1 (
    echo ERROR: Failed to install visualization packages
    pause
    exit /b 1
)
echo.

REM Step 10: Install Jupyter
echo [10/10] Installing Jupyter...
python -m pip install "jupyter>=1.0.0" "ipykernel>=6.25.0"
if errorlevel 1 (
    echo ERROR: Failed to install Jupyter
    pause
    exit /b 1
)
echo.

echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo All required packages have been installed successfully.
echo.
echo Next steps:
echo   1. Copy .env.example to .env and add your API keys
echo   2. Configure models in config/models.yaml
echo   3. Configure datasets in config/datasets.yaml
echo   4. Run: python experiments/run_benchmark.py
echo.
echo Optional: For local model support, install:
echo   pip install bitsandbytes^>=0.41.0       # For quantization
echo   pip install llama-cpp-python^>=0.2.0    # For GGUF models
echo.
echo ============================================================
pause
