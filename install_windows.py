"""
Windows Installation Script for LLM Benchmarking Framework

This script handles the installation of dependencies with workarounds for
Windows-specific issues, particularly with sentencepiece on Python 3.13.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a pip command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during: {description}")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Main installation process."""
    print("="*60)
    print("LLM Benchmarking Framework - Windows Installation")
    print("="*60)
    
    # Check Python version
    py_version = sys.version_info
    print(f"\nPython version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version < (3, 9):
        print("✗ Error: Python 3.9 or higher required")
        sys.exit(1)
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("\n⚠ Warning: Not in a virtual environment")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled. Create a venv with:")
            print("  python -m venv .venv")
            print("  .venv\\Scripts\\activate")
            sys.exit(0)
    
    steps = [
        # Step 1: Upgrade pip
        (
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip"
        ),
        
        # Step 2: Install sentencepiece first (with wheels for Python 3.13)
        (
            f"{sys.executable} -m pip install sentencepiece>=0.2.0",
            "Installing sentencepiece (with pre-built wheels)"
        ),
        
        # Step 3: Install promptbench without dependencies to avoid sentencepiece conflict
        (
            f"{sys.executable} -m pip install promptbench --no-deps",
            "Installing promptbench (without dependencies)"
        ),
        
        # Step 4: Install core dependencies
        (
            f"{sys.executable} -m pip install torch>=2.0.0 transformers>=4.30.0 accelerate>=0.20.0",
            "Installing PyTorch and Transformers"
        ),
        
        # Step 5: Install API clients
        (
            f"{sys.executable} -m pip install openai>=1.0.0 anthropic>=0.18.0",
            "Installing API clients (OpenAI, Anthropic)"
        ),
        
        # Step 6: Install data handling packages
        (
            f"{sys.executable} -m pip install pandas>=2.0.0 numpy>=1.24.0 datasets>=2.14.0 huggingface-hub>=0.17.0",
            "Installing data handling packages"
        ),
        
        # Step 7: Install configuration packages
        (
            f"{sys.executable} -m pip install pyyaml>=6.0 python-dotenv>=1.0.0",
            "Installing configuration packages"
        ),
        
        # Step 8: Install metrics and evaluation
        (
            f"{sys.executable} -m pip install scikit-learn>=1.3.0 scipy>=1.11.0",
            "Installing metrics and evaluation packages"
        ),
        
        # Step 9: Install logging and visualization
        (
            f"{sys.executable} -m pip install tqdm>=4.65.0 matplotlib>=3.7.0 seaborn>=0.12.0 tensorboard>=2.13.0",
            "Installing logging and visualization packages"
        ),
        
        # Step 10: Install Jupyter support
        (
            f"{sys.executable} -m pip install jupyter>=1.0.0 ipykernel>=6.25.0",
            "Installing Jupyter support"
        ),
    ]
    
    failed_steps = []
    
    for command, description in steps:
        if not run_command(command, description):
            failed_steps.append(description)
            response = input("\n⚠ Step failed. Continue with remaining steps? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    
    if not failed_steps:
        print("✓ All packages installed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Configure models in config/models.yaml")
        print("3. Configure datasets in config/datasets.yaml")
        print("4. Run: python experiments/run_benchmark.py")
    else:
        print("✗ Some installations failed:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nYou can try installing failed packages manually:")
        print("  pip install <package-name>")
    
    # Optional packages
    print("\n" + "="*60)
    print("Optional: Local Model Support")
    print("="*60)
    print("\nFor local model execution, you can install:")
    print("  • bitsandbytes (quantization) - Requires CUDA")
    print("  • llama-cpp-python (GGUF models)")
    print("  • vllm (high-performance serving) - Requires CUDA")
    print("\nInstall with:")
    print("  pip install bitsandbytes>=0.41.0  # For quantization")
    print("  pip install llama-cpp-python>=0.2.0  # For GGUF models")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
