# Quick Start Guide

## Installation Steps

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template
copy .env.example .env

# 4. Run setup verification
python setup.py
```

## Quick Test

```bash
# Create example datasets
python -m datasets.custom_loader

# Run benchmark with enabled models and datasets
python experiments/run_benchmark.py
```

## Configuration

Edit these files before running:

- `config/models.yaml` - Enable/disable models
- `config/datasets.yaml` - Enable/disable datasets
- `.env` - Add your API keys

## Common Commands

```bash
# Run all experiments
python experiments/run_benchmark.py

# Run specific model
python experiments/run_benchmark.py --model gpt-3.5-turbo

# Run specific dataset
python experiments/run_benchmark.py --dataset sst2

# Custom output directory
python experiments/run_benchmark.py --output-dir ./my_results

# View results
jupyter notebook notebooks/analysis.ipynb
```
