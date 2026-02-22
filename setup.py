"""
Quick Setup Script
Run this to verify installation and create example datasets
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'yaml',
        'pandas',
        'numpy',
        'dotenv',
        'torch',
        'transformers',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing packages:", ", ".join(missing))
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All core dependencies installed")
        return True

def check_promptbench():
    """Check if PromptBench is installed"""
    try:
        import promptbench
        print("✓ PromptBench installed")
        return True
    except ImportError:
        print("⚠ PromptBench not installed")
        print("  Install with: pip install promptbench")
        print("  (Optional - you can still use custom datasets)")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'HuggingFace': os.getenv('HUGGINGFACE_TOKEN'),
        'Google': os.getenv('GOOGLE_API_KEY'),
    }
    
    configured = {k: bool(v) for k, v in keys.items()}
    
    print("\nAPI Keys:")
    for provider, is_set in configured.items():
        status = "✓" if is_set else "✗"
        print(f"  {status} {provider}")
    
    if not any(configured.values()):
        print("\n⚠ No API keys configured")
        print("  Copy .env.example to .env and add your keys")
    
    return any(configured.values())

def create_example_datasets():
    """Create example datasets"""
    from datasets.custom_loader import DatasetLoader
    
    print("\nCreating example datasets...")
    try:
        DatasetLoader.create_example_datasets()
        print("✓ Example datasets created in datasets/data/")
        return True
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return False

def verify_config():
    """Verify configuration files"""
    config_dir = project_root / "config"
    
    required_configs = ['models.yaml', 'datasets.yaml']
    
    print("\nConfiguration Files:")
    all_exist = True
    for config_file in required_configs:
        path = config_dir / config_file
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {config_file}")
        if not exists:
            all_exist = False
    
    return all_exist

def test_config_loading():
    """Test loading configurations"""
    try:
        from experiments.experiment_config import ExperimentConfig
        
        print("\nTesting configuration loading...")
        config = ExperimentConfig()
        config.load_configs()
        
        enabled_models = config.get_enabled_models()
        enabled_datasets = config.get_enabled_datasets()
        
        print(f"✓ Configuration loaded successfully")
        print(f"  Enabled models: {len(enabled_models)}")
        print(f"  Enabled datasets: {len(enabled_datasets)}")
        
        if len(enabled_models) > 0 and len(enabled_datasets) > 0:
            print(f"\n  Ready to run {len(enabled_models) * len(enabled_datasets)} experiments")
        
        return True
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def main():
    print("=" * 60)
    print("LLM POWER LAW - SETUP VERIFICATION")
    print("=" * 60)
    
    checks = []
    
    # Check dependencies
    print("\n1. Checking Dependencies...")
    checks.append(check_dependencies())
    
    # Check PromptBench
    print("\n2. Checking PromptBench...")
    check_promptbench()  # Optional
    
    # Check API keys
    print("\n3. Checking API Keys...")
    check_api_keys()  # Not required for setup
    
    # Verify configs
    print("\n4. Verifying Configuration Files...")
    checks.append(verify_config())
    
    # Create example datasets
    print("\n5. Creating Example Datasets...")
    checks.append(create_example_datasets())
    
    # Test config loading
    print("\n6. Testing Configuration Loading...")
    checks.append(test_config_loading())
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if all(checks):
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit config/models.yaml to enable models")
        print("2. Edit config/datasets.yaml to enable datasets")
        print("3. Add your API keys to .env file")
        print("4. Run: python experiments/run_benchmark.py")
    else:
        print("⚠ Some setup steps need attention")
        print("Please review the messages above")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
