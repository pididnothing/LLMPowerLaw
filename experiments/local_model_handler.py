"""
Local Model Handler
Manages loading and inference for locally-run LLMs
Supports HuggingFace, GGUF, vLLM, and other local model formats
"""

import os
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path


class LocalModelHandler:
    """Handler for local model loading and inference"""
    
    def __init__(self, model_config: Dict[str, Any], global_settings: Dict[str, Any]):
        self.model_config = model_config
        self.global_settings = global_settings
        self.model = None
        self.tokenizer = None
        self.provider = model_config.get('provider')
        
    def load_model(self):
        """Load model based on provider type"""
        if self.provider == 'huggingface_local':
            return self._load_huggingface_model()
        elif self.provider == 'gguf':
            return self._load_gguf_model()
        elif self.provider == 'vllm':
            return self._load_vllm_model()
        else:
            raise ValueError(f"Unsupported local provider: {self.provider}")
    
    def _load_huggingface_model(self):
        """Load HuggingFace model locally"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for local HuggingFace models. "
                "Install with: pip install transformers"
            )
        
        model_id = self.model_config.get('model_id')
        local_path = self.model_config.get('local_model_path')
        
        # Use local path if provided, otherwise download from HF
        model_path = local_path if local_path else model_id
        
        print(f"Loading model: {model_path}")
        
        # Prepare loading kwargs
        load_kwargs = self._prepare_hf_load_kwargs()
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.get('trust_remote_code', False),
            cache_dir=self.global_settings.get('local_models', {}).get('hf_cache_dir')
        )
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"Loading model with config: {load_kwargs}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        print(f"Model loaded successfully on device: {self.model.device}")
        
        return self.model, self.tokenizer
    
    def _prepare_hf_load_kwargs(self) -> Dict[str, Any]:
        """Prepare kwargs for HuggingFace model loading"""
        kwargs = {}
        
        # Device configuration
        device = self.model_config.get('device', 'auto')
        if device != 'auto':
            kwargs['device_map'] = device
        else:
            kwargs['device_map'] = 'auto'
        
        # Quantization settings
        load_in_8bit = self.model_config.get('load_in_8bit', False)
        load_in_4bit = self.model_config.get('load_in_4bit', False)
        
        if load_in_8bit:
            kwargs['load_in_8bit'] = True
            print("Using 8-bit quantization")
        elif load_in_4bit:
            kwargs['load_in_4bit'] = True
            print("Using 4-bit quantization")
            # Optionally add BitsAndBytes config for 4-bit
            try:
                from transformers import BitsAndBytesConfig
                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                print("Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")
        
        # Data type
        torch_dtype = self.model_config.get('torch_dtype', 'auto')
        if torch_dtype == 'float16':
            kwargs['torch_dtype'] = torch.float16
        elif torch_dtype == 'bfloat16':
            kwargs['torch_dtype'] = torch.bfloat16
        elif torch_dtype == 'float32':
            kwargs['torch_dtype'] = torch.float32
        elif torch_dtype == 'auto':
            kwargs['torch_dtype'] = 'auto'
        
        # Trust remote code
        kwargs['trust_remote_code'] = self.model_config.get('trust_remote_code', False)
        
        # Cache directory
        cache_dir = self.global_settings.get('local_models', {}).get('hf_cache_dir')
        if cache_dir:
            kwargs['cache_dir'] = cache_dir
        
        # Max memory
        max_memory = self.global_settings.get('local_models', {}).get('max_memory')
        if max_memory:
            kwargs['max_memory'] = max_memory
        
        # Offline mode
        if self.global_settings.get('local_models', {}).get('offline_mode', False):
            kwargs['local_files_only'] = True
        
        return kwargs
    
    def _load_gguf_model(self):
        """Load GGUF model using llama-cpp-python or ctransformers"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python"
            )
        
        model_file = self.model_config.get('model_file')
        local_path = self.model_config.get('local_model_path')
        
        if local_path:
            model_path = local_path
        else:
            # Try to download from HuggingFace
            model_id = self.model_config.get('model_id')
            model_path = self._download_gguf_from_hf(model_id, model_file)
        
        print(f"Loading GGUF model: {model_path}")
        
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.model_config.get('n_ctx', 2048),
            n_gpu_layers=self.model_config.get('n_gpu_layers', 0),
            n_threads=self.global_settings.get('gguf', {}).get('n_threads', 4),
            use_mlock=self.global_settings.get('gguf', {}).get('use_mlock', False),
        )
        
        print("GGUF model loaded successfully")
        
        return self.model, None  # GGUF doesn't use separate tokenizer
    
    def _download_gguf_from_hf(self, repo_id: str, filename: str) -> Path:
        """Download GGUF file from HuggingFace"""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required. "
                "Install with: pip install huggingface_hub"
            )
        
        cache_dir = self.global_settings.get('cache_dir', './cache')
        
        print(f"Downloading GGUF file from {repo_id}/{filename}")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        
        return Path(model_path)
    
    def _load_vllm_model(self):
        """Load model using vLLM for high-performance serving"""
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vllm is required for vLLM models. "
                "Install with: pip install vllm"
            )
        
        model_id = self.model_config.get('model_id')
        local_path = self.model_config.get('local_model_path')
        
        model_path = local_path if local_path else model_id
        
        print(f"Loading vLLM model: {model_path}")
        
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=self.model_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.9),
            trust_remote_code=self.model_config.get('trust_remote_code', False),
        )
        
        print("vLLM model loaded successfully")
        
        return self.model, None  # vLLM has built-in tokenizer
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        if self.provider == 'huggingface_local':
            return self._generate_huggingface(prompt, max_tokens, temperature, **kwargs)
        elif self.provider == 'gguf':
            return self._generate_gguf(prompt, max_tokens, temperature, **kwargs)
        elif self.provider == 'vllm':
            return self._generate_vllm(prompt, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        **kwargs
    ) -> str:
        """Generate using HuggingFace model"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            'max_new_tokens': max_tokens or self.model_config.get('max_tokens', 512),
            'temperature': temperature if temperature is not None else self.model_config.get('temperature', 0.0),
            'do_sample': temperature is not None and temperature > 0,
            'pad_token_id': self.tokenizer.pad_token_id,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the generated tokens (exclude prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _generate_gguf(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        **kwargs
    ) -> str:
        """Generate using GGUF model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        response = self.model(
            prompt,
            max_tokens=max_tokens or self.model_config.get('max_tokens', 512),
            temperature=temperature if temperature is not None else self.model_config.get('temperature', 0.0),
            **kwargs
        )
        
        return response['choices'][0]['text'].strip()
    
    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        **kwargs
    ) -> str:
        """Generate using vLLM model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.model_config.get('max_tokens', 512),
            temperature=temperature if temperature is not None else self.model_config.get('temperature', 0.0),
            **kwargs
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        
        return outputs[0].outputs[0].text.strip()
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model unloaded and memory cleared")


def check_local_model_dependencies():
    """Check which local model libraries are available"""
    dependencies = {
        'transformers': False,
        'torch': False,
        'bitsandbytes': False,
        'llama-cpp-python': False,
        'vllm': False,
        'flash-attn': False,
    }
    
    try:
        import transformers
        dependencies['transformers'] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import bitsandbytes
        dependencies['bitsandbytes'] = True
    except ImportError:
        pass
    
    try:
        import llama_cpp
        dependencies['llama-cpp-python'] = True
    except ImportError:
        pass
    
    try:
        import vllm
        dependencies['vllm'] = True
    except ImportError:
        pass
    
    try:
        import flash_attn
        dependencies['flash-attn'] = True
    except ImportError:
        pass
    
    return dependencies


if __name__ == "__main__":
    print("Checking local model dependencies...")
    deps = check_local_model_dependencies()
    
    print("\nAvailable Libraries:")
    for lib, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {lib}")
    
    print("\nInstallation commands for missing libraries:")
    if not deps['transformers']:
        print("  pip install transformers")
    if not deps['torch']:
        print("  pip install torch")
    if not deps['bitsandbytes']:
        print("  pip install bitsandbytes")
    if not deps['llama-cpp-python']:
        print("  pip install llama-cpp-python")
    if not deps['vllm']:
        print("  pip install vllm")
    if not deps['flash-attn']:
        print("  pip install flash-attn")
