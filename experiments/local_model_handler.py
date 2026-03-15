"""
Local Model Handler
Manages loading and inference for locally-run LLMs
Supports HuggingFace, GGUF, vLLM, and other local model formats
"""

import os
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
from tqdm import tqdm


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

    def load_tokenizer_only(self):
        """Load only the tokenizer for previewing HF chat templates."""
        if self.provider != 'huggingface_local':
            return None
        if self.tokenizer is not None:
            return self.tokenizer

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for local HuggingFace models. "
                "Install with: pip install transformers"
            )

        model_id = self.model_config.get('model_id')
        local_path = self.model_config.get('local_model_path')
        model_path = local_path if local_path else model_id

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_config.get('trust_remote_code', False),
            cache_dir=self.global_settings.get('local_models', {}).get('hf_cache_dir')
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
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
        
        # Load tokenizer with progress
        with tqdm(total=2, desc="Loading model components", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            pbar.set_description("Loading tokenizer")
            self.load_tokenizer_only()
            pbar.update(1)

            # Load model
            pbar.set_description(f"Loading model ({load_kwargs.get('load_in_4bit', False) and '4-bit' or load_kwargs.get('load_in_8bit', False) and '8-bit' or 'full precision'})")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            pbar.update(1)
        
        print(f"✓ Model loaded successfully on device: {self.model.device}")
        
        return self.model, self.tokenizer

    def format_prompt(
        self,
        prompt: str,
        use_hf_chat_template: bool = False,
        chat_messages: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Return the exact prompt text sent to the model."""
        if not use_hf_chat_template:
            return prompt

        tokenizer = self.load_tokenizer_only()
        if not hasattr(tokenizer, 'apply_chat_template'):
            return prompt

        messages = chat_messages or [{'role': 'user', 'content': prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            return prompt
    
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
        
        with tqdm(total=1, desc="Loading GGUF model", bar_format='{l_bar}{bar}| [{elapsed}<{remaining}]') as pbar:
            pbar.set_description(f"Loading GGUF model: {Path(model_path).name}")
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.model_config.get('n_ctx', 2048),
                n_gpu_layers=self.model_config.get('n_gpu_layers', 0),
                n_threads=self.global_settings.get('gguf', {}).get('n_threads', 4),
                use_mlock=self.global_settings.get('gguf', {}).get('use_mlock', False),
            )
            pbar.update(1)
        
        print("✓ GGUF model loaded successfully")
        
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
        
        with tqdm(total=1, desc="Loading vLLM model", bar_format='{l_bar}{bar}| [{elapsed}<{remaining}]') as pbar:
            pbar.set_description(f"Loading vLLM model: {model_path}")
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=self.model_config.get('tensor_parallel_size', 1),
                gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.9),
                trust_remote_code=self.model_config.get('trust_remote_code', False),
            )
            pbar.update(1)
        
        print("✓ vLLM model loaded successfully")
        
        return self.model, None  # vLLM has built-in tokenizer
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        # Pass task_type to generation methods for task-specific optimization
        if task_type:
            kwargs['task_type'] = task_type
        
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

        use_hf_chat_template = kwargs.pop('use_hf_chat_template', False)
        chat_messages = kwargs.pop('chat_messages', None)
        prompt = self.format_prompt(
            prompt,
            use_hf_chat_template=use_hf_chat_template,
            chat_messages=chat_messages
        )

        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get task type hint from kwargs if available
        task_type = kwargs.pop('task_type', None)
        
        # Adjust max_tokens for classification tasks (need fewer tokens)
        if task_type == 'classification' and max_tokens is None:
            max_tokens = 20  # Classification usually needs just 1-2 words
        
        # Stop sequences to prevent over-generation
        stop_sequences = kwargs.pop('stop_sequences', None)
        if stop_sequences is None:
            # Default stop sequences for classification and short answers
            stop_sequences = ['\n\n', '###', 'Question:', 'Text:', '\n\nText:', 'Answer:']
        
        # Encode stop sequences — only include sequences that map to a single token.
        # Multi-token sequences (e.g. '\n\n' → [newline, newline]) must NOT contribute
        # their first token as a stop ID; doing so would stop generation on any lone '\n',
        # which ChatML models emit right after <|im_start|>assistant, causing empty output.
        stop_token_ids = []
        if stop_sequences:
            for seq in stop_sequences:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                if len(tokens) == 1:
                    stop_token_ids.append(tokens[0])
        
        # Generation parameters
        gen_kwargs = {
            'max_new_tokens': max_tokens or self.model_config.get('max_tokens', 512),
            'temperature': temperature if temperature is not None else self.model_config.get('temperature', 0.0),
            'do_sample': temperature is not None and temperature > 0,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Add stop token IDs if available
        if stop_token_ids:
            # Use eos_token_id as base, add stop tokens
            gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id] + stop_token_ids if isinstance(self.tokenizer.eos_token_id, int) else stop_token_ids
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the generated tokens (exclude prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def extract_answer(self, response: str, task_type: str, label_space: list = None) -> dict:
        """Extract answer from model response based on task type
        
        Returns:
            dict with 'raw' and 'extracted' predictions
        """
        raw_response = response.strip()
        
        if task_type == 'classification':
            extracted = self._extract_classification_answer(raw_response, label_space or [])
        elif task_type == 'qa':
            extracted = self._extract_qa_answer(raw_response)
        else:
            extracted = raw_response
        
        return {
            'raw': raw_response,
            'extracted': extracted
        }
    
    def _extract_classification_answer(self, response: str, label_space: list = None) -> str:
        """Extract clean classification answer from potentially verbose output.

        Strategy:
        1. Strip common preamble prefixes ("Answer:", "Sentiment:", etc.)
        2. If label_space provided, scan the *full* (stripped) response for the
           first occurrence of any valid label (case-insensitive).  This handles
           verbose outputs like "The text is positive, as it expresses...".
        3. Fall back to the first-word heuristic if no valid label is found.
        """
        # Remove common prefixes
        response = response.strip()
        prefixes_to_remove = ['### Answer:', 'Answer:', 'Sentiment:', 'Classification:', '###']
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # If we know the valid label set, scan the response for the first match.
        if label_space:
            import re
            lower_response = response.lower()
            # Sort labels longest-first to avoid prefix-matching issues
            for label in sorted([str(l) for l in label_space], key=len, reverse=True):
                pattern = r'\b' + re.escape(label.lower()) + r'\b'
                if re.search(pattern, lower_response):
                    return label.lower()

        # Filter out responses that are just stop sequences
        stop_sequences = ['###', '\n\n', 'Question:', 'Text:']
        if response in stop_sequences or not response:
            return ''
        
        # Take only the first line
        first_line = response.split('\n')[0].strip()
        
        # Take only the first word if multiple words
        first_word = first_line.split()[0] if first_line.split() else first_line
        
        # Remove common punctuation
        first_word = first_word.rstrip('.,;:!?')
        
        # Normalize case
        first_word = first_word.lower()
        
        return first_word
    
    def _extract_qa_answer(self, response: str) -> str:
        """Extract answer from QA task response"""
        # Remove common QA prefixes
        response = response.strip()
        prefixes_to_remove = ['Answer:', 'A:', 'The answer is:', 'The answer is']
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # For QA, take the first sentence or line
        first_line = response.split('\n')[0].strip()
        
        # If still too long, take first sentence
        if '. ' in first_line:
            first_sentence = first_line.split('. ')[0] + '.'
            return first_sentence
        
        return first_line
    
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
