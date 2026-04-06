"""Experiments module for running benchmarks"""

# ---------------------------------------------------------------------------
# Compatibility patch — applied at package import time
# ---------------------------------------------------------------------------
# The cached stale modeling_phi3.py (loaded via trust_remote_code=True) calls
# DynamicCache.from_legacy_cache() which was removed in transformers v5.x.
# We unconditionally restore it so every code path (shell subprocess AND
# Colab in-process importlib.reload) gets the fix without a runtime restart.
try:
    from transformers.cache_utils import DynamicCache as _DC

    @classmethod  # type: ignore[misc]
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx, layer_past in enumerate(past_key_values):
                cache.update(layer_past[0], layer_past[1], layer_idx)
        return cache

    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Shim: get_usable_length removed in transformers v5."""
        max_length = self.get_max_length() if hasattr(self, 'get_max_length') else None
        seq_length = self.get_seq_length(layer_idx) if hasattr(self, 'get_seq_length') else 0
        if max_length is not None and new_seq_length > max_length:
            return max_length - new_seq_length
        return seq_length

    _DC.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]
    _DC.get_usable_length = _get_usable_length   # type: ignore[attr-defined]
except Exception:
    pass  # transformers not installed yet — will be caught at model-load time


from .experiment_config import ExperimentConfig, ModelConfig, DatasetConfig
from .run_benchmark import BenchmarkRunner

__all__ = ['ExperimentConfig', 'ModelConfig', 'DatasetConfig', 'BenchmarkRunner']
