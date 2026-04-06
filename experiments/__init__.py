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

    _DC.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]
except Exception:
    pass  # transformers not installed yet — will be caught at model-load time

from .experiment_config import ExperimentConfig, ModelConfig, DatasetConfig
from .run_benchmark import BenchmarkRunner

__all__ = ['ExperimentConfig', 'ModelConfig', 'DatasetConfig', 'BenchmarkRunner']
