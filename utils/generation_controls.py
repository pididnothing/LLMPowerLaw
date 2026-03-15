"""Generation controls utility.

Centralizes inference-time generation settings so experiments can tune
reasoning budget and decoding behavior without touching model code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


SUPPORTED_KEYS = {
    "max_new_tokens",
    "min_new_tokens",
    "classification_max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "do_sample",
    "num_beams",
    "stop_sequences",
    "max_input_tokens",
}

PRESETS: Dict[str, Dict[str, Any]] = {
    # Deterministic + short outputs.
    "concise": {
        "max_new_tokens": 64,
        "temperature": 0.0,
    },
    # Good default for mixed tasks.
    "balanced": {
        "max_new_tokens": 192,
        "temperature": 0.15,
        "top_p": 0.9,
        "repetition_penalty": 1.03,
    },
    # More budget for multi-step tasks.
    "reasoning": {
        "max_new_tokens": 512,
        "min_new_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
    },
    # Strongly constrained outputs for label tasks.
    "classification": {
        "classification_max_new_tokens": 20,
        "temperature": 0.0,
    },
}


def _filtered(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {}
    return {
        key: value
        for key, value in data.items()
        if key in SUPPORTED_KEYS and value is not None
    }


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


def _pick_flat_controls(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Allow direct model YAML keys like top_p/top_k without nested blocks."""
    flat_controls = {}
    for key in SUPPORTED_KEYS:
        if key in model_config and model_config[key] is not None:
            flat_controls[key] = model_config[key]
    return flat_controls


def resolve_generation_controls(
    model_config: Dict[str, Any],
    global_settings: Dict[str, Any],
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge generation controls by precedence.

    Precedence (lowest -> highest):
    1. preset defaults
    2. global_settings.generation_controls
    3. model_config.generation_controls + flat model keys
    4. runtime_overrides (CLI)
    """
    runtime_overrides = dict(runtime_overrides or {})

    global_controls = _filtered(global_settings.get("generation_controls", {}))
    model_controls = _filtered(model_config.get("generation_controls", {}))
    model_flat_controls = _filtered(_pick_flat_controls(model_config))

    preset_name = runtime_overrides.get("preset")
    if not preset_name:
        preset_name = model_config.get("generation_preset")
    if not preset_name:
        preset_name = global_settings.get("generation_controls", {}).get("default_preset")

    preset_controls = PRESETS.get(str(preset_name).lower(), {}) if preset_name else {}

    runtime_filtered = _filtered({
        key: value
        for key, value in runtime_overrides.items()
        if key != "preset"
    })

    controls = {}
    controls = _merge(controls, preset_controls)
    controls = _merge(controls, global_controls)
    controls = _merge(controls, model_flat_controls)
    controls = _merge(controls, model_controls)
    controls = _merge(controls, runtime_filtered)

    if preset_name and "preset" not in controls:
        controls["preset"] = str(preset_name)

    return controls


def apply_generation_controls(
    base_max_tokens: Optional[int],
    base_temperature: Optional[float],
    controls: Dict[str, Any],
    task_type: Optional[str] = None,
) -> Tuple[Optional[int], Optional[float], Dict[str, Any]]:
    """Compute final generation values and provider kwargs."""
    max_tokens = controls.get("max_new_tokens", base_max_tokens)
    temperature = controls.get("temperature", base_temperature)

    if task_type == "classification":
        class_cap = controls.get("classification_max_new_tokens")
        if class_cap is not None:
            if max_tokens is None:
                max_tokens = class_cap
            else:
                max_tokens = min(int(max_tokens), int(class_cap))

    provider_kwargs = {}
    for key in (
        "min_new_tokens",
        "top_p",
        "top_k",
        "repetition_penalty",
        "do_sample",
        "num_beams",
        "stop_sequences",
        "max_input_tokens",
    ):
        if key in controls and controls[key] is not None:
            provider_kwargs[key] = controls[key]

    return max_tokens, temperature, provider_kwargs


def for_promptbench(controls: Dict[str, Any]) -> Dict[str, Any]:
    """Return PromptBench-safe kwargs from merged controls."""
    pb = {}
    if controls.get("max_new_tokens") is not None:
        pb["max_new_tokens"] = controls["max_new_tokens"]
    if controls.get("temperature") is not None:
        pb["temperature"] = controls["temperature"]
    return pb


def format_controls(controls: Dict[str, Any]) -> str:
    """Compact, stable string for logs."""
    if not controls:
        return "{}"

    keys = [
        "preset",
        "max_new_tokens",
        "classification_max_new_tokens",
        "min_new_tokens",
        "temperature",
        "do_sample",
        "top_p",
        "top_k",
        "repetition_penalty",
        "num_beams",
        "max_input_tokens",
        "stop_sequences",
    ]

    ordered = []
    for key in keys:
        if key in controls:
            ordered.append(f"{key}={controls[key]}")

    for key in sorted(controls.keys()):
        if key not in keys:
            ordered.append(f"{key}={controls[key]}")

    return "{" + ", ".join(ordered) + "}"


def build_runtime_overrides_from_args(args: Any) -> Dict[str, Any]:
    """Create runtime override dict from argparse namespace."""
    overrides: Dict[str, Any] = {}

    if getattr(args, "gen_preset", None):
        overrides["preset"] = args.gen_preset

    for arg_name, key in (
        ("gen_max_new_tokens", "max_new_tokens"),
        ("gen_min_new_tokens", "min_new_tokens"),
        ("gen_classification_max_new_tokens", "classification_max_new_tokens"),
        ("gen_temperature", "temperature"),
        ("gen_top_p", "top_p"),
        ("gen_top_k", "top_k"),
        ("gen_repetition_penalty", "repetition_penalty"),
        ("gen_num_beams", "num_beams"),
        ("gen_max_input_tokens", "max_input_tokens"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[key] = value

    gen_do_sample = getattr(args, "gen_do_sample", None)
    if gen_do_sample is not None and gen_do_sample != "auto":
        overrides["do_sample"] = gen_do_sample == "true"

    stop_sequences = getattr(args, "gen_stop_sequences", None)
    if stop_sequences:
        parsed = [item.strip() for item in stop_sequences.split("||") if item.strip()]
        if parsed:
            overrides["stop_sequences"] = parsed

    return overrides
