#!/usr/bin/env python3
"""
Prompt Template Generator
=========================
Generates experiment-specific prompt templates by combining:
  - config/chat_templates.yaml        (model-family chat format)
  - config/prompt_technique_template.yaml  (technique structure)
  - config/few_shot_samples.yaml       (few-shot examples + reasoning)
  - config/domain_experts.yaml         (expert personas for role_expert)
  - config/dataset_instructions.yaml   (task instructions per dataset)

Output:
  - config/generated_prompts.yaml      (125 templates)
  - config/model_template_map.yaml     (model-name → template-name mapping)

Naming convention:
    {dataset}_{technique}_{model_family}
    e.g.  sst2_zero_shot_smollm
          arc_challenge_few_shot_cot_qwen2_5

Usage:
    python -m utils.generate_prompt_templates [--config-dir CONFIG_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from jinja2 import BaseLoader, Environment, TemplateSyntaxError
except ImportError:
    sys.exit(
        "jinja2 is required.  Install with:  pip install jinja2"
    )


# ── constants ────────────────────────────────────────────────────────────────

DATASETS = ["sst2", "arc_challenge", "hellaswag", "gsm8k", "mmlu"]
TECHNIQUES = ["zero_shot", "few_shot", "chain_of_thought", "few_shot_cot", "role_expert"]

# Mapping from chat_templates.yaml key → short family name used in output names
TEMPLATE_KEY_TO_FAMILY = {
    "smollm_instruct": "smollm",
    "qwen2_5_instruct": "qwen2_5",
    "tinyllama_chat": "tinyllama",
    "phi3_instruct": "phi3",
    "mistral_7b_instruct_v0_3": "mistral",
}

# Default system prompts per family (when technique != role_expert)
DEFAULT_SYSTEM_PROMPTS = {
    "smollm": None,  # SmolLM auto-injects its own system prompt
    "qwen2_5": "You are a helpful assistant.",
    "tinyllama": "You are a helpful assistant.",
    "phi3": "You are a helpful assistant.",
    "mistral": "You are a helpful assistant.",
}


# ── config loading ───────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(config_dir: Path) -> dict:
    """Load all five source config files and return a dict of parsed data."""
    return {
        "chat_templates": _load_yaml(config_dir / "chat_templates.yaml").get(
            "model_chat_templates", {}
        ),
        "technique_templates": _load_yaml(
            config_dir / "prompt_technique_template.yaml"
        ).get("prompt_techniques", {}),
        "few_shot_samples": _load_yaml(config_dir / "few_shot_samples.yaml").get(
            "few_shot_examples", {}
        ),
        "domain_experts": _load_yaml(config_dir / "domain_experts.yaml").get(
            "domain_experts", {}
        ),
        "dataset_instructions": _load_yaml(
            config_dir / "dataset_instructions.yaml"
        ).get("dataset_instructions", {}),
    }


# ── model → family mapping ──────────────────────────────────────────────────

def build_model_family_map(
    chat_templates: dict, models_yaml_path: Optional[Path] = None
) -> Dict[str, str]:
    """
    Build a mapping from model *name* (as used in models.yaml / experiments.yaml)
    to the short family key (smollm, qwen2_5, tinyllama, phi3, mistral).

    Strategy: each chat template entry has an ``applies_to`` list of
    *display names*.  We lower-case / normalise both sides and match
    against the model names in models.yaml.
    """
    # Build display-name → family map from chat_templates
    display_to_family: Dict[str, str] = {}
    for tpl_key, tpl_data in chat_templates.items():
        family = TEMPLATE_KEY_TO_FAMILY.get(tpl_key)
        if family is None:
            continue
        for display_name in tpl_data.get("applies_to", []):
            display_to_family[display_name.lower()] = family

    # If models.yaml exists, load model names and match via model_id
    model_to_family: Dict[str, str] = {}
    if models_yaml_path and models_yaml_path.exists():
        models_cfg = _load_yaml(models_yaml_path)
        for model in models_cfg.get("models", []):
            name = model.get("name", "")
            model_id = model.get("model_id", "")
            # Try matching against display names in applies_to
            for display_lower, family in display_to_family.items():
                # Match if the display name appears in the model_id or name
                if display_lower in model_id.lower() or display_lower in name.lower():
                    model_to_family[name] = family
                    break
            else:
                # Heuristic fallback: match on known substrings
                id_lower = model_id.lower()
                name_lower = name.lower()
                if "smollm" in id_lower or "smollm" in name_lower:
                    model_to_family[name] = "smollm"
                elif "qwen2.5" in id_lower or "qwen2.5" in name_lower:
                    model_to_family[name] = "qwen2_5"
                elif "tinyllama" in id_lower or "tinyllama" in name_lower:
                    model_to_family[name] = "tinyllama"
                elif "phi-3" in id_lower or "phi-3" in name_lower:
                    model_to_family[name] = "phi3"
                elif "mistral" in id_lower or "mistral" in name_lower:
                    model_to_family[name] = "mistral"

    return model_to_family


# ── example formatting helpers ───────────────────────────────────────────────

def _format_example_input(dataset_key: str, example: dict) -> str:
    """Format a single example's *input* field for template rendering.

    For MCQ datasets the input includes the options block.
    """
    if dataset_key == "sst2":
        return example.get("input", "")
    elif dataset_key in ("arc_challenge", "mmlu"):
        parts = [example.get("question", "")]
        for key in ("A", "B", "C", "D"):
            opt = example.get("options", {}).get(key)
            if opt:
                parts.append(f"{key}: {opt}")
        return "\n".join(parts)
    elif dataset_key == "hellaswag":
        parts = [example.get("context", "")]
        for key in ("A", "B", "C", "D"):
            opt = example.get("options", {}).get(key)
            if opt:
                parts.append(f"{key}: {opt}")
        return "\n".join(parts)
    elif dataset_key == "gsm8k":
        return example.get("question", "")
    return str(example)


def _get_example_answer(dataset_key: str, example: dict) -> str:
    """Return the answer string from an example dict."""
    if dataset_key == "sst2":
        return example.get("output", "")
    return example.get("answer", "")


def prepare_examples(
    dataset_key: str, raw_examples: list, include_reasoning: bool = False
) -> List[Dict[str, str]]:
    """Prepare examples into a list of dicts with ``input``, ``answer``,
    and optionally ``reasoning`` keys expected by the technique templates.
    """
    prepared = []
    for ex in raw_examples:
        item = {
            "input": _format_example_input(dataset_key, ex),
            "answer": _get_example_answer(dataset_key, ex),
        }
        if include_reasoning:
            item["reasoning"] = ex.get("reasoning", "")
        prepared.append(item)
    return prepared


# ── inner prompt building ────────────────────────────────────────────────────

_JINJA_ENV = Environment(loader=BaseLoader(), keep_trailing_newline=True)


def build_inner_prompt(
    technique_key: str,
    dataset_key: str,
    technique_templates: dict,
    dataset_instructions: dict,
    few_shot_samples: dict,
    domain_experts: dict,
) -> str:
    """Render the technique template with dataset-specific content.

    The runtime input placeholder is ``$input`` (kept literal).
    """
    tpl_data = technique_templates[technique_key]
    tpl_string = tpl_data["template"]

    # Build the full instruction line (instruction + answer_format)
    ds_instr = dataset_instructions[dataset_key]
    instruction = ds_instr["instruction"].strip()
    answer_fmt = ds_instr.get("answer_format", "").strip()
    full_instruction = f"{instruction} {answer_fmt}".strip()

    # Prepare examples for few-shot variants
    examples: list = []
    if technique_key in ("few_shot", "few_shot_cot"):
        raw = few_shot_samples.get(dataset_key, {}).get("examples", [])
        examples = prepare_examples(
            dataset_key, raw, include_reasoning=(technique_key == "few_shot_cot")
        )

    # Domain string for role_expert
    domain = ""
    if technique_key == "role_expert":
        domain = domain_experts.get(dataset_key, {}).get("domain", "")

    # Render the Jinja2 technique template
    template = _JINJA_ENV.from_string(tpl_string)
    rendered = template.render(
        instruction=full_instruction,
        input="$input",          # literal placeholder for runtime
        examples=examples,
        domain=domain,
    )

    # Clean up excessive blank lines while preserving structure
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered


# ── chat template rendering ─────────────────────────────────────────────────

def _make_safe_jinja_env() -> Environment:
    """Create a Jinja2 env with a no-op ``raise_exception`` global
    (used by the Mistral template) and the ``do`` extension."""
    env = Environment(
        loader=BaseLoader(),
        keep_trailing_newline=True,
        extensions=["jinja2.ext.do"],
    )

    def _raise_exception(msg: str = ""):
        raise ValueError(msg)

    env.globals["raise_exception"] = _raise_exception
    return env


_CHAT_JINJA_ENV = _make_safe_jinja_env()


def render_chat_template(
    user_content: str,
    system_content: Optional[str],
    chat_template_str: str,
    family_key: str = "",
) -> str:
    """Render a messages list through a model-family Jinja2 chat template.

    Returns the fully formatted prompt string (with ``$input`` still
    present as a literal placeholder).
    """
    messages: List[Dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    # ChatML / Phi-3 templates need add_generation_prompt=True to emit
    # the assistant turn marker.  Llama / Mistral [INST] templates already
    # end with [/INST] which signals generation;  setting True would inject
    # a spurious opening [INST] tag.
    add_gen = family_key in ("smollm", "qwen2_5", "phi3")

    try:
        template = _CHAT_JINJA_ENV.from_string(chat_template_str)
    except TemplateSyntaxError as exc:
        raise RuntimeError(f"Chat template syntax error: {exc}") from exc

    rendered = template.render(
        messages=messages,
        add_generation_prompt=add_gen,
        bos_token="<s>",
        eos_token="</s>",
    )
    return rendered.rstrip()


# ── main generator ───────────────────────────────────────────────────────────

def generate_all_templates(config_dir: Path, output_dir: Path) -> dict:
    """Generate all (dataset × technique × model_family) prompt templates.

    Returns the generated template dict and writes YAML output files.
    """
    cfg = load_configs(config_dir)

    chat_templates = cfg["chat_templates"]
    technique_templates = cfg["technique_templates"]
    few_shot_samples = cfg["few_shot_samples"]
    domain_experts = cfg["domain_experts"]
    dataset_instructions = cfg["dataset_instructions"]

    # Build model → family mapping
    models_yaml = config_dir / "models.yaml"
    model_family_map = build_model_family_map(chat_templates, models_yaml)

    # Map family short name → chat template string
    family_to_tpl_str: Dict[str, str] = {}
    for tpl_key, tpl_data in chat_templates.items():
        family = TEMPLATE_KEY_TO_FAMILY.get(tpl_key)
        if family:
            family_to_tpl_str[family] = tpl_data["template"]

    generated: Dict[str, dict] = {}
    count = 0

    for dataset_key in DATASETS:
        if dataset_key not in dataset_instructions:
            print(f"  [skip] no instructions for dataset '{dataset_key}'")
            continue

        for technique_key in TECHNIQUES:
            if technique_key not in technique_templates:
                print(f"  [skip] no template for technique '{technique_key}'")
                continue

            # Build the inner (technique) prompt once per (dataset, technique)
            inner_prompt = build_inner_prompt(
                technique_key,
                dataset_key,
                technique_templates,
                dataset_instructions,
                few_shot_samples,
                domain_experts,
            )

            for family_key, chat_tpl_str in family_to_tpl_str.items():
                template_name = f"{dataset_key}_{technique_key}_{family_key}"

                # Determine system message
                if technique_key == "role_expert":
                    system_msg = domain_experts.get(dataset_key, {}).get(
                        "preamble", ""
                    ).strip()
                else:
                    system_msg = DEFAULT_SYSTEM_PROMPTS.get(family_key)

                # Render through the model chat template
                final_template = render_chat_template(
                    user_content=inner_prompt,
                    system_content=system_msg,
                    chat_template_str=chat_tpl_str,
                    family_key=family_key,
                )

                # Build the messages list (structured alternative)
                messages = []
                if system_msg:
                    messages.append({"role": "system", "content": system_msg})
                messages.append({"role": "user", "content": inner_prompt})

                generated[template_name] = {
                    "dataset": dataset_key,
                    "technique": technique_key,
                    "model_family": family_key,
                    "template": final_template,
                    "messages": messages,
                }
                count += 1

    # ── write outputs ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. generated_prompts.yaml
    prompts_path = output_dir / "generated_prompts.yaml"
    prompts_yaml: Dict[str, Any] = {"generated_prompts": {}}
    for name, data in generated.items():
        prompts_yaml["generated_prompts"][name] = {
            "dataset": data["dataset"],
            "technique": data["technique"],
            "model_family": data["model_family"],
            "template": data["template"],
            "messages": data["messages"],
        }
    with open(prompts_path, "w", encoding="utf-8") as f:
        yaml.dump(
            prompts_yaml,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=200,
        )
    print(f"  Wrote {count} templates → {prompts_path}")

    # 2. model_template_map.yaml  (model name → list of template names)
    map_path = output_dir / "model_template_map.yaml"
    model_map: Dict[str, Any] = {"model_template_map": {}}
    for model_name, family in sorted(model_family_map.items()):
        tpl_names = [
            f"{ds}_{tech}_{family}"
            for ds in DATASETS
            for tech in TECHNIQUES
            if f"{ds}_{tech}_{family}" in generated
        ]
        model_map["model_template_map"][model_name] = {
            "family": family,
            "templates": tpl_names,
        }
    with open(map_path, "w", encoding="utf-8") as f:
        yaml.dump(
            model_map,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=200,
        )
    print(f"  Wrote model mapping → {map_path}")

    # 3. prompting_techniques_generated.yaml (PromptManager-compatible format)
    pm_path = output_dir / "prompting_techniques_generated.yaml"
    pm_entries = []
    for name, data in generated.items():
        pm_entries.append(
            {
                "name": name,
                "type": "custom",
                "description": (
                    f"{data['technique'].replace('_', ' ').title()} "
                    f"for {data['dataset']} on {data['model_family']} models"
                ),
                "enabled": True,
                "template": data["template"],
                "params": {},
                "fields": {
                    "dataset": data["dataset"],
                    "technique": data["technique"],
                    "model_family": data["model_family"],
                },
            }
        )
    with open(pm_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"prompting_techniques": pm_entries},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=200,
        )
    print(f"  Wrote PromptManager-compatible entries → {pm_path}")

    print(f"\n✓ Generated {count} prompt templates "
          f"({len(DATASETS)} datasets × {len(TECHNIQUES)} techniques "
          f"× {len(family_to_tpl_str)} model families)")

    return generated


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment-specific prompt templates."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("./config"),
        help="Directory containing source YAML configs (default: ./config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./config"),
        help="Directory for generated output files (default: ./config)",
    )
    args = parser.parse_args()

    print(f"Config dir : {args.config_dir.resolve()}")
    print(f"Output dir : {args.output_dir.resolve()}")
    print()

    generate_all_templates(args.config_dir, args.output_dir)


if __name__ == "__main__":
    main()
