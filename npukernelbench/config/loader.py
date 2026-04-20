"""Load and merge configuration from YAML + CLI overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import yaml

from npukernelbench.cli import parse_args
from npukernelbench.config.models import FullConfig, ChatConfig, CompileConfig, EvalConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base* dict (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    config_path: Optional[str] = None,
    argv: Optional[List[str]] = None,
) -> FullConfig:
    """Load YAML config, apply CLI overrides, return an immutable FullConfig."""

    args = parse_args(argv)

    # Resolve config file path
    project_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = project_root / (config_path or args.config_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Extract nested sections
    chat_cfg = ChatConfig(**raw.pop("chat", {}))
    compile_cfg = CompileConfig(**raw.pop("compile", {}))
    eval_cfg = EvalConfig(**raw.pop("eval", {}))
    active_stages = list(args.stages)

    # Build base FullConfig from YAML
    config = FullConfig(
        **raw,
        chat=chat_cfg,
        compile=compile_cfg,
        eval=eval_cfg,
        active_stages=active_stages,
    )

    # --- Apply CLI overrides (mutable proxy, then rebuild frozen) ---
    overrides: dict = {}

    if args.run_dir:
        overrides["run_dir"] = args.run_dir
    if args.test_mode:
        overrides["static_shape_mode"] = args.test_mode == "static"
    if args.template_mode:
        overrides["kernel_only_mode"] = args.template_mode == "kernel_only"
    if args.n_sample is not None:
        overrides["n_sample"] = args.n_sample
    if args.n_case is not None:
        overrides["n_case"] = args.n_case
    if args.tiling_dump:
        overrides["tiling_dump"] = True
    if args.skip_op_build:
        overrides["skip_op_build"] = True

    # Chat flag
    if args.chat:
        chat_overrides = {"active": True}
    else:
        chat_overrides = {"active": False}
        # Only default to sample 0 if user didn't explicitly set n_sample
        if args.n_sample is None:
            overrides["n_sample"] = 1
        # In non-chat mode with full template, use cann-ops
        compile_overrides: dict = {}
        template_mode = overrides.get("kernel_only_mode", config.kernel_only_mode)
        if not template_mode:
            compile_overrides["mode"] = "cann-ops"
        if compile_overrides:
            overrides["compile"] = CompileConfig(**{
                **{f.name: getattr(config.compile, f.name)
                   for f in config.compile.__dataclass_fields__.values()},
                **compile_overrides,
            })

    if chat_overrides:
        overrides["chat"] = ChatConfig(**{
            **{f.name: getattr(config.chat, f.name)
               for f in config.chat.__dataclass_fields__.values()},
            **chat_overrides,
        })

    if not overrides:
        return config

    # Rebuild frozen config with overrides
    base_dict = {
        f.name: overrides.get(f.name, getattr(config, f.name))
        for f in config.__dataclass_fields__.values()
    }
    return FullConfig(**base_dict)
