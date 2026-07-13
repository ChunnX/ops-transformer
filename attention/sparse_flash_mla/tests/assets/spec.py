#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TestSpec adapter for SparseFlashMla TTK assets."""

import importlib.util
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"smla_ttk_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")


class SparseFlashMlaSpec:
    golden = golden_module.cpu_sparse_flash_mla
    customize_inputs = inputs_module.generate_sparse_flash_mla_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


__spec__ = {
    "smla_ttk_ops.sparse_flash_mla_ttk": "SparseFlashMlaSpec",
}
