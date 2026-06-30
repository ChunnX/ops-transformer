#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TestSpec adapter for SparseFlashAttention assets_v3."""

import importlib.util
from pathlib import Path


def _load_sibling(stem):
    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(f"sfa_assets_v3_{stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_golden = _load_sibling("_golden")
_inputs = _load_sibling("_inputs")


class SparseFlashAttentionSpec:
    golden = _golden.npu_sparse_flash_attention
    customize_inputs = _inputs.generate_valid_sparse_indices
    tolerance = {
        "float16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.005, "atol": 0.000025},
        "bfloat16": {"standard": "IsClose", "rtol": 0.0078125, "ptol": 0.005, "atol": 0.0001},
    }


__spec__ = {
    "torch_npu.npu_sparse_flash_attention": SparseFlashAttentionSpec,
}
