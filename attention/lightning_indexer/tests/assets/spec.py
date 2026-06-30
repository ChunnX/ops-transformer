#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TestSpec adapter for LightningIndexer assets_v3."""

import importlib.util
from pathlib import Path


def _load_sibling(stem):
    path = Path(__file__).with_name(f"{stem}.py")
    spec = importlib.util.spec_from_file_location(f"li_assets_v3_{stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_golden = _load_sibling("_golden")
_inputs = _load_sibling("_inputs")
_compare = _load_sibling("_compare")


class LightningIndexerSpec:
    golden = _golden.npu_lightning_indexer
    customize_inputs = _inputs.generate_li_inputs
    tolerance = {
        "float16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.05, "atol": 0.000025},
        "bfloat16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.05, "atol": 0.0001},
    }

    def compare(*outputs, **kwargs):
        return _compare.compare(*outputs, scores=_golden.get_topk_scores())


__spec__ = {
    "torch_npu.npu_lightning_indexer": LightningIndexerSpec,
}
