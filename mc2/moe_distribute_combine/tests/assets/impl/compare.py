#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MoeDistributeCombine comparison policy from moe_A3 / moe_v3."""

import importlib.util
from pathlib import Path


def _load_compare_module():
    path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "mc2_compare.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"mc2_common_compare_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(*outputs, **kwargs):
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }

    module = _load_compare_module()
    output = outputs[0]
    golden = outputs[len(outputs) // 2]
    return module.moe_data_compare_np(
        output,
        golden,
        diff_thd=0.001,
        pct_thd=0.001,
        max_diff_hd=0.0,
    )
