#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MatmulAllReduceAddRmsNorm comparison policy from mc2_test."""

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
    spec = importlib.util.spec_from_file_location("mc2_common_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(*outputs, third_party=None, **kwargs):
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    module = _load_compare_module()
    results = []
    for index, output in enumerate(outputs[:half]):
        benchmark = third_party[index] if index < len(third_party or ()) else None
        if benchmark is None:
            results.append(
                {
                    "pass": False,
                    "precision": "third_party_unavailable",
                    "error_info": f"MatmulAllReduceAddRmsNorm third_party output[{index}] is unavailable",
                }
            )
            continue
        results.append(module.data_compare_np(output, benchmark))
    return results
