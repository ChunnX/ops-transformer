#!/usr/bin/python
# -*- coding: utf-8 -*-
"""AllGatherMatmul comparison policy from mc2_test_0416."""

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
    policies = (
        {"threshold": 0.005, "ratio_threshold": 0.005, "max_relative_error": 2.0},
        {"threshold": 0.000001, "ratio_threshold": 0.000001, "max_relative_error": 0.0},
    )
    results = []
    for index, (output, golden) in enumerate(zip(outputs[:half], outputs[half:])):
        results.append(module.data_compare_np(output, golden, **policies[index % len(policies)]))
    return results
