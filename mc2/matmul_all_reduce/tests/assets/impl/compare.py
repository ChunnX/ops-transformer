#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MatmulAllReduce three-party comparison policy from mc2_test."""

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
    for output, golden in zip(outputs[:half], outputs[half:]):
        benchmark = third_party[0] if third_party else None
        if benchmark is None:
            results.append(
                {
                    "pass": False,
                    "precision": "third_party_unavailable",
                    "error_info": "MatmulAllReduce third_party output is unavailable",
                }
            )
            continue
        results.append(module.tripartite_compare(output, golden, benchmark))
    return results
