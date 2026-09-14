#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MC2 comparison policy ported from mc2_test new_compare.checkResult."""

import importlib.util
from pathlib import Path


def _load_compare_module():
    path = Path(__file__).resolve().parents[4] / "common" / "tests" / "assets" / "mc2_compare.py"
    spec = importlib.util.spec_from_file_location(f"mc2_common_compare_{abs(hash(path))}", path)
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
    benchmark = third_party[0] if third_party else None
    if benchmark is None:
        return [
            {
                "pass": False,
                "precision": "third_party_unavailable",
                "error_info": "output[0] HCCL cascade benchmark is unavailable",
            }
        ]
    result = module.tripartite_compare(outputs[0], outputs[half], benchmark, moe=True)
    if not result["pass"]:
        print(
            f"MC2 compare output[0] metrics: "
            f"standard={result['metrics']['standard']}, "
            f"target={result['metrics']['target']}, "
            f"third_party={result['metrics']['third_party']}, "
            f"reason={result['metrics']['reason']}",
            flush=True,
        )
    return [result]
