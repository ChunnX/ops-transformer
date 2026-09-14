#!/usr/bin/env python
"""Default MC2 comparison policy ported from mc2_test new_compare.checkResult."""

import importlib.util
from pathlib import Path


def _load_compare_module():
    path = Path(__file__).resolve().parents[1] / "mc2_compare.py"
    spec = importlib.util.spec_from_file_location("mc2_common_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(*outputs, third_party=None, **kwargs):
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {"pass": False, "precision": "invalid", "error_info": "expected NPU outputs followed by goldens"}
    half = len(outputs) // 2
    module = _load_compare_module()
    results = []
    for index, (output, golden) in enumerate(zip(outputs[:half], outputs[half:])):
        benchmark = third_party[index] if index < len(third_party or ()) else None
        if benchmark is None:
            results.append({"pass": False, "precision": "third_party_unavailable", "error_info": f"output[{index}] benchmark unavailable"})
            continue
        results.append(module.tripartite_compare(output, golden, benchmark))
    return results


def eplb_compare(*outputs, **kwargs):
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "expected NPU outputs followed by goldens",
        }
    half = len(outputs) // 2
    module = _load_compare_module()
    return [
        module.eplb_data_compare_np(output, golden)
        for output, golden in zip(outputs[:half], outputs[half:])
    ]
