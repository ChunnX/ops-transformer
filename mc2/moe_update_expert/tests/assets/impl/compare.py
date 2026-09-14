#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MoeUpdateExpert comparison policy ported from test_eplb.py."""

import importlib.util
from pathlib import Path


def _load_compare_module():
    path = Path(__file__).resolve().parents[4] / "common" / "tests" / "assets" / "mc2_compare.py"
    spec = importlib.util.spec_from_file_location(f"mc2_common_compare_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(*outputs, **kwargs):
    if len(outputs) != 4:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "compare expects balancedExpertIds/balancedActiveMask outputs and goldens",
        }
    module = _load_compare_module()
    results = [
        module.eplb_data_compare_np(output, golden)
        for output, golden in zip(outputs[:2], outputs[2:])
    ]
    return results
