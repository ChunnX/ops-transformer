#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MoeDistributeDispatch comparison policy from moe_A3."""

import importlib.util
from pathlib import Path


def _row_count(value):
    return int(getattr(value, "shape", (0,))[0])


def _valid_prefix(value, count):
    if value is None or count <= 0:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu()[:count]
    return value[:count]


def _load_module(relative_path, name):
    path = Path(__file__).resolve().parents[4] / relative_path
    spec = importlib.util.spec_from_file_location(
        f"{name}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_compare_module():
    return _load_module("common/tests/assets/mc2_compare.py", "mc2_common_compare")


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

    compare_context = kwargs.get("compare_context")
    valid_rows = int(
        getattr(compare_context, "attributes", {}).get("_moeValidExpandRows", 0)
    )
    output = _valid_prefix(output, valid_rows)
    golden = _valid_prefix(golden, valid_rows)
    if output is None or golden is None:
        return {"pass": True, "precision": 100.0}

    is_int8 = str(getattr(output, "dtype", "")) in ("int8", "torch.int8")
    return module.moe_data_compare_np(
        output,
        golden,
        diff_thd=0.001,
        pct_thd=0.001,
        max_diff_hd=1.0 if is_int8 else 0.0,
    )
