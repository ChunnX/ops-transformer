#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Golden for deterministic row-gate mapping test.

Kernel debug mode overwrites:
  global_out = 1
  local_out  = 0
  local_gate = -20
  global_gate(row) = (row % 64 - 32) * 0.25

So merged(row) = sigmoid(global_gate(row)).
Fixed scale=1 is used, so outputs are FP8(merged) and E8M0(1.0).
"""
from pathlib import Path
import importlib.util

import numpy as np

_IMPL = Path(__file__).with_name("golden.py")
_SPEC = importlib.util.spec_from_file_location("cla_gate_full_golden", _IMPL)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

__golden__ = {
    "kernel": {
        "cla_gate_mx_quant_with_dual_axis": "cla_gate_mx_quant_with_dual_axis_debug_mapping_golden"
    }
}


def _fp8():
    return _MODULE._numpy_mx_dtype("float8_e4m3fn")


def _e8m0():
    return _MODULE._numpy_mx_dtype("float8_e8m0")


def cla_gate_mx_quant_with_dual_axis_debug_mapping_golden(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=36,
    scale_alg=1,
    **kwargs,
):
    m = 1024
    n = 24576
    rows = np.arange(m, dtype=np.float32)
    gates = ((rows % 64) - 32.0) * 0.25
    sig = 1.0 / (1.0 + np.exp(-gates))  # [M]

    row_data = np.repeat(sig[:, None], n, axis=1).astype(np.float32).astype(_fp8())
    col_data = np.repeat(sig[:, None], n, axis=1).astype(np.float32).astype(_fp8())

    row_scale = np.full((m, 384, 2), 1.0, dtype=np.float32).astype(_e8m0())
    col_scale = np.full((16, n, 2), 1.0, dtype=np.float32).astype(_e8m0())
    return row_data, row_scale, col_data, col_scale
