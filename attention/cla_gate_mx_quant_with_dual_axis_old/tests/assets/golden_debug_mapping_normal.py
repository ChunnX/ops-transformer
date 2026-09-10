#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Golden for normal MX scale on deterministic dual-path mapping.

Kernel debug mode overwrites:
  global_out = 1
  local_out  = 1
  global_gate(row) = (row % 64 - 32) * 0.25
  local_gate(row)  = (row % 64 - 32) * 0.25

So merged(row) = 2 * sigmoid(gate(row)).
This golden runs the normal dual-axis MX quantization on that deterministic merged.
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
        "cla_gate_mx_quant_with_dual_axis": "cla_gate_mx_quant_with_dual_axis_debug_mapping_normal_golden"
    }
}


def cla_gate_mx_quant_with_dual_axis_debug_mapping_normal_golden(
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
    merged = 2.0 * sig
    merged = np.repeat(merged[:, None], n, axis=1).astype(np.float32)

    row_scale, row_data = _MODULE._mx_quantize(
        merged,
        mx_ele_dtype="float8_e4m3fn",
        axis=-1,
        block_size=32,
        round_mode="rint",
        scale_alg=scale_alg,
    )
    col_scale, col_data = _MODULE._mx_quantize(
        merged,
        mx_ele_dtype="float8_e4m3fn",
        axis=-2,
        block_size=32,
        round_mode="rint",
        scale_alg=scale_alg,
    )
    return row_data, row_scale, col_data, col_scale
