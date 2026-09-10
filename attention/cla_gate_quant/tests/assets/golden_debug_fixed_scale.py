#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Debug golden for CLA gate fixed-scale (scale=1) isolation.

The kernel debug path bypasses MX scale computation:
  row_data / col_data = FP8(merged)
  row_scale / col_scale = E8M0(1.0)

Use with run_ttk_kernel.sh:
  TTK_GOLDEN_PLUGIN=.../golden_debug_fixed_scale.py bash .../run_ttk_kernel.sh
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
        "cla_gate_quant": "cla_gate_quant_debug_fixed_scale_golden"
    }
}


def _fp8():
    return _MODULE._numpy_mx_dtype("float8_e4m3fn")


def _e8m0():
    return _MODULE._numpy_mx_dtype("float8_e8m0")


def cla_gate_quant_debug_fixed_scale_golden(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=36,
    scale_alg=1,
    **kwargs,
):
    """Return fixed-scale FP8 cast of merged plus scale=1 outputs."""
    merged = _MODULE.cla_gate_np(global_out, local_out, global_gate_logits, local_gate_logits)
    merged = merged.reshape(merged.shape[0] * merged.shape[1],
                            merged.shape[2] * merged.shape[3])
    merged_f32 = merged.astype(np.float32)

    row_data = merged_f32.astype(_fp8())
    col_data = merged_f32.astype(_fp8())

    row_scale = np.full((merged.shape[0], 384, 2), 1.0, dtype=np.float32).astype(_e8m0())
    col_scale = np.full((16, merged.shape[1], 2), 1.0, dtype=np.float32).astype(_e8m0())

    return row_data, row_scale, col_data, col_scale
