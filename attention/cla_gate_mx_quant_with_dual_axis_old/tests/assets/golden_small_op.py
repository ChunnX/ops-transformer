#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Torch small-op concatenated golden for cla_gate_mx_quant_with_dual_axis.

This golden mirrors the small-op benchmark path:
    torch.sigmoid(gate_logits).unsqueeze(-1) -> broadcast mul -> add
    -> reshape [S*B, N*D]
    -> dual-axis MXFP8 quant (E4M3FN).

The quantization part reuses the CPU implementation from golden.py; only the
merge stage is expressed with torch operations, matching the small-op script
more closely than the pure numpy golden.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np

# Avoid auto-loading torch_npu inside TTK's CPU golden generation process.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402

_CPU_GOLDEN = Path(__file__).with_name("golden.py")
_SPEC = importlib.util.spec_from_file_location("cla_gate_cpu_golden", _CPU_GOLDEN)
_CPU = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CPU)

__golden__ = {
    "kernel": {
        "cla_gate_mx_quant_with_dual_axis": "cla_gate_mx_quant_with_dual_axis_small_op_golden"
    }
}


def _numpy_to_torch(array):
    if array.dtype == np.float16:
        return torch.from_numpy(array)
    # torch.from_numpy does not accept ml_dtypes.bfloat16. BF16 -> FP32 -> BF16
    # is exact for every BF16 value, so this is a lossless conversion.
    return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)


def _torch_to_numpy(tensor, dtype):
    if dtype == np.float16:
        return tensor.numpy()
    return tensor.float().numpy().astype(dtype)


def cla_gate_small_op_np(global_out, local_out, global_gate_logits, local_gate_logits):
    """Torch small-op merge, returns numpy array in the original dtype."""
    calc_dtype = global_out.dtype
    go = _numpy_to_torch(global_out)
    lo = _numpy_to_torch(local_out)
    gg = _numpy_to_torch(global_gate_logits)
    lg = _numpy_to_torch(local_gate_logits)
    if gg.dim() == 4:
        gg = gg.squeeze(-1)
        lg = lg.squeeze(-1)

    global_scale = torch.sigmoid(gg).unsqueeze(-1)
    local_scale = torch.sigmoid(lg).unsqueeze(-1)
    merged = go * global_scale + lo * local_scale
    merged = merged.reshape(merged.shape[0] * merged.shape[1], merged.shape[2] * merged.shape[3])
    return _torch_to_numpy(merged, calc_dtype)


def cla_gate_mx_quant_with_dual_axis_small_op_golden(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=36,
    scale_alg=1,
    **kwargs,
):
    """Small-op concatenated golden for the fixed CLA gate forward shape."""
    if dst_type != 36:
        raise ValueError("cla_gate_mx_quant_with_dual_axis only supports dst_type=36")

    merged = cla_gate_small_op_np(
        global_out, local_out, global_gate_logits, local_gate_logits
    )
    row_scale, row_data = _CPU._mx_quantize(
        merged,
        mx_ele_dtype="float8_e4m3fn",
        axis=-1,
        block_size=32,
        round_mode="rint",
        scale_alg=scale_alg,
    )
    col_scale, col_data = _CPU._mx_quantize(
        merged,
        mx_ele_dtype="float8_e4m3fn",
        axis=-2,
        block_size=32,
        round_mode="rint",
        scale_alg=scale_alg,
    )
    return row_data, row_scale, col_data, col_scale
