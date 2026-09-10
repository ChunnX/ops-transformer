#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Torch small-op golden for cla_gate_quant (NPUSim).

This plugin mirrors the small-op reference on CPU:
    torch.sigmoid(gate_logits).unsqueeze(-1) -> broadcast mul -> add
    -> reshape [T, N*D] (current TND API; legacy SBND is also supported)
    -> dual-axis MX quantization using the NumPy implementation in golden.py.
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
        "cla_gate_quant": "cla_gate_quant_small_op_golden"
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
    orig_dtype = go.dtype
    if gg.dim() == go.dim():
        # [T,N,1] (or legacy [S,B,N,1]) -> squeeze the singleton head/feature dim.
        gg = gg.squeeze(-1)
        lg = lg.squeeze(-1)

    global_scale = torch.sigmoid(gg.to(torch.float32)).unsqueeze(-1)
    local_scale = torch.sigmoid(lg.to(torch.float32)).unsqueeze(-1)
    merged = (go.to(torch.float32) * global_scale + lo.to(torch.float32) * local_scale).to(orig_dtype)
    if merged.dim() == 3:
        # Current TND layout: [T, N, D] -> [T, N*D]
        merged = merged.reshape(merged.shape[0], merged.shape[1] * merged.shape[2])
    elif merged.dim() == 4:
        # Legacy SBND layout: [S, B, N, D] -> [S*B, N*D]
        merged = merged.reshape(merged.shape[0] * merged.shape[1], merged.shape[2] * merged.shape[3])
    else:
        raise ValueError("cla_gate_quant expects 3D [T,N,D] or 4D [S,B,N,D]")
    return _torch_to_numpy(merged, calc_dtype)


def cla_gate_quant_small_op_golden(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=36,
    scale_alg=1,
    round_mode="rint",
    **kwargs,
):
    """Small-op concatenated golden for the TND CLA gate forward operator."""
    mx_ele_dtype = {
        35: "float8_e5m2",
        36: "float8_e4m3fn",
        40: "float4_e2m1",
        41: "float4_e1m2",
    }.get(dst_type)
    if mx_ele_dtype is None:
        raise ValueError("unsupported dst_type: %s" % dst_type)

    merged = cla_gate_small_op_np(
        global_out, local_out, global_gate_logits, local_gate_logits
    )
    row_scale, row_data = _CPU._mx_quantize(
        merged,
        mx_ele_dtype=mx_ele_dtype,
        axis=-1,
        block_size=32,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )
    col_scale, col_data = _CPU._mx_quantize(
        merged,
        mx_ele_dtype=mx_ele_dtype,
        axis=-2,
        block_size=32,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )
    return row_data, row_scale, col_data, col_scale
