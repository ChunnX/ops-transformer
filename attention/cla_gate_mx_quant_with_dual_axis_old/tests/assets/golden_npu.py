#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""NPU small-op golden for cla_gate_mx_quant_with_dual_axis.

This plugin runs the small-op reference path on the NPU itself:
    torch.sigmoid(gate).unsqueeze(-1) -> broadcast mul -> add
    -> reshape [S*B, N*D]
    -> torch_npu.npu_dynamic_mx_quant_with_dual_axis(E4M3FN, scale_alg)

Use it only with the real-device backend (--backend npu). It cannot be used
with --backend npusim because NPUSim golden generation runs on CPU.
"""

import os

import numpy as np

# The NPU golden intentionally loads torch_npu.
import torch
import torch_npu  # noqa: F401


def _float8_dtype():
    return getattr(torch, "float8_e4m3fn", None) or getattr(torch_npu, "float8_e4m3fn", None)


def _e8m0_numpy_dtype():
    from en_dtypes import float8_e8m0

    return float8_e8m0


def _e4m3_numpy_dtype():
    from ml_dtypes import float8_e4m3fn

    return float8_e4m3fn


def _to_npu(array, device):
    if array.dtype == np.float16:
        return torch.from_numpy(array).to(device)
    # torch.from_numpy does not accept ml_dtypes.bfloat16. BF16 -> FP32 -> BF16
    # is exact for every BF16 value, so this is lossless.
    return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16).to(device)


def _torch_uint8_bytes(tensor):
    return tensor.cpu().view(torch.uint8).numpy().tobytes()


def cla_gate_quant_dual_npu(global_out, local_out, global_gate_logits, local_gate_logits, *, scale_alg=1):
    device = "npu:0"
    torch.npu.set_device(device)

    go = _to_npu(global_out, device)
    lo = _to_npu(local_out, device)
    gg = _to_npu(global_gate_logits, device)
    lg = _to_npu(local_gate_logits, device)
    if gg.dim() == 4:
        gg = gg.squeeze(-1)
        lg = lg.squeeze(-1)

    global_scale = torch.sigmoid(gg).unsqueeze(-1)
    local_scale = torch.sigmoid(lg).unsqueeze(-1)
    merged = go * global_scale + lo * local_scale
    merged = merged.reshape(merged.shape[0] * merged.shape[1], merged.shape[2] * merged.shape[3])

    row_data, row_scale, col_data, col_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        merged, dst_type=_float8_dtype(), scale_alg=scale_alg
    )
    torch.npu.synchronize()

    row_data_np = np.frombuffer(_torch_uint8_bytes(row_data), dtype=_e4m3_numpy_dtype()).reshape(
        row_data.shape
    )
    col_data_np = np.frombuffer(_torch_uint8_bytes(col_data), dtype=_e4m3_numpy_dtype()).reshape(
        col_data.shape
    )
    row_scale_np = np.frombuffer(_torch_uint8_bytes(row_scale), dtype=_e8m0_numpy_dtype()).reshape(
        row_scale.shape
    )
    col_scale_np = np.frombuffer(_torch_uint8_bytes(col_scale), dtype=_e8m0_numpy_dtype()).reshape(
        col_scale.shape
    )
    return row_data_np, row_scale_np, col_data_np, col_scale_np


def cla_gate_mx_quant_with_dual_axis_npu_golden(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=36,
    scale_alg=1,
    **kwargs,
):
    """Same-device small-op reference for the CLA gate fused operator."""
    if dst_type != 36:
        raise ValueError("cla_gate_mx_quant_with_dual_axis only supports dst_type=36")
    return cla_gate_quant_dual_npu(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        scale_alg=scale_alg,
    )


__golden__ = {
    "kernel": {
        "cla_gate_mx_quant_with_dual_axis": "cla_gate_mx_quant_with_dual_axis_npu_golden"
    }
}
