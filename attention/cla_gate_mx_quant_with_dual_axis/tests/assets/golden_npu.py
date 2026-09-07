#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""NPU small-op golden for cla_gate_mx_quant_with_dual_axis.

This plugin runs the small-op reference path on the NPU itself:
    torch.sigmoid(gate).unsqueeze(-1) -> broadcast mul -> add
    -> reshape [T, N*D]
    -> torch_npu.npu_dynamic_mx_quant_with_dual_axis(...)

FP8 (E5M2/E4M3FN) uses the NPU small-op reference; FP4 falls back to the
CPU NumPy golden because torch_npu returns FP4 in a packed uint8 layout.

Use it only with the real-device backend (--backend npu). It cannot be used
with --backend npusim because NPUSim golden generation runs on CPU.
"""

import os

import numpy as np

# The NPU golden intentionally loads torch_npu.
import torch
import torch_npu  # noqa: F401


def _dst_torch_dtype(dst_type):
    if dst_type == 35:
        return getattr(torch, "float8_e5m2", None) or getattr(torch_npu, "float8_e5m2", None)
    if dst_type == 36:
        return getattr(torch, "float8_e4m3fn", None) or getattr(torch_npu, "float8_e4m3fn", None)
    raise ValueError("unsupported dst_type for NPU small-op: %s" % dst_type)


def _data_numpy_dtype(dst_type):
    if dst_type == 35:
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if dst_type == 36:
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if dst_type == 40:
        from en_dtypes import float4_e2m1

        return float4_e2m1
    if dst_type == 41:
        from en_dtypes import float4_e1m2

        return float4_e1m2
    raise ValueError("unsupported dst_type: %s" % dst_type)


def _e8m0_numpy_dtype():
    from en_dtypes import float8_e8m0

    return float8_e8m0


def _to_npu(array, device):
    if array.dtype == np.float16:
        return torch.from_numpy(array).to(device)
    # torch.from_numpy does not accept ml_dtypes.bfloat16. BF16 -> FP32 -> BF16
    # is exact for every BF16 value, so this is lossless.
    return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16).to(device)


def _torch_uint8_bytes(tensor):
    return tensor.cpu().view(torch.uint8).numpy().tobytes()


def _load_cpu_golden():
    """Load the NumPy CPU golden as a fallback for FP4 layout conversion."""
    from pathlib import Path
    import importlib.util

    impl = Path(__file__).with_name("golden.py")
    spec = importlib.util.spec_from_file_location("cla_gate_cpu_golden", impl)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cla_gate_quant_dual_npu_tensors(global_out, local_out, global_gate_logits, local_gate_logits, *,
                                    scale_alg=1, dst_type=36):
    """Run the NPU small-op reference and return raw torch tensors.

    This path is only used for FP8 outputs.  FP4 output from
    torch_npu.npu_dynamic_mx_quant_with_dual_axis uses packed uint8 with
    half the last dimension, which does not match the fused-op golden layout;
    for FP4 we fall back to the CPU NumPy golden in golden.py.
    """
    if dst_type not in (35, 36):
        raise ValueError("NPU small-op golden path only supports dst_type=35/36, got %s" % dst_type)

    device = "npu:0"
    torch.npu.set_device(device)

    go = _to_npu(global_out, device)
    lo = _to_npu(local_out, device)
    gg = _to_npu(global_gate_logits, device)
    lg = _to_npu(local_gate_logits, device)
    if gg.dim() == go.dim():
        gg = gg.squeeze(-1)
        lg = lg.squeeze(-1)

    global_scale = torch.sigmoid(gg).unsqueeze(-1)
    local_scale = torch.sigmoid(lg).unsqueeze(-1)
    merged = go * global_scale + lo * local_scale
    if merged.dim() == 3:
        # TND layout: [T, N, D] -> [T, N*D]
        merged = merged.reshape(merged.shape[0], merged.shape[1] * merged.shape[2])
    elif merged.dim() == 4:
        # Legacy SBND layout: [S, B, N, D] -> [S*B, N*D]
        merged = merged.reshape(merged.shape[0] * merged.shape[1], merged.shape[2] * merged.shape[3])
    else:
        raise ValueError("cla_gate_mx_quant_with_dual_axis expects 3D [T,N,D] or 4D [S,B,N,D]")

    row_data, row_scale, col_data, col_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        merged,
        dst_type=_dst_torch_dtype(dst_type),
        scale_alg=scale_alg,
    )
    torch.npu.synchronize()
    return row_data, row_scale, col_data, col_scale


def cla_gate_quant_dual_npu(global_out, local_out, global_gate_logits, local_gate_logits, *,
                            scale_alg=1, dst_type=36, round_mode="rint"):
    if dst_type in (35, 36):
        row_data, row_scale, col_data, col_scale = cla_gate_quant_dual_npu_tensors(
            global_out,
            local_out,
            global_gate_logits,
            local_gate_logits,
            scale_alg=scale_alg,
            dst_type=dst_type,
        )
        row_data_np = np.frombuffer(_torch_uint8_bytes(row_data), dtype=_data_numpy_dtype(dst_type)).reshape(
            row_data.shape
        )
        col_data_np = np.frombuffer(_torch_uint8_bytes(col_data), dtype=_data_numpy_dtype(dst_type)).reshape(
            col_data.shape
        )
        row_scale_np = np.frombuffer(_torch_uint8_bytes(row_scale), dtype=_e8m0_numpy_dtype()).reshape(
            row_scale.shape
        )
        col_scale_np = np.frombuffer(_torch_uint8_bytes(col_scale), dtype=_e8m0_numpy_dtype()).reshape(
            col_scale.shape
        )
        return row_data_np, row_scale_np, col_data_np, col_scale_np

    # FP4: use the CPU golden because its output layout matches the fused op.
    cpu = _load_cpu_golden()
    return cpu.cla_gate_mx_quant_with_dual_axis_golden(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        dst_type=dst_type,
        scale_alg=scale_alg,
        round_mode=round_mode,
    )


def cla_gate_mx_quant_with_dual_axis_npu_golden(
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
    """Same-device/CPU golden reference for the CLA gate fused operator."""
    if dst_type not in (35, 36, 40, 41):
        raise ValueError("unsupported dst_type: %s" % dst_type)

    # FP8 uses the NPU small-op reference; FP4 falls back to CPU NumPy golden.
    if dst_type in (35, 36):
        return cla_gate_quant_dual_npu(
            global_out,
            local_out,
            global_gate_logits,
            local_gate_logits,
            scale_alg=scale_alg,
            dst_type=dst_type,
        )

    cpu = _load_cpu_golden()
    return cpu.cla_gate_mx_quant_with_dual_axis_golden(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        dst_type=dst_type,
        scale_alg=scale_alg,
        round_mode=round_mode,
    )


__golden__ = {
    "kernel": {
        "cla_gate_mx_quant_with_dual_axis": "cla_gate_mx_quant_with_dual_axis_npu_golden"
    }
}
