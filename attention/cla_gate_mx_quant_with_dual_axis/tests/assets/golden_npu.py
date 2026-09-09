#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""NPU small-op golden for cla_gate_mx_quant_with_dual_axis.

This plugin runs the small-op reference path on the NPU itself:
    torch.sigmoid(gate).unsqueeze(-1) -> broadcast mul -> add
    -> reshape [T, N*D]
    -> torch_npu.npu_dynamic_mx_quant_with_dual_axis(...)

FP8/FP4 both use the NPU small-op reference.  For FP4, torch_npu returns the
data tensors already in packed uint8 layout (two 4-bit values per byte), so the
golden keeps that packed layout for TTK's uint8 FP4 cases.

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
    if dst_type == 40:
        # torch_npu.npu_dynamic_mx_quant_with_dual_axis expects the torch_npu
        # FP4 enum (296 = float4_e2m1fn_x2), not GE dst_type 40. Passing the
        # dtype object makes op-plugin expand packed output to 2x width, and
        # passing 40 is interpreted as int4; use the torch_npu enum integer.
        return 296
    if dst_type == 41:
        return 297
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


def _get_device_id():
    """Select the NPU device used by the golden.

    TTK_GOLDEN_DEVICE_ID is set by run_ttk_kernel_single_axis.sh from
    --device-whitelist. If it is present, use only that device.
    """
    explicit = os.environ.get("TTK_GOLDEN_DEVICE_ID", "").strip()
    if explicit.isdigit():
        dev_id = int(explicit)
        try:
            torch.npu.set_device(dev_id)
            _ = torch.zeros(1, device=f"npu:{dev_id}")
            return dev_id
        except Exception:
            pass

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if visible:
        for item in visible.split(","):
            item = item.strip()
            if not item.isdigit():
                continue
            dev_id = int(item)
            try:
                torch.npu.set_device(dev_id)
                _ = torch.zeros(1, device=f"npu:{dev_id}")
                return dev_id
            except Exception:
                continue

    candidates = []
    try:
        current = torch.npu.current_device()
        if current is not None and int(current) >= 0:
            candidates.append(int(current))
    except Exception:
        pass

    try:
        device_count = torch.npu.device_count()
    except Exception:
        device_count = 0
    for dev_id in range(max(device_count, 1)):
        if dev_id not in candidates:
            candidates.append(dev_id)

    for dev_id in candidates:
        try:
            torch.npu.set_device(dev_id)
            _ = torch.zeros(1, device=f"npu:{dev_id}")
            return dev_id
        except Exception:
            continue

    return 0


def _unpack_packed_fp4(tensor, dst_type):
    """Unpack a torch_npu packed uint8 FP4 tensor to en_dtypes logical shape."""
    from ttk.utilities import unpack_4bits

    np_dtype = _data_numpy_dtype(dst_type)
    raw = np.frombuffer(_torch_uint8_bytes(tensor), dtype=np.uint8)
    unpacked = unpack_4bits(raw, np_dtype)
    logical_shape = list(tensor.shape)
    logical_shape[-1] *= 2
    return unpacked.reshape(logical_shape)


def cla_gate_quant_dual_npu_tensors(global_out, local_out, global_gate_logits, local_gate_logits, *,
                                    scale_alg=1, dst_type=36, round_mode="rint"):
    """Run the NPU small-op reference and return raw torch tensors.

    Supports FP8 (dst_type=35/36) and FP4 (dst_type=40/41).  FP4 data tensors
    returned by torch_npu are packed uint8 with half the last dimension.
    """
    if dst_type not in (35, 36, 40, 41):
        raise ValueError("NPU small-op golden path only supports dst_type=35/36/40/41, got %s" % dst_type)

    device_id = _get_device_id()
    device = f"npu:{device_id}"

    go = _to_npu(global_out, device)
    lo = _to_npu(local_out, device)
    gg = _to_npu(global_gate_logits, device)
    lg = _to_npu(local_gate_logits, device)
    if gg.dim() == go.dim():
        gg = gg.squeeze(-1)
        lg = lg.squeeze(-1)

    orig_dtype = go.dtype
    go = go.to(torch.float32)
    lo = lo.to(torch.float32)
    gg = gg.to(torch.float32)
    lg = lg.to(torch.float32)
    global_scale = torch.sigmoid(gg).unsqueeze(-1)
    local_scale = torch.sigmoid(lg).unsqueeze(-1)
    merged = (go * global_scale + lo * local_scale).to(orig_dtype)
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
        round_mode=round_mode,
        dst_type=_dst_torch_dtype(dst_type),
        scale_alg=scale_alg,
    )
    torch.npu.synchronize()
    return row_data, row_scale, col_data, col_scale


def cla_gate_quant_dual_npu(global_out, local_out, global_gate_logits, local_gate_logits, *,
                            scale_alg=1, dst_type=36, round_mode="rint", quant_mode=1):
    row_data, row_scale, col_data, col_scale = cla_gate_quant_dual_npu_tensors(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        scale_alg=scale_alg,
        dst_type=dst_type,
        round_mode=round_mode,
    )

    if dst_type in (35, 36):
        row_data_np = np.frombuffer(_torch_uint8_bytes(row_data), dtype=_data_numpy_dtype(dst_type)).reshape(
            row_data.shape
        )
        col_data_np = np.frombuffer(_torch_uint8_bytes(col_data), dtype=_data_numpy_dtype(dst_type)).reshape(
            col_data.shape
        )
    else:
        # TTK CSV keeps the logical FP4 dtype/shape, so unpack the NPU small-op
        # packed uint8 outputs back to full float4 numpy arrays.
        row_data_np = _unpack_packed_fp4(row_data, dst_type)
        col_data_np = _unpack_packed_fp4(col_data, dst_type)

    row_scale_np = np.frombuffer(_torch_uint8_bytes(row_scale), dtype=_e8m0_numpy_dtype()).reshape(
        row_scale.shape
    )
    col_scale_np = np.frombuffer(_torch_uint8_bytes(col_scale), dtype=_e8m0_numpy_dtype()).reshape(
        col_scale.shape
    )
    if quant_mode == 0:
        # Single-axis mode outputs empty col_data/col_scale.
        col_data_np = np.empty((0,), dtype=col_data_np.dtype)
        col_scale_np = np.empty((0,), dtype=col_scale_np.dtype)

    return row_data_np, row_scale_np, col_data_np, col_scale_np


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
    """Same-device NPU small-op golden reference for the CLA gate fused operator."""
    if dst_type not in (35, 36, 40, 41):
        raise ValueError("unsupported dst_type: %s" % dst_type)
    if dst_type in (35, 36) and round_mode != "rint":
        raise ValueError("FP8 only supports round_mode=rint")
    if dst_type in (40, 41) and scale_alg != 0:
        raise ValueError("FP4 only supports scale_alg=0")

    quant_mode = int(kwargs.get("quant_mode", 1))
    return cla_gate_quant_dual_npu(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        scale_alg=scale_alg,
        dst_type=dst_type,
        round_mode=round_mode,
        quant_mode=quant_mode,
    )


__golden__ = {
    "kernel": {
        "cla_gate_mx_quant_with_dual_axis": "cla_gate_mx_quant_with_dual_axis_npu_golden"
    }
}
