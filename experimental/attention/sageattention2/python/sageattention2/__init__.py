# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SageAttention2 Ascend 950: per-token INT8 QK, FP8 PV, FP32 accumulation."""

import ctypes
import functools
import math
import os
from pathlib import Path
import torch


class _Params(ctypes.Structure):
    _fields_ = (
        [
            (x, ctypes.c_uint32)
            for x in (
                "b",
                "hq",
                "hk",
                "sq",
                "sk",
                "d",
                "dp",
                "qp",
                "kp",
                "cores",
                "bf16",
                "causal",
                "smoothK",
                "returnLse",
            )
        ]
        + [("smScale", ctypes.c_float), ("vMax", ctypes.c_float)]
        + [(x, ctypes.c_uint64 * 3) for x in ("qs", "ks", "vs")]
        + [("smoothV", ctypes.c_uint32)]
    )


@functools.lru_cache(None)
def _library():
    path = os.environ.get(
        "SAGEATTN_LIBRARY",
        str(
            Path(__file__).resolve().parents[2]
            / "build/native/libsageattention2_native.so"
        ),
    )
    lib = ctypes.CDLL(path)
    lib.sage_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Params),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    lib.sage_launch.restype = ctypes.c_int
    return lib


def _validate(q, k, v, layout, causal, gran, kwargs):
    if layout not in ("HND", "NHD"):
        raise ValueError("tensor_layout must be HND or NHD")
    if gran != "per_token":
        raise ValueError("only per_token Q/K quantization is supported")
    if kwargs.pop("attn_mask", None) is not None:
        raise ValueError("attn_mask is unsupported")
    if kwargs.pop("dropout_p", 0.0) != 0:
        raise ValueError("dropout is unsupported")
    if kwargs:
        raise TypeError(f"unsupported arguments: {sorted(kwargs)}")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("q/k/v must have rank 4")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != q.dtype for t in (k, v)
    ):
        raise ValueError("q/k/v must share FP16 or BF16 dtype")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("q/k/v must share a device")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("last dimension must be contiguous")
    if any(any(n <= 0 for n in t.shape) for t in (q, k, v)):
        raise ValueError("empty dimensions are unsupported")
    if any(any(n >= 2**32 - 64 for n in t.shape) for t in (q, k, v)):
        raise ValueError("dimensions exceed the kernel uint32 range")
    qc, kc, vc = (
        (q, k, v) if layout == "HND" else tuple(t.transpose(1, 2) for t in (q, k, v))
    )
    b, h, s, d = qc.shape
    if not 0 < d <= 128:
        raise ValueError("head_dim must be in [1,128]")
    if kc.shape != vc.shape or kc.shape[0] != b or kc.shape[-1] != d:
        raise ValueError("incompatible q/k/v shapes")
    if h % kc.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    if b * kc.shape[1] >= 2**32:
        raise ValueError("batch times KV heads exceeds the kernel uint32 range")
    if causal and s != kc.shape[2]:
        raise ValueError("causal requires Sq == Sk")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("forward-only operator does not support autograd")
    return qc, kc, vc


def _run(
    q,
    k,
    v,
    *,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_token",
    sm_scale=None,
    pv_accum_dtype="fp32",
    smooth_k=True,
    smooth_v=False,
    return_lse=False,
    quant_only=False,
    _force_wide=False,
    **kwargs,
):
    qc, kc, vc = _validate(q, k, v, tensor_layout, is_causal, qk_quant_gran, kwargs)
    if q.device.type != "npu":
        raise ValueError(
            "Ascend implementation requires NPU tensors; use reference explicitly for CPU"
        )
    import torch_npu  # noqa: F401

    scale = q.shape[-1] ** -0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("sm_scale must be representable as finite FP32")
    b, hq, sq, d = qc.shape
    hk, sk = kc.shape[1:3]
    dp = 64 if d <= 64 else 128
    qp = (sq + 127) // 128 * 128
    kp = (sk + 511) // 512 * 512
    p = _Params(
        b,
        hq,
        hk,
        sq,
        sk,
        d,
        dp,
        qp,
        kp,
        torch.npu.get_device_properties(q.device).cube_core_num,
        int(q.dtype == torch.bfloat16),
        int(is_causal),
        int(smooth_k),
        int(return_lse),
        scale,
        448.0,
        *[(ctypes.c_uint64 * 3)(*t.stride()[:3]) for t in (qc, kc, vc)],
    )
    p.smoothV = int(smooth_v)
    with torch.npu.device(q.device):
        stream = torch.npu.current_stream(q.device)

        def alloc(shape, dtype):
            return torch.empty(shape, device=q.device, dtype=dtype)

        m = alloc((2, b, hk, dp) if smooth_v else (b, hk, dp), torch.float32)
        vs = alloc((b, hk, dp), torch.float32)
        q8 = alloc((b, hq, qp, dp), torch.int8)
        k8 = alloc((b, hk, kp, dp), torch.int8)
        v8 = alloc((b, hk, kp, dp), torch.uint8)
        qs = alloc((b, hq, qp), torch.float32)
        ks = alloc((b, hk, kp), torch.float32)
        out = alloc((b, hq, sq, d), q.dtype)
        lse = alloc((b, hq, sq), torch.float32)
        qh = alloc((b, hq, qp), torch.float16)
        kh = alloc((b, hk, kp), torch.float16)
        tensors = [qc, kc, vc, m, vs, q8, k8, v8, qs, ks, out, lse, qh, kh]
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
        launch = _library().sage_launch
        if _force_wide:
            if dp != 128 or sq % 128 or sk % 512 or sk < 1024:
                raise ValueError(
                    "wide512 requires D>64, Sq divisible by 128, Sk divisible by 512 and >=1024"
                )
            launch = _library().sage_launch_wide
            launch.argtypes = _library().sage_launch.argtypes
            launch.restype = ctypes.c_int
        code = launch(stream.npu_stream, ctypes.byref(p), pointers, int(quant_only))
        if code:
            raise RuntimeError(f"sage_launch failed: {code}")
        for t in tensors:
            t.record_stream(stream)
    if quant_only:
        return (
            (q8, k8, v8, qs, ks, m[0], vs, m[1])
            if smooth_v
            else (q8, k8, v8, qs, ks, m, vs)
        )
    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return (out, lse) if return_lse else out


def sageattn_qk_int8_pv_fp8_asc(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    qk_quant_gran="per_token",
    sm_scale=None,
    pv_accum_dtype="fp32",
    smooth_k=True,
    smooth_v=False,
    return_lse=False,
    **kwargs,
):
    backend = os.environ.get("SAGEATTN_BACKEND", "aclnn")
    if backend == "aclnn":
        from .aclnn import run
    elif backend == "native":
        run = _run
    else:
        raise ValueError("SAGEATTN_BACKEND must be aclnn or native")
    return run(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        **kwargs,
    )


def sageattn(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    sm_scale=None,
    return_lse=False,
    **kwargs,
):
    return sageattn_qk_int8_pv_fp8_asc(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
        **kwargs,
    )
