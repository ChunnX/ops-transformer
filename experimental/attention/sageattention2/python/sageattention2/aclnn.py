# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ctypes bridge to the real two-phase aclnn API in the installed custom OPP."""

import ctypes as C
import functools
import os
from pathlib import Path
import torch


@functools.lru_cache(None)
def _libs():
    base = C.CDLL("libnnopbase.so")
    base.aclCreateTensor.argtypes = [
        C.POINTER(C.c_int64),
        C.c_uint64,
        C.c_int,
        C.POINTER(C.c_int64),
        C.c_int64,
        C.c_int,
        C.POINTER(C.c_int64),
        C.c_uint64,
        C.c_void_p,
    ]
    base.aclCreateTensor.restype = C.c_void_p
    base.aclDestroyTensor.argtypes = [C.c_void_p]
    base.aclDestroyTensor.restype = C.c_int
    default = Path(__file__).resolve().parents[2] / "build/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
    lib = C.CDLL(os.environ.get("SAGEATTN_ACLNN_LIBRARY", str(default)))
    lib.aclnnSageAttention2GetWorkspaceSize.argtypes = [C.c_void_p] * 3 + [
        C.c_char_p,
        C.c_bool,
        C.c_char_p,
        C.c_double,
        C.c_char_p,
        C.c_bool,
        C.c_bool,
        C.c_bool,
        C.c_void_p,
        C.c_void_p,
        C.POINTER(C.c_uint64),
        C.POINTER(C.c_void_p),
    ]
    lib.aclnnSageAttention2GetWorkspaceSize.restype = C.c_int
    lib.aclnnSageAttention2.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
    lib.aclnnSageAttention2.restype = C.c_int
    return base, lib


def run(
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
    **kwargs,
):
    from . import _validate
    import math
    import torch_npu

    qc, _, _ = _validate(q, k, v, tensor_layout, is_causal, qk_quant_gran, kwargs)
    if q.device.type != "npu":
        raise ValueError("aclnn requires NPU tensors")
    scale = q.shape[-1] ** -0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("invalid FP32 sm_scale")
    base, lib = _libs()
    with torch.npu.device(q.device):
        # The ACL bridge describes ND storage explicitly.
        q, k, v = [
            torch_npu.npu_format_cast(t, 2) if torch_npu.get_npu_format(t) not in (0, 2) else t for t in (q, k, v)
        ]
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        lse = torch.empty(qc.shape[:3], device=q.device, dtype=torch.float32) if return_lse else None
        handles = []

        def desc(t):
            if t is None:
                return None
            dims = (C.c_int64 * t.ndim)(*t.shape)
            strides = (C.c_int64 * t.ndim)(*t.stride())
            storage_shape = (
                tuple(t.shape)
                if t.is_contiguous() and t.storage_offset() == 0
                else (t.untyped_storage().nbytes() // t.element_size(),)
            )
            storage = (C.c_int64 * len(storage_shape))(*storage_shape)
            dtype = {torch.float16: 1, torch.bfloat16: 27, torch.float32: 0}[t.dtype]
            ptr = base.aclCreateTensor(
                dims,
                t.ndim,
                dtype,
                strides,
                t.storage_offset(),
                2,
                storage,
                len(storage_shape),
                t.untyped_storage().data_ptr(),
            )
            if not ptr:
                raise RuntimeError("aclCreateTensor failed")
            handles.append(ptr)
            return ptr

        try:
            size = C.c_uint64()
            executor = C.c_void_p()
            code = lib.aclnnSageAttention2GetWorkspaceSize(
                desc(q),
                desc(k),
                desc(v),
                tensor_layout.encode(),
                is_causal,
                qk_quant_gran.encode(),
                scale,
                b"fp32",
                smooth_k,
                smooth_v,
                return_lse,
                desc(out),
                desc(lse),
                C.byref(size),
                C.byref(executor),
            )
            if code:
                raise RuntimeError(f"aclnnSageAttention2GetWorkspaceSize failed: {code}")
            workspace = torch.empty(size.value, device=q.device, dtype=torch.uint8)
            stream = torch.npu.current_stream(q.device)
            code = lib.aclnnSageAttention2(workspace.data_ptr(), size.value, executor, stream.npu_stream)
            if code:
                raise RuntimeError(f"aclnnSageAttention2 failed: {code}")
            for t in (q, k, v, out, lse, workspace):
                if t is not None:
                    t.record_stream(stream)
        finally:
            for ptr in handles:
                base.aclDestroyTensor(ptr)
    return (out, lse) if return_lse else out
