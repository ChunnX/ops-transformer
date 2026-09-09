#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------

import math

import numpy as np



# TTK TestSpec registration (ops-test-kit): kernel/GEIR use op_name, ACLNN uses api_name.
__spec__ = {
    "pdist": "PdistTestSpec",
    "aclnnPdist": "AclnnPdistTestSpec",
    "aclnnPdistForward": "AclnnPdistForwardTestSpec",
}

# Legacy atk registration (ops_st_test.sh framework); consumers migrate to TestSpec.
__golden__ = {
    "aclnn": {
        "aclnnPdist": "aclnn_pdist_golden",
        "aclnnPdistForward": "aclnn_pdist_forward_golden",
    },
    "kernel": {"pdist": "pdist_golden"},
}

# SPEC machine contract tolerance: atol = rtol = 1e-3 for both IO dtypes
# (fp16 error is already contained by the on-chip fp32 accumulation chain).
# max_abs_error_limit is the mix_tolerance secondary absolute cap: pdist
# outputs are unbounded in magnitude (p<1 raises sum^(1/p); wide ranges raise
# |y| to ~1e3-1e4), so the cap must scale with the output ulp, not stay at the
# small-value default (0.01/0.1). Observed max_abs_error is always output-
# magnitude * ulp (~6e-6 rel fp32, ~2 ulp fp16 at the exit CAST_RINT).
_TOLERANCE = {
    "float16": {"standard": "mix_tolerance", "rtol": 0.001, "atol": 0.001, "max_abs_error_limit": 2.0},
    "float32": {"standard": "mix_tolerance", "rtol": 0.001, "atol": 0.001, "max_abs_error_limit": 1.0},
}

_INF_STRINGS = ("inf", "+inf", "infinity", "+infinity")

# Work budgets for the chunked pair enumeration (performance knobs only).
_WORK_ELEMS = 128 * 1024 * 1024   # max elements of the (c, T, M) fp32 diff temporary
_DM_ELEMS = 16 * 1024 * 1024      # max elements of the (c, T) reduced distance matrix


def _normalize_p(p):
    """Normalize the p attribute: accepts int/float and the string "inf" forms."""
    if isinstance(p, str):
        s = p.strip().lower()
        if s in _INF_STRINGS:
            return float("inf")
        return float(s)
    return float(p)


def _pdist_fp32_np(x, p):
    """Chunked FP32 pairwise p-norm distance on an [N, M] numpy array.

    Pure numpy on purpose: the kernel-path golden runs in the same process as
    TTK's online compile, and the pypto opc wrapper stubs sys.modules["torch"]
    with a MagicMock at import time (build-only stub). Any torch call after
    that would silently compute on mocks.

    Semantics follow op_kernel/pdist.py (frozen SPEC contract):
      - 0 < p < +inf : y[idx] = ( sum_k |x[i,k]-x[j,k]|**p )**(1/p)   Minkowski
      - p == 0       : y[idx] = sum_k (x[i,k] != x[j,k])               Hamming
                       (NaN dims count as mismatched, no NaN in output)
      - p == +inf    : y[idx] = max_k |x[i,k]-x[j,k]|                  Chebyshev
                       (NaN propagates per pair via amax)
    Output is the upper-triangular row-major vector of length N*(N-1)/2 with
    idx = i*(2*N-i-1)//2 + (j-i-1).
    """
    N, M = x.shape
    L = N * (N - 1) // 2
    y = np.empty(L, dtype=np.float32)
    pos = 0
    i0 = 0
    while i0 < N - 1:
        T = N - i0  # targets x[i0:] cover every j > i for rows in this chunk
        c = _WORK_ELEMS // (T * M)
        c = min(c, max(1, _DM_ELEMS // T))
        c = max(1, min(c, N - 1 - i0))
        rows = x[i0:i0 + c]                                  # (c, M)
        tgt = x[i0:]                                         # (T, M)
        diff = np.abs(rows[:, None, :] - tgt[None, :, :])    # (c, T, M) fp32
        if p == 0.0:
            dm = np.count_nonzero(diff, axis=-1).astype(np.float32)
        elif math.isinf(p):
            dm = np.amax(diff, axis=-1)
        else:
            dm = np.power(np.power(diff, p).sum(axis=-1), 1.0 / p)
        # Pairs (r, t) with t > r in row-major order: exactly the global
        # upper-triangular order for rows [i0, i0+c).
        r = np.arange(c)[:, None]
        t = np.arange(T)[None, :]
        vals = dm[r < t]
        y[pos:pos + vals.size] = vals
        pos += vals.size
        i0 += c
    return y


def _golden_common(input_x, p):
    """Shared computation: widen at entry (fp16->fp32), narrow at the exit."""
    p = _normalize_p(p)
    x_dtype = input_x.dtype
    xf = input_x.astype(np.float32, copy=False)
    if _is_tile_aligned(xf.shape):
        # Tile-aligned shapes: the chunked numpy reference (fast, validated).
        output = _pdist_fp32_np(xf, p)
    else:
        # Non-aligned shapes (M tail / pair tail): torch small-op composition
        # (abs/pow/sum/amax/count_nonzero — NOT F.pdist, whose p=0 NaN
        # semantics diverge from the kernel contract). Falls back to the numpy
        # reference only when the real torch module is not recoverable in this
        # process (pypto stub with no pre-imported torch.* submodule).
        try:
            output = _pdist_fp32_torch(xf, p)
        except RuntimeError:
            output = _pdist_fp32_np(xf, p)
    return output.astype(x_dtype, copy=False)


# Kernel tile alignment (op_kernel/pdist.py frozen constants): the M axis is
# processed in M_BLK=128 blocks and the pair axis in TP=64 lanes (row 0 owns
# the longest run of N-1 pairs). A shape is tile-aligned iff neither axis
# produces a tail block; every other case routes to the torch-composed golden.
_M_BLK = 128
_TP = 64


def _is_tile_aligned(shape):
    N, M = shape
    return M % _M_BLK == 0 and (N - 1) % _TP == 0


def _real_torch():
    """Return the real torch module, recovering from the pypto build stub.

    The pypto opc dynamic-compile wrapper (custom_transformer_impl/dynamic/
    <op>.py) replaces sys.modules["torch"] with a MagicMock at import time so
    the DSL kernel can be loaded without a real torch. TTK's kernel mode
    compiles and computes the golden in the SAME process, so a plain
    `import torch` in the golden would compute on mocks. A fresh re-import is
    NOT possible either: torch._C is still loaded and re-executing torch's
    __init__ aborts on double TORCH_LIBRARY registration. Instead, the real
    module object is recovered from any torch.* submodule imported before the
    stub (their module globals keep the parent reference, e.g. torch._utils),
    and sys.modules["torch"] is restored so later imports get the real module.
    """
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    import sys

    mod = sys.modules.get("torch")
    if mod is not None and not type(mod).__module__.startswith("unittest.mock"):
        _TORCH = mod
        return mod
    for name in list(sys.modules):
        if not name.startswith("torch."):
            continue
        cand = getattr(sys.modules[name], "__dict__", {}).get("torch", None)
        if cand is not None and hasattr(cand, "from_numpy") and hasattr(cand, "__version__"):
            _TORCH = cand
            sys.modules["torch"] = cand  # un-stub for the rest of the process
            return cand
    raise RuntimeError(
        "pdist golden: real torch module not recoverable "
        "(pypto stub active and no pre-imported torch.* submodule)"
    )


_TORCH = None


def _pdist_fp32_torch(x, p):
    """Chunked FP32 pairwise p-norm distance on an [N, M] numpy array.

    Torch small-operator composition (per contract branch):
      - 0 < p < +inf : abs -> pow(p) -> sum(M) -> pow(1/p)
      - p == 0       : abs -> count_nonzero(M)   (NaN != 0, counted, no NaN out)
      - p == +inf    : abs -> amax(M)            (NaN propagates per pair)
    Same chunking/order as _pdist_fp32_np; same upper-triangular layout.
    """
    torch = _real_torch()
    xt = torch.from_numpy(np.ascontiguousarray(x))
    N, M = xt.shape
    L = N * (N - 1) // 2
    y = torch.empty(L, dtype=torch.float32)
    pos = 0
    i0 = 0
    with torch.no_grad():
        while i0 < N - 1:
            T = N - i0  # targets x[i0:] cover every j > i for rows in this chunk
            c = _WORK_ELEMS // (T * M)
            c = min(c, max(1, _DM_ELEMS // T))
            c = max(1, min(c, N - 1 - i0))
            rows = xt[i0:i0 + c]                                # (c, M)
            tgt = xt[i0:]                                       # (T, M)
            diff = (rows.unsqueeze(1) - tgt.unsqueeze(0)).abs()  # (c, T, M) fp32
            if p == 0.0:
                dm = torch.count_nonzero(diff, dim=-1).to(torch.float32)
            elif math.isinf(p):
                dm = torch.amax(diff, dim=-1)
            else:
                dm = diff.pow(p).sum(dim=-1).pow(1.0 / p)
            # Pairs (r, t) with t > r in row-major order: exactly the global
            # upper-triangular order for rows [i0, i0+c).
            r = torch.arange(c).unsqueeze(1)
            t = torch.arange(T).unsqueeze(0)
            vals = torch.masked_select(dm, r < t)
            y[pos:pos + vals.numel()] = vals
            pos += vals.numel()
            i0 += c
    return y.numpy()


def pdist_golden(input_x, p=2.0, **kwargs):
    """
    Kernel golden for pdist.
    All the parameters follow @pdist_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    return _golden_common(input_x, p)


def aclnn_pdist_golden(self, p=2.0, out=None, **kwargs):
    """
    Aclnn golden for aclnnPdist.
    Parameters follow @aclnnPdistGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    p = _normalize_p(p)
    arr = self.detach().cpu().numpy().astype(np.float32, copy=False)
    # Same aligned/non-aligned dispatch as the kernel golden (shared core):
    # non-aligned shapes route to the torch small-op composition.
    if _is_tile_aligned(arr.shape):
        output = _pdist_fp32_np(arr, p)
    else:
        try:
            output = _pdist_fp32_torch(arr, p)
        except RuntimeError:
            output = _pdist_fp32_np(arr, p)
    if self.dtype == torch.float16:
        output = output.astype(np.float16, copy=False)
    return [torch.from_numpy(np.ascontiguousarray(output))]


def aclnn_pdist_forward_golden(self, pScalar=2.0, out=None, **kwargs):
    """
    Aclnn golden for aclnnPdistForward (scalar-p composite entry; same math).
    Parameters follow @aclnnPdistForwardGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    return aclnn_pdist_golden(self, p=pScalar, out=out, **kwargs)


class PdistTestSpec:
    """TTK TestSpec for the pdist kernel / GEIR pathways (numpy in, list out)."""

    @staticmethod
    def golden(input_x, p=2.0, **kwargs):
        return [pdist_golden(input_x, p=p, **kwargs)]

    tolerance = _TOLERANCE


class AclnnPdistTestSpec:
    """TTK TestSpec for the aclnnPdist pathway (torch in, list out)."""

    @staticmethod
    def golden(self, p=2.0, out=None, **kwargs):
        return aclnn_pdist_golden(self, p=p, out=out, **kwargs)

    tolerance = _TOLERANCE


class AclnnPdistForwardTestSpec:
    """TTK TestSpec for the aclnnPdistForward pathway (scalar-p composite)."""

    @staticmethod
    def golden(self, pScalar=2.0, out=None, **kwargs):
        return aclnn_pdist_forward_golden(self, pScalar=pScalar, out=out, **kwargs)

    tolerance = _TOLERANCE
