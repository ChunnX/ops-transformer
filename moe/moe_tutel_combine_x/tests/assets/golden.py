#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TTK TestSpec golden for the ``moe_tutel_combine_x`` operator.

Single source of truth: ``spec/spec.yaml``.  The reference math below is a
line-by-line transcription of ``math_semantics.formula`` so it stays auditable
against the spec; nothing is taken from any kernel artifact.

Registrations (TTK TestSpec ``__spec__`` dict — loader resolves by CSV name):
  * ``moe_tutel_combine_x``   — Kernel / GEIR flow: golden gets numpy.ndarray.
  * ``aclnnMoeTutelCombineX`` — ACLNN flow: golden gets torch.Tensor (device);
    the marshaling accepts either currency, so a downstream torch-layer spec
    may inherit this golden function verbatim.

Float64 accumulation rule (workflow-mandated, overrides any lower accumulator
dtype declared in the spec): every floating-point operand is cast to float64
BEFORE the combine, the accumulation is evaluated entirely in float64, and
only the final result is cast back to the target output dtype
(``x_grad.dtype = y_grad.dtype`` per the spec dtype_rule).  Under this op's
shape contract each output row is fed by exactly one routing entry (spec
shape_constraints notes), so the zero-init + per-entry accumulate degenerates
to one guarded multiply per row — still computed in float64, never in
float16/float32.  Overflow on the final cast becomes +/-inf (the spec's fp16
overflow boundary semantics), NaN/inf propagate on valid rows only.

Tolerance derivation (never inferred from the operator name):
  * op.paradigms = [IndexGather, MaskPredicate] — compute paradigms with float
    output, so the default classification would be ``stat_rel_err``; the spec's
    numerical_tolerance (per-dtype rtol/atol, metric max_relative) is the
    declared refinement of those thresholds.
  * The spec's top-level ``cross_check`` block overrides the classification for
    every float16/bfloat16/float32 output dtype:
    ``{standard: cross_check, level: <spec cross_check.level>}`` (here ``L1``).
    This operator has no other output dtypes, so the tolerance table below is
    exactly those three entries.

third_party (competitor reference — cross_check requires it): the spec
reference_oracle is ``torch.index_select`` with kwargs ``{dim: 0}``.  A single
API cannot receive all four operator inputs, so the competitor is declared as
a class composing the spec's equivalent reference decomposition —
index_select row gathering + gates broadcast multiply + invalid-entry zeroing.
    Competitor APIs are reached through the ``third_party`` declaration (executed
    in an isolated XPU subprocess), never via an in-process import.  The xpu
    server binds ``__call__`` parameters by name against the caller's param
    pool, and that pool differs per flow (aclnn: C-header camelCase
    ``yGrad``/``gates``/``indices``/``locations``/``out``; proto/framework:
    spec snake_case ``y_grad``/...), so ``__call__`` accepts both spellings of
    the first operand (all binding params defaulted — names absent from a pool
    are skipped by the dispatcher, never mis-bound); ``dim`` carries the
    reference_oracle kwarg default.
"""

__spec__ = {
    "moe_tutel_combine_x": "MoeTutelCombineXTestSpec",
    "aclnnMoeTutelCombineX": "AclnnMoeTutelCombineXTestSpec",
}

import numpy as np

# spec.yaml cross_check: {standard: cross_check, level: L1} — applied to each
# float16/bfloat16/float32 output dtype (the only output dtypes of this op).
_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _moe_tutel_combine_x_fp64(y_grad, gates, indices, locations):
    """Reference math — verbatim transcription of spec.yaml math_semantics.formula.

    ``y_grad`` / ``gates`` must already be float64 (cast by the callers); the
    float64 accumulation rule is applied at these boundaries.  ``indices`` /
    ``locations`` may be any integer dtype (cast to int64 below, mirroring the
    spec formula).  Returns a float64 ndarray shaped ``gates.shape + (H,)``.
    """
    H = y_grad.shape[-1]
    S = gates.shape[-1]
    B = (gates.size // S) if S > 0 else 1
    EC = (y_grad.size // (B * H)) if (B * H) > 0 else 0
    y3 = y_grad.reshape(B, EC, H)
    g2 = gates.reshape(B, S)
    i2 = indices.reshape(B, S).astype(np.int64)
    l2 = locations.reshape(B, S).astype(np.int64)
    E = (int(np.max(i2)) + 1) if i2.size else 1
    C = (EC // E) if E >= 1 else 0
    src = i2 * C + l2
    v1 = i2 >= 0
    v2 = l2 >= 0
    v3 = l2 < C
    v4 = src >= 0
    v5 = src < EC
    valid = np.logical_and(
        np.logical_and(np.logical_and(v1, v2), np.logical_and(v3, v4)), v5
    )
    safe = np.clip(src, 0, EC - 1)
    y2 = y3.reshape(B * EC, H)
    ypad = np.concatenate([y2, np.zeros((1, H), dtype=y2.dtype)], axis=0)
    flat = np.arange(B)[:, None] * EC + safe
    gathered = ypad[np.minimum(flat, B * EC), :]
    # Accumulation (spec notes: x_grad is zero-initialized then accumulated per
    # routing entry; each row has exactly one entry here): the guarded multiply
    # runs in float64; 0 * inf = NaN on a valid row is IEEE-correct, and the
    # np.where else-branch keeps invalid rows at an exact +0.0.
    with np.errstate(invalid="ignore"):
        x_grad = np.where(
            valid[:, :, None], g2[:, :, None] * gathered, np.zeros((), dtype=y2.dtype)
        )
    return x_grad.reshape(gates.shape + (H,))


def _to_numpy(t, np_dtype):
    """Marshal one operand to a numpy array of ``np_dtype`` (torch or numpy input)."""
    if hasattr(t, "detach"):  # torch.Tensor (any device) -> CPU numpy
        import torch

        torch_dtype = torch.float64 if np_dtype == np.float64 else torch.int64
        return t.detach().cpu().to(torch_dtype).numpy()
    return np.asarray(t).astype(np_dtype)


def _cast_back(x64, like):
    """Cast the float64 result back to the target output dtype.

    ``x_grad.dtype = y_grad.dtype`` (spec dtype_rule), so the target dtype and
    the return currency follow ``like`` (the y_grad operand as received).
    """
    if hasattr(like, "detach"):  # torch in -> CPU torch tensor out
        import torch

        return torch.from_numpy(np.ascontiguousarray(x64)).to(like.dtype)
    # numpy in -> numpy out; overflow -> +/-inf is the spec's fp16/bf16 boundary
    # semantics, keep the cast warning-free.
    with np.errstate(over="ignore", invalid="ignore"):
        return x64.astype(like.dtype)


def _combine_x_golden_impl(y_grad, gates, indices, locations):
    """Shared golden entry: fp64 accumulation, cast back to the output dtype."""
    x64 = _moe_tutel_combine_x_fp64(
        _to_numpy(y_grad, np.float64),
        _to_numpy(gates, np.float64),
        _to_numpy(indices, np.int64),
        _to_numpy(locations, np.int64),
    )
    return _cast_back(x64, y_grad)


class MoeTutelCombineXTorchRef:
    """third_party torch reference — executed isolated on the xpu-server.

    Anchored on the spec reference_oracle (``torch.index_select``, kwargs
    ``dim: 0``) composed with the spec's equivalent reference decomposition:
    index_select row gather + gates broadcast multiply + invalid-entry zeroing.
    ``__call__`` binds by name against the caller's param pool: the aclnn flow
    derives names from the C header (``yGrad``/``gates``/``indices``/
    ``locations``/``out``), proto/framework flows from the spec inputs
    (``y_grad``/...).  All binding params are keyword-only and defaulted so a
    name absent from one pool is skipped (never mis-bound to a shifted
    position; leftovers land in ``**kwargs``), and the first operand accepts
    both spellings (``yGrad`` wins when bound).  ``dim`` keeps the
    reference_oracle kwarg default.  Computed in the tensors' native dtype —
    the competitor's own semantics.
    """

    def __init__(self, dim=0, **kwargs):
        self.dim = dim

    def __call__(
        self,
        *,
        y_grad=None,
        gates=None,
        indices=None,
        locations=None,
        yGrad=None,
        **kwargs,
    ):
        import torch

        y = yGrad if yGrad is not None else y_grad
        H = y.shape[-1]
        S = gates.shape[-1]
        B = (gates.numel() // S) if S > 0 else 1
        EC = (y.numel() // (B * H)) if (B * H) > 0 else 0
        y3 = y.reshape(B, EC, H)
        g2 = gates.reshape(B, S)
        i2 = indices.reshape(B, S).to(torch.long)
        l2 = locations.reshape(B, S).to(torch.long)
        E = (int(i2.max().item()) + 1) if i2.numel() else 1
        C = (EC // E) if E >= 1 else 0
        src = i2 * C + l2
        valid = (i2 >= 0) & (l2 >= 0) & (l2 < C) & (src >= 0) & (src < EC)
        # np.clip(src, 0, EC - 1) semantics: minimum(maximum(x, lo), hi).
        safe = torch.minimum(
            torch.maximum(src, torch.zeros_like(src)), torch.full_like(src, EC - 1)
        )
        y2 = y3.reshape(B * EC, H)
        ypad = torch.cat(
            (y2, torch.zeros((1, H), dtype=y2.dtype, device=y2.device)), dim=0
        )
        flat = torch.arange(B, device=src.device)[:, None] * EC + safe
        # EC == 0 degenerates safe to -1 (lo > hi); route it to the zero pad row
        # so index_select (which rejects negative indices) stays in bounds.
        flat = torch.clamp(flat, min=0)
        gathered = torch.index_select(ypad, self.dim, flat.reshape(-1)).reshape(B, S, H)
        x = g2.unsqueeze(-1) * gathered
        x = torch.where(
            valid.unsqueeze(-1), x, torch.zeros((), dtype=y.dtype, device=y.device)
        )
        return [x.reshape(tuple(gates.shape) + (H,))]


class MoeTutelCombineXTestSpec:
    """Kernel / GEIR flow — golden receives numpy.ndarray, returns numpy.ndarray."""

    def golden(y_grad, gates, indices, locations, **kwargs):
        """Parameters follow the operator input order (spec.yaml ``inputs``).

        **kwargs carries the TTK framework context (input_dtypes, shapes,
        soc version, testcase name, ...) — not used by this golden.
        """
        return [_combine_x_golden_impl(y_grad, gates, indices, locations)]

    third_party = {"torch": MoeTutelCombineXTorchRef}
    tolerance = _TOLERANCE


class AclnnMoeTutelCombineXTestSpec:
    """ACLNN flow — golden receives torch.Tensor (device), returns torch.Tensor (CPU).

    Parameter order follows ``aclnnMoeTutelCombineXGetWorkspaceSize`` minus
    workspaceSize/executor.  The C header declares the output tensor ``out``
    as the 5th positional param and TTK passes all 5 positionally, so the
    golden carries a trailing ``out=None`` placeholder — the kernel's
    (uninitialized) output buffer, ignored: the golden returns its own
    expected value.
    """

    def golden(y_grad, gates, indices, locations, out=None, **kwargs):
        return [_combine_x_golden_impl(y_grad, gates, indices, locations)]

    third_party = {"torch": MoeTutelCombineXTorchRef}
    tolerance = _TOLERANCE
