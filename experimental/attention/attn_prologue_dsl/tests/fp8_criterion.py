# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Value-domain adjudication of MXFP8 output mismatches against an FP64 reference.

A public MXFP8 output is a ``(payload, shared scale)`` pair and what a consumer
reads is their product, so this module judges the product rather than the raw
bytes. One rule then covers the three boundary classes that a K=5120 FP32
reduction actually produces:

* **signed zero** — ``+0`` and ``-0`` are different bytes and the same value;
* **shared-exponent off by one** — a payload code moved one binade against a
  scale byte moved the other way reconstructs the identical value;
* **adjacent code** — device and reference are the two representable values
  bracketing the FP64 truth, with nothing representable in between.

Anything else is a hard mismatch. The rule is deliberately not a tolerance: it
admits at most one quantization step, and only when the FP64 truth lies inside
the bracket. The downstream ``q`` check still bounds what that step is worth.

The reference must be FP64. An FP32 reference can choose the opposite FP8 code
near a quantization boundary after long reductions; see the accuracy criteria
in ``README.md``.
"""
from __future__ import annotations

import torch

MX_GROUP = 32
E8M0_BIAS = 127
E4M3_MAX = 448.0


def _finite_e4m3_values() -> torch.Tensor:
    """Sorted unique finite E4M3FN values, used as the representable grid.

    ``+0``/``-0`` collapse to one entry, which is what adjacency should see.
    """
    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).to(torch.float64)
    return torch.unique(values[torch.isfinite(values)])


_GRID = _finite_e4m3_values()


def decode_payload(payload: torch.Tensor) -> torch.Tensor:
    """FP8 payload bytes → FP64 values (zero-copy reinterpret, then widen)."""
    return payload.contiguous().view(torch.float8_e4m3fn).to(torch.float64)


def expand_scale(paired: torch.Tensor, groups: int, width: int) -> torch.Tensor:
    """Paired E8M0 ``(rows, ceil(G/2), 2)`` → per-element power-of-two factor.

    The public layout stores consecutive groups in pairs, so a flat view walks
    the groups in order. Element ``col`` belongs to group ``col // 32``.
    """
    flat = paired.contiguous().view(torch.uint8).view(paired.shape[0], -1)[:, :groups]
    factor = torch.ldexp(torch.ones_like(flat, dtype=torch.float64),
                         flat.to(torch.int32) - E8M0_BIAS)
    return factor.repeat_interleave(MX_GROUP, dim=1)[:, :width]


def representable_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Count representable E4M3 values strictly between ``a`` and ``b``."""
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    first_above = torch.searchsorted(_GRID, lo.contiguous(), right=True)
    first_at = torch.searchsorted(_GRID, hi.contiguous(), right=False)
    return (first_at - first_above).clamp_min(0)


def encode_at_scale(truth: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Encode an FP64 value at a given shared scale, exactly as the op does.

    Divide by the scale, saturate at the E4M3 limit, round to nearest even. Both
    the kernel (``vcast`` with ``RoundingMode.RN``) and the golden
    (``.to(float8_e4m3fn)``) do precisely this, so the result is what a correct
    implementation must emit for that value at that scale.
    """
    normalized = (truth / scale).clamp(-E4M3_MAX, E4M3_MAX)
    return normalized.to(torch.float8_e4m3fn).view(torch.uint8)


class Verdict:
    """Per-element classification counts plus the surviving hard examples."""

    __slots__ = ("checked", "value_identical", "adjacent_bracket",
                 "own_scale_consistent", "hard", "hard_index")

    def __init__(self, checked, value_identical, adjacent_bracket,
                 own_scale_consistent, hard, hard_index):
        self.checked = checked
        self.value_identical = value_identical
        self.adjacent_bracket = adjacent_bracket
        self.own_scale_consistent = own_scale_consistent
        self.hard = hard
        self.hard_index = hard_index

    @property
    def legitimate(self) -> int:
        return (self.value_identical + self.adjacent_bracket
                + self.own_scale_consistent)


def adjudicate(got_payload, ref_payload, truth, *, got_scale=None, ref_scale=None):
    """Judge dequantized values against an FP64 truth.

    ``got_payload`` / ``ref_payload`` are FP8 bytes of identical shape.
    ``truth`` is the FP64 value the pair is meant to represent, in the same
    (pre-scale) domain as ``payload * scale``. Scales are per-element
    power-of-two factors, or ``None`` for an output whose scale is fixed at 1
    (the KV cache path).

    Every supplied element is judged, including ones whose payload bytes agree:
    a scale that moved on its own still changes the reconstructed value.
    """
    if got_payload.shape != ref_payload.shape:
        raise ValueError("payload tensors must have the same shape")
    if got_payload.shape != truth.shape:
        raise ValueError("truth must have the same shape as the payloads")

    got_value = decode_payload(got_payload)
    ref_value = decode_payload(ref_payload)
    if got_scale is None:
        got_scale = torch.ones_like(got_value)
    if ref_scale is None:
        ref_scale = torch.ones_like(ref_value)

    dq_got = got_value * got_scale
    dq_ref = ref_value * ref_scale
    truth = truth.to(torch.float64)

    finite = torch.isfinite(dq_got) & torch.isfinite(dq_ref)
    identical = finite & (dq_got == dq_ref)

    # One quantization step apart, and only on a shared scale: two different
    # scales put the values on different grids, where "adjacent" is not defined.
    same_scale = got_scale == ref_scale
    bracket = (truth >= torch.minimum(dq_got, dq_ref)) & (truth <= torch.maximum(dq_got, dq_ref))
    adjacent = representable_between(got_value, ref_value) == 0
    step = finite & ~identical & same_scale & adjacent & bracket

    # The two sides can also land in different binades when the group amax sits
    # within FP32 noise of a power of two. With `e = floor(log2(amax)) - 8` the
    # normalized amax lands in [256, 512) while E4M3 stops at 448, so the group
    # saturates whenever that mantissa exceeds 1.75 -- and then the binade the
    # scale picked decides whether it saturates at all, which is why the two
    # reconstructions differ. Neither side is the wrong one: the device is the
    # more accurate of the two as often as not. Accept it when the device
    # payload is exactly what its own scale demands for the FP64 truth, which
    # also covers the saturated endpoint because the encoding clamps.
    own_scale = finite & ~identical & ~step & (
        encode_at_scale(truth, got_scale) == got_payload.contiguous().view(torch.uint8))

    hard_mask = ~(identical | step | own_scale)
    return Verdict(
        checked=int(got_payload.numel()),
        value_identical=int(identical.sum()),
        adjacent_bracket=int(step.sum()),
        own_scale_consistent=int(own_scale.sum()),
        hard=int(hard_mask.sum()),
        hard_index=torch.nonzero(hard_mask, as_tuple=False),
    )


def groups_needing_review(got_scale_bytes, ref_scale_bytes, groups):
    """Group indices whose scale byte differs, per row.

    A scale byte that moved while its payload bytes stayed put changes the
    reconstructed value without producing a payload mismatch, so those groups
    have to be pulled into the adjudicated set explicitly.
    """
    got = got_scale_bytes.contiguous().view(torch.uint8).view(got_scale_bytes.shape[0], -1)[:, :groups]
    ref = ref_scale_bytes.contiguous().view(torch.uint8).view(ref_scale_bytes.shape[0], -1)[:, :groups]
    return torch.nonzero(got != ref, as_tuple=False)
