# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bit-exactness tests for the raw-VF integer emulation of MXFP4 quantization.

The vector unit has no fp4 cast (verified against this wheel: ``vcast`` rejects
every f32/f16 <-> fp4 pair, and ``Buffer`` rejects fp4 dtypes), so
``VectorQ.epilogue`` emulates E2M1 rounding with integer bit manipulation plus a
gather table.  This module reimplements *exactly* that integer sequence in torch
and asserts it reproduces the CPU golden bit for bit.

Keep :func:`emulate_mx_quant` in lockstep with ``VectorQ.epilogue``: it is the
only place the register algorithm is checked, since the target NPU is not
available for runtime tests here.
"""

from __future__ import annotations

import pytest
import torch

from golden import (  # noqa: E402
    MX_GROUP_SIZE,
    ceil_div,
    mx_quant_along_last,
)

pytest.importorskip("cannbotdsl")

from kernel import (  # noqa: E402
    E8M0_BIAS,
    FP32_HALF_MANTISSA,
    FP32_MAG_MASK,
    LUT_LEN,
    LUT_SHIFT,
    build_e2m1_lut,
)

LUT = build_e2m1_lut()


def _group_amax(x: torch.Tensor) -> torch.Tensor:
    """Pass 1: |max| per 32-element MX group along the last axis."""
    *lead, d = x.shape
    groups = ceil_div(d, MX_GROUP_SIZE)
    pad = groups * MX_GROUP_SIZE - d
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(*lead, groups, MX_GROUP_SIZE).abs().amax(dim=-1)


def _e8m0_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Pass 2: division-free ``ceil(log2(amax/6))``, exactly as the kernel does."""
    ab = amax.view(torch.int32)
    exp_bits = ab >> 23
    mant = ab & 0x7FFFFF
    er = exp_bits - 2 + (mant > FP32_HALF_MANTISSA).to(torch.int32)
    er = er.clamp_min(1)
    return torch.where(ab != 0, er, torch.full_like(er, E8M0_BIAS))


def _nibbles(x: torch.Tensor, er: torch.Tensor) -> torch.Tensor:
    """Pass 3: descale by the E8M0 exponent, round to E2M1, keep the sign."""
    off = (er - E8M0_BIAS) << 3
    mag = x.view(torch.int32) & FP32_MAG_MASK
    idx = ((mag >> LUT_SHIFT) - off).clamp_min(0)
    assert int(idx.max()) < LUT_LEN, f"LUT index {int(idx.max())} out of range"
    code = LUT[idx.long()]
    # vlts(x, 0.0): a float compare, so -0.0 is NOT negative -- same as the
    # golden's ``values < 0``.
    sign = torch.where(x < 0, torch.full_like(code, 8), torch.zeros_like(code))
    return (code | sign).to(torch.uint8)


def emulate_mx_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Register-level emulation of ``VectorQ.epilogue``. x is (T, N, D) fp32."""
    t, n, d = x.shape
    groups = ceil_div(d, MX_GROUP_SIZE)
    er = _e8m0_from_amax(_group_amax(x))
    er_per_elem = er.repeat_interleave(MX_GROUP_SIZE, dim=-1)[..., :d]
    nib = _nibbles(x.contiguous(), er_per_elem)
    packed = (nib[..., 0::2].to(torch.int16) | (nib[..., 1::2].to(torch.int16) << 4))
    packed = packed.to(torch.uint8).contiguous()

    pair_groups = ceil_div(d, 64)
    pair_elems = pair_groups * 2
    if groups < pair_elems:
        er = torch.nn.functional.pad(er, (0, pair_elems - groups), value=E8M0_BIAS)
    descale = er.to(torch.uint8).reshape(t, n, pair_groups, 2).contiguous()
    return packed, descale


def _assert_bit_exact(x: torch.Tensor) -> None:
    want_q, want_dsc = mx_quant_along_last(x.clone())
    got_q, got_dsc = emulate_mx_quant(x.clone())
    bad = (got_dsc != want_dsc).nonzero()
    assert not len(bad), (
        f"{len(bad)} descale bytes differ, first at {tuple(bad[0].tolist())}: "
        f"golden={int(want_dsc[tuple(bad[0].tolist())])} got={int(got_dsc[tuple(bad[0].tolist())])}"
    )
    bad = (got_q != want_q).nonzero()
    assert not len(bad), (
        f"{len(bad)} q bytes differ, first at {tuple(bad[0].tolist())}: "
        f"golden={int(want_q[tuple(bad[0].tolist())]):#04x} "
        f"got={int(got_q[tuple(bad[0].tolist())]):#04x}"
    )


def test_operator_shape_random():
    torch.manual_seed(0)
    _assert_bit_exact(torch.randn(64, 32, 128) * 3.0)


def test_scaled_magnitudes():
    """Every binade the GEMM can plausibly produce."""
    torch.manual_seed(1)
    base = torch.randn(8, 32, 128)
    for exponent in (-40, -12, -1, 0, 1, 12, 40):
        _assert_bit_exact(base * (2.0**exponent))


def test_exact_codebook_and_midpoints():
    """Ties must round away from zero, and exact codebook values must be exact."""
    from golden import E2M1_POS

    probes = list(E2M1_POS) + [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    probes = [p for p in probes] + [-p for p in probes]
    values = torch.tensor(probes, dtype=torch.float32)
    row = torch.zeros(128)
    row[: values.numel()] = values
    row[values.numel()] = 6.0  # pin the group amax so scale == 1.0
    _assert_bit_exact(row.reshape(1, 1, 128))
    # And one ulp either side of every midpoint.
    for delta in (-(2**-20), 2**-20):
        _assert_bit_exact((row * (1.0 + delta)).reshape(1, 1, 128))


def test_signed_zero_encodes_as_positive_zero():
    """-0.0 must give nibble 0x0: the golden's sign test is ``q < 0``."""
    row = torch.zeros(128)
    row[0] = -0.0
    row[1] = -1.5
    row[2] = 6.0
    _assert_bit_exact(row.reshape(1, 1, 128))
    got_q, _ = emulate_mx_quant(row.reshape(1, 1, 128))
    assert int(got_q[0, 0, 0]) & 0x0F == 0x0


def test_all_zero_group_uses_unit_scale():
    x = torch.zeros(2, 32, 128)
    _assert_bit_exact(x)
    _, dsc = emulate_mx_quant(x)
    assert int(dsc.min()) == E8M0_BIAS and int(dsc.max()) == E8M0_BIAS


def test_wide_dynamic_range_inside_a_group():
    """Elements far below the group amax must clamp to code 0 with their sign."""
    row = torch.zeros(128)
    row[0] = 1e30
    row[1] = -1e-3
    row[2] = -1e30
    row[3] = 1e-3
    _assert_bit_exact(row.reshape(1, 1, 128))


def test_tiny_amax_subnormal_scale_region():
    """amax < 1.5*2^-125 is where the golden's amax/6.0 goes subnormal.

    This is the case the ``er.clamp_min(1)`` in :func:`_e8m0_from_amax` exists
    for.  Elements stay normal (>= 2^-126); subnormal *elements* are out of
    domain, see :func:`test_known_divergence_needs_unreachable_dynamic_range`.
    """
    generator = torch.Generator().manual_seed(2)
    shape = (4, 32, 128)
    for hi in (1, 2, 3, 4):
        mantissa = torch.randint(
            0, 2**23, shape, dtype=torch.int64, generator=generator
        ).to(torch.int32)
        exponent = torch.randint(
            1, hi + 1, shape, dtype=torch.int64, generator=generator
        ).to(torch.int32)
        sign = torch.randint(0, 2, shape, dtype=torch.int64, generator=generator)
        bits = mantissa | (exponent << 23) | (sign.to(torch.int32) << 31)
        x = bits.view(torch.float32)
        assert (x.abs() >= 2.0**-126).all()  # every element still normal
        _assert_bit_exact(x)


def _fuzz_bit_patterns(spread: int, seed: int) -> torch.Tensor:
    """Random fp32 bit patterns with a bounded per-group exponent spread."""
    generator = torch.Generator().manual_seed(seed)
    shape = (16, 32, 128)
    mantissa = torch.randint(
        0, 2**23, shape, dtype=torch.int64, generator=generator
    ).to(torch.int32)
    # One base exponent per MX group, then a bounded per-element deviation, so
    # the within-group dynamic range stays inside what an fp32 GEMM can produce.
    base = torch.randint(
        60 + spread // 2, 200 - spread // 2, (16, 32, 4), dtype=torch.int64,
        generator=generator,
    ).repeat_interleave(MX_GROUP_SIZE, dim=-1)
    deviation = torch.randint(
        -(spread // 2), spread // 2 + 1, shape, dtype=torch.int64, generator=generator
    )
    exponent = (base + deviation).to(torch.int32)
    sign = torch.randint(0, 2, shape, dtype=torch.int64, generator=generator)
    bits = mantissa | (exponent << 23) | (sign.to(torch.int32) << 31)
    x = bits.view(torch.float32)
    assert torch.isfinite(x).all()
    return x


@pytest.mark.parametrize("spread", [0, 8, 60, 120])
def test_random_bit_patterns(spread):
    """Fuzz over exponent spreads up to 2^120 within a single MX group."""
    _assert_bit_exact(_fuzz_bit_patterns(spread, seed=3 + spread))


def test_lut_index_stays_in_range_without_an_upper_clamp():
    """The gather has only ``vmaxs(idx, 0)``; the high side must be structural.

    ``2^(er-127) >= amax/6`` gives ``|x|/scale <= 6``, whose bucket is
    ``8*129 + 4 = 1036``.  Every other test in this module also runs the
    assertion inside :func:`_nibbles`; this one pins the bound itself so a
    future change to LUT_SHIFT or the offset cannot silently start reading past
    the table.
    """
    # A group whose amax is exactly 6*2^k hits the bound: scale == 2^k, so the
    # largest element descales to exactly 6.0.
    cases = [torch.full((1, 1, 128), 6.0 * 2.0**k) for k in (-40, 0, 40)]
    cases += [_fuzz_bit_patterns(spread, seed=17 + spread) for spread in (0, 60, 130)]
    worst = 0
    for x in cases:
        er = _e8m0_from_amax(_group_amax(x))
        off = (er - E8M0_BIAS).repeat_interleave(MX_GROUP_SIZE, dim=-1)[
            ..., : x.shape[-1]
        ] << 3
        mag = x.contiguous().view(torch.int32) & FP32_MAG_MASK
        worst = max(worst, int(((mag >> LUT_SHIFT) - off).max()))
    assert worst == 1036, worst  # bound is reached, so this is tight
    assert 1036 < LUT_LEN


def test_known_divergence_needs_unreachable_dynamic_range():
    """Document the one input class where the integer path leaves the golden.

    When ``|x| / scale`` underflows fp32 to +-0 the golden's sign test
    (``q < 0``, and -0.0 is not negative) drops the sign, while the integer path
    keeps it.  Note the golden is internally inconsistent here: a negative value
    that rounds to magnitude zero with a *nonzero* quotient does keep its sign
    (0x8), which is why the kernel cannot simply drop the sign for code 0.

    Matching this exactly needs the group's flush threshold
    ``max((er - 150) << 23, 0)`` gathered next to ``off``, then a compare and a
    select: three instructions per register on the path that is already the
    critical one (~23 ops, so ~13%), including a second UB gather.  It buys
    nothing reachable.  The divergence needs ``amax/|x| > 2^147`` inside one
    32-element group, i.e. one lane of the fp32 GEMM output smaller than another
    by 2^147 relative.  Cancellation in fp32 bottoms out near 2^-24 relative,
    not 2^-147, so the only attainable case is an output that is *exactly* zero
    -- and +-0.0 is handled correctly by the float compare.  Subnormal
    *elements* diverge for the same underflow reason and are equally
    unreachable from a real GEMM output.
    """
    row = torch.zeros(128)
    row[0] = 1e26
    row[1] = -8.1e-21  # q flushes to -0.0
    want_q, _ = mx_quant_along_last(row.reshape(1, 1, 128).clone())
    got_q, _ = emulate_mx_quant(row.reshape(1, 1, 128).clone())
    assert int(want_q[0, 0, 0]) >> 4 == 0x0
    assert int(got_q[0, 0, 0]) >> 4 == 0x8
    ratio = float(row[0].abs() / row[1].abs())
    assert ratio > 2.0**147
