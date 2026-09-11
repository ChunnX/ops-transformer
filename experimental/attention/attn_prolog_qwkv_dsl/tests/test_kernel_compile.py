# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile-time tests for the fused NPU kernel. No NPU device required.

These run the DSL front end plus the whole CANNIR pass pipeline (through
``CANNBOTDSL_PIPE_STAGE=translate``), so they catch channel/ledger, layout,
raw-VF and budget regressions without hardware.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cannbotdsl")

from cannbotdsl import TensorSpec, dtypes  # noqa: E402

from golden import (  # noqa: E402
    E2M1_POS,
    E8M0_BIAS,
    fp32_to_e2m1_nibble,
)
from kernel import (  # noqa: E402
    LUT_LEN,
    LUT_SHIFT,
    UB_BYTES,
    AttnPrologQwkv,
    Plan,
    build_e2m1_lut,
)

DIM, Q_LORA, N, D, DR = 5120, 1280, 32, 128, 64


def _specs(t: int):
    groups = D // 32
    return (
        TensorSpec((t, DIM), dtypes.bfloat16),
        TensorSpec((t, Q_LORA), dtypes.uint8),
        TensorSpec((N * D, Q_LORA), dtypes.uint8),
        TensorSpec((N, DIM), dtypes.bfloat16),
        TensorSpec((t, Q_LORA // 64, 2), dtypes.uint8),
        TensorSpec((N * D, Q_LORA // 64, 2), dtypes.uint8),
        TensorSpec((t, DR), dtypes.float32),
        TensorSpec((t, DR), dtypes.float32),
        TensorSpec((LUT_LEN,), dtypes.int32),
        TensorSpec((1,), dtypes.float32),
        TensorSpec((t, N * D // 2), dtypes.uint8),
        TensorSpec((t, N * groups), dtypes.uint8),
        TensorSpec((t, N), dtypes.float32),
    )


def test_plan_fits_on_chip_memory():
    p = Plan(4096, DIM, Q_LORA, N, D, DR)
    assert p.base_m == 128
    assert p.q_lora % p.k_l1_q == 0
    assert p.k_l1_q % p.base_k_q == 0
    assert p.w_interleaved and p.w_per_head >= 1
    assert p.k_l1_tiles_w == p.heads_per_block * p.w_per_head or p.w_separate
    assert p.dim == p.k_l1_tiles_w * p.k_l1_w
    assert p.n_head_blocks == 1 and p.heads_per_block == N
    assert p.used_cores == 32


def test_plan_splits_heads_to_fill_the_chip():
    """T-tiles < 32 used to leave most AICs idle; heads are independent GEMMs."""
    for t, n_tiles, n_hb, hpb in (
        (128, 1, 32, 1),
        (200, 2, 16, 2),
        (256, 2, 16, 2),
        (1024, 8, 4, 8),
        (2048, 16, 2, 16),
        (4096, 32, 1, 32),
    ):
        p = Plan(t, DIM, Q_LORA, N, D, DR)
        assert p.n_tiles == n_tiles, (t, p.n_tiles)
        assert p.n_head_blocks == n_hb, (t, p.n_head_blocks)
        assert p.heads_per_block == hpb, (t, p.heads_per_block)
        assert p.used_cores == n_tiles * n_hb
        assert p.used_cores == 32
        assert p.base_n_w == N
        assert p.w_leader_only == (n_hb > 1)
        if p.w_separate:
            assert p.w_per_head == 0
        else:
            assert p.w_interleaved and p.w_per_head >= 1
    for name, cap in (("L1", 512 * 1024), ("L0A", 65536), ("L0B", 65536)):
        assert p.bytes_used[name] <= cap, (name, p.bytes_used)
    assert p.bytes_used["UB"] <= UB_BYTES
    assert p.rows_vec * 2 == p.base_m


def test_plan_rejects_unsupported_geometry():
    with pytest.raises(NotImplementedError):
        Plan(128, DIM, Q_LORA, N, 96, 32)
    with pytest.raises(NotImplementedError):
        Plan(128, DIM, 1000, N, D, DR)
    with pytest.raises(NotImplementedError):
        Plan(128, DIM, Q_LORA, N, 256, 128)


def test_e2m1_lut_reproduces_golden_rounding():
    """The LUT must give the golden's nibble for every fp32 in E2M1's range."""
    lut = build_e2m1_lut()
    assert lut.shape == (LUT_LEN,)
    probes = [0.0]
    for value in E2M1_POS:
        probes += [value, value * (1 - 2**-20), value * (1 + 2**-20)]
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    for mid in midpoints:
        probes += [mid, mid * (1 - 2**-20), mid * (1 + 2**-20)]
    probes += torch.rand(4096).mul(6.0).tolist()
    values = torch.tensor(probes, dtype=torch.float32).abs().clamp(max=6.0)
    bits = values.view(torch.int32) & 0x7FFFFFFF
    idx = (bits >> LUT_SHIFT).clamp_min(0)
    assert int(idx.max()) < LUT_LEN
    got = lut[idx.long()].to(torch.uint8)
    want = fp32_to_e2m1_nibble(values) & 0x7
    assert torch.equal(got, want)


def test_e8m0_identity_matches_golden():
    """``exp(amax) - 2 + (mant > 1.5)`` == golden's ``ceil(log2(amax/6))``."""
    boundary = [0x400000 + delta for delta in (-2, -1, 0, 1, 2)]
    bits = []
    for exponent in range(1, 255):
        for mant in boundary + [0, 1, 0x7FFFFF]:
            bits.append((exponent << 23) | mant)
    amax = torch.tensor(bits, dtype=torch.int32).view(torch.float32)
    ab = amax.view(torch.int32)
    kernel_er = (ab >> 23) - 2 + ((ab & 0x7FFFFF) > 0x400000).to(torch.int32)

    raw = (amax / 6.0).view(torch.int32)
    golden_er = (raw >> 23) + ((raw & 0x7FFFFF) != 0).to(torch.int32)

    finite = torch.isfinite(amax) & (amax > 0) & torch.isfinite(amax / 6.0)
    normal = (raw >> 23) > 0
    keep = finite & normal
    assert torch.equal(kernel_er[keep], golden_er[keep])

    zero = torch.zeros(1)
    assert int((zero.view(torch.int32) >> 23)) == 0
    assert E8M0_BIAS == 127


@pytest.mark.parametrize("t", [128, 200, 256, 4096])
def test_fused_kernel_compiles(t, monkeypatch):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    op = AttnPrologQwkv(t, DIM, Q_LORA, N, D, DR)
    op.run.compile(*_specs(t))
