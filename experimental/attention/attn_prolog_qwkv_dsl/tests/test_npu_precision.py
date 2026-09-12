# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real-hardware precision tests: the fused kernel vs the CPU golden.

Skipped unless an NPU is visible.  Needs a complete CANN install (one that ships
``libopapi.so``); the kernel-development toolkit alone cannot move tensors to the
device.  See README.md for the environment.

``q`` and ``descale_q`` are compared byte-for-byte because the whole MXFP4
epilogue is integer math, so anything short of bit-exact is a bug.  ``w`` is a
BF16 GEMM reduced in fp32, so it gets a tolerance.
"""

from __future__ import annotations

import pytest
import torch

from golden import attn_prolog_qwkv_golden, make_inputs
from op import attn_prolog_qwkv

DIM, Q_LORA, N, D, DR = 5120, 1280, 32, 128, 64

torch_npu = pytest.importorskip("torch_npu", reason="torch_npu not installed")

if not torch.npu.is_available():  # pragma: no cover - depends on the host
    pytest.skip("no NPU visible", allow_module_level=True)


def _to_npu(inputs):
    return {
        name: getattr(inputs, name).npu()
        for name in (
            "x", "qr", "wqb", "ww",
            "descale_qr", "descale_wqb", "rope_sin", "rope_cos",
        )
    }


def _run(t: int, *, seed: int = 0, softmax_scale: float = 1.0, rope_identity: bool = False):
    inputs = make_inputs(
        t, DIM, Q_LORA, N, D, DR,
        softmax_scale=softmax_scale, rope_identity=rope_identity, seed=seed,
    )
    want = attn_prolog_qwkv_golden(inputs)
    got_q, got_dsc, got_w = attn_prolog_qwkv(
        **_to_npu(inputs), softmax_scale=softmax_scale
    )
    return want, (got_q.cpu(), got_dsc.cpu(), got_w.cpu())


def _assert_matches(want, got, *, w_rtol=2e-2, w_atol=2e-2):
    got_q, got_dsc, got_w = got

    assert got_dsc.shape == want.descale_q.shape
    n_bad = int((got_dsc != want.descale_q).sum())
    assert n_bad == 0, f"{n_bad} descale_q bytes differ"

    assert got_q.shape == want.q.shape
    bad = got_q != want.q
    n_bad = int(bad.sum())
    assert n_bad == 0, (
        f"{n_bad}/{want.q.numel()} q bytes differ, first at "
        f"{tuple(int(v) for v in torch.nonzero(bad)[0])}"
    )

    assert got_w.shape == want.w.shape
    torch.testing.assert_close(got_w, want.w, rtol=w_rtol, atol=w_atol)


@pytest.mark.parametrize("t", [128, 256])
def test_matches_golden(t):
    _assert_matches(*_run(t))


@pytest.mark.parametrize("t", [136, 144, 200, 240])
def test_tail_tile_not_a_multiple_of_the_row_tile(t):
    """A short last row tile hits two separate paths.

    q goes out through the vector epilogue, which clips the UB->GM store to the
    live row count.  w is reduced straight into L0C, and a short tile there used
    to shorten x_l1's actual shape and silently corrupt the upper half of the
    head axis, so these sizes are the regression guard for padding x to whole
    tiles.  The tails cover both a multiple of 16 and not.
    """
    _assert_matches(*_run(t))


def test_rope_identity_isolates_the_gemm_and_quant():
    """cos=1, sin=0 -- any mismatch here is the GEMM or the quant, not RoPE."""
    _assert_matches(*_run(128, rope_identity=True))


def test_softmax_scale_is_applied_to_w():
    """softmax_scale rides the L0C->GM fixpipe as deq_scale_val."""
    _assert_matches(*_run(128, softmax_scale=0.08838834764831845))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_seeds(seed):
    _assert_matches(*_run(128, seed=seed))
