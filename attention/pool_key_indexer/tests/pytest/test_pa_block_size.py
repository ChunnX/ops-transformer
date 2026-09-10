# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PA block_size=2048 规格放开专项测试(A2/A3/A5 全架构)。

背景: host tiling 原 PA block_size 上限 1024(lightning_indexer 家族沿袭的
规格保守值, 非实现能力上限)。arch22(A2/A3) 与 arch35(A5) 的 PA 寻址均按
运行时 blockSize 动态计算, 现统一放开至 2048(见
op_host/pool_key_indexer_tiling.h PKI_BLOCK_SIZE_LIMIT)。

覆盖点:
  1. 精度: bs=2048 恰满一块 / 跨 1~2 个块边界 / 多 batch(含 0 池) /
     TND+PA / pool_size=1 退化 / 0 轴非连续 / bf16 / mask 0/3 / tail;
  2. 边界: bs=2032(127*16, 上限内非 2 次幂)通过, bs=1024(旧上限)回归;
  3. 负向: bs=2064(>2048) 与 bs=2040(非 16 倍数)在 host tiling 被拒
     (错误信息含 "blockSize"), 确认新校验真实生效。
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from pki_common import (  # noqa: E402
    assert_output_spec,
    build_case_inputs,
    run_golden,
    run_op_eager,
)
from result_compare import compare_indices, compare_values  # noqa: E402

FP16 = torch.float16
BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# 1. bs=2048 精度用例(标杆 vs NPU eager, 集合比对口径与精度套件一致)
# ---------------------------------------------------------------------------

BS2048_CASES = [
    dict(
        id="bs2048_exact_one_block",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[32],
        k_pools=[2048],  # 恰满一块(无尾块 padding, 块边界即序列尾)
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=2048,
        key_non_contig=False,
        tail=[0],
    ),
    dict(
        id="bs2048_cross_one_boundary",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[32],
        k_pools=[3000],  # 1 满块 + 952: 块边界落在 s2 基本块(512)中间
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=2048,
        key_non_contig=False,
        tail=[15],
    ),
    dict(
        id="bs2048_cross_two_boundaries",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[16],
        k_pools=[4200],  # 2 满块 + 104: 连续跨两个块边界
        n1=4,
        topk=256,
        pool_size=16,
        mask_mode=0,
        return_value=False,
        pa_block_size=2048,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="bs2048_multibatch_zeropool",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[16, 16, 16],
        k_pools=[2048, 1999, 0],  # 不等池数 + 0 池 batch(PA 允许)
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=0,
        return_value=True,
        pa_block_size=2048,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="bs2048_tnd_pa",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[32, 16],
        k_pools=[2500, 2100],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=2048,
        key_non_contig=False,
        tail=[3, 0],
    ),
    dict(
        id="bs2048_ps1_degenerate",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[16],
        k_pools=[2100],  # pool_size=1 退化(块内每行一个 token 池)
        n1=4,
        topk=64,
        pool_size=1,
        mask_mode=3,
        return_value=False,
        pa_block_size=2048,
        key_non_contig=False,
        tail=[0],
    ),
    dict(
        id="bs2048_noncontig_tail",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[16, 16],
        k_pools=[3000, 1000],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=2048,
        key_non_contig=True,  # 0 轴非连续: stride0 = 2*blockSize*N2*D
        tail=[15, 7],
    ),
    dict(
        id="bs2048_bf16_mask0",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=BF16,
        q_lens=[16],
        k_pools=[4096],  # 2 满块
        n1=4,
        topk=128,
        pool_size=8,
        mask_mode=0,
        return_value=False,
        pa_block_size=2048,
        key_non_contig=False,
        tail=None,
    ),
]


@pytest.mark.parametrize("cfg", BS2048_CASES, ids=[c["id"] for c in BS2048_CASES])
def test_pa_bs2048_precision(cfg):
    """bs=2048 精度: 标杆 vs NPU eager, 规格断言 + indices 集合比对。"""
    cfg = {**cfg, "seed": cfg.get("seed", 20480)}
    inputs = build_case_inputs(cfg)
    g_indices, g_values = run_golden(inputs, cfg, ref_device="cpu")
    npu_indices, npu_values = run_op_eager(inputs, cfg)

    assert_output_spec(npu_indices, npu_values, cfg)
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{cfg['id']}] indices set comparison failed: "
        f"layout=({cfg['layout_q']},{cfg['layout_k']}) ps={cfg['pool_size']} "
        f"topk={cfg['topk']} mask={cfg['mask_mode']} k_pools={cfg['k_pools']}"
    )
    if cfg["return_value"]:
        compare_values(g_values, npu_values, cfg["dtype"], label=cfg["id"])


# ---------------------------------------------------------------------------
# 2. 上限内边界 + 旧上限回归
# ---------------------------------------------------------------------------


def _make_boundary_cfg(bs, k_pools):
    return dict(
        id=f"boundary_bs{bs}",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[16],
        k_pools=k_pools,
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=bs,
        key_non_contig=False,
        tail=None,
        seed=20481,
    )


@pytest.mark.parametrize(
    "bs,k_pools",
    [(2032, [2032 + 100]), (1024, [1024 + 100])],  # 上限内最大非 2 次幂 / 旧上限回归
    ids=["bs2032_largest_valid_nonpow2", "bs1024_old_limit_regression"],
)
def test_pa_boundary_block_sizes(bs, k_pools):
    """bs=2032(127*16, 上限内最大非 2 次幂)与 bs=1024(旧上限)精度回归。"""
    cfg = _make_boundary_cfg(bs, k_pools)
    inputs = build_case_inputs(cfg)
    g_indices, g_values = run_golden(inputs, cfg, ref_device="cpu")
    npu_indices, npu_values = run_op_eager(inputs, cfg)
    assert_output_spec(npu_indices, npu_values, cfg)
    assert compare_indices(g_indices, npu_indices), f"[{cfg['id']}] indices mismatch"
    compare_values(g_values, npu_values, cfg["dtype"], label=cfg["id"])


# ---------------------------------------------------------------------------
# 3. 负向: 超上限 / 非 16 倍数在 host tiling 被拒
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bs",
    [2064, 2040],  # 129*16 > 2048 超上限 / 非 16 倍数(2040 = 127.5*16)
    ids=["bs2064_over_limit", "bs2040_not_mult16"],
)
def test_pa_invalid_block_size_rejected(bs):
    """非法 block_size 被 host tiling 拒绝(错误信息含 blockSize)。"""
    cfg = _make_boundary_cfg(bs, [64])  # 池数任意, 校验发生在 shape 维度
    inputs = build_case_inputs(cfg)
    with pytest.raises(Exception, match="blockSize"):
        run_op_eager(inputs, cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
