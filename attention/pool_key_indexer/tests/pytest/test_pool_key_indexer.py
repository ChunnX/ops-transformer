# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Pytest test cases for PoolKeyIndexer torch extension.

Reference:
  - Golden: tests/pytest/pool_key_indexer_reference.py (独立 CPU 标杆,
    与算子非量化路径计算过程一致, 可单独运行)

比对策略(集合比对):
  1. indices(第一输出): 逐行有效索引(>=0)排序集合比对, 100% 行通过
     作为唯一精度门禁(严格标准, 不允许任何行差异; 允许 TopK 输出顺序
     不同, 但选中索引集合不得有任何行差异);
  2. values(第二输出): 逐行有效值(有限值)排序数值比对, 仅打印参考日志,
     不作为通过/失败判据;
  3. 不使用任何归一化/重排/重建后处理, 直接对算子原始输出比对。

Coverage:
  - Layout: BSND, TND, PA_BBND (含 PA key 0 轴非连续)
  - dtype: fp16, bf16
  - mask_mode: 0 (non-causal), 3 (causal)
  - pool_size: 1 (degenerate), 8, 16 (expanded), 32, 64, 128 (upper limit)
  - return_value: True, False
  - topk: 16..256 base, 512/2048 (recommended), 3072/4096/8192 (special values)
  - Known defect (arch22): pool_size>=32 with topk>=6144 triggers aicore
    exception (EE9999/507015) and poisons the process; crash combos are
    documented in TEST_CASES comments but NOT executed (see NOTE).
    2026-08-26: arch35 复测不复现, 已纳入
    test_pool_key_indexer_known_defects.py 回归保护(见该文件与
    docs/build_fix_record.md); arch22 待责任人处理
  - pool_tail_k: random, deterministic 0 / pool_size-1 / per-batch mixed
  - n1: 1 (min), 4..32, 64 (max); S1=1 minimal shape
  - PA: unequal pool counts per batch (block_table -1 slots + padded tail
    block), block_size 16/32/64, batch up to 4
"""

import os
import sys
import pytest
import torch
import torch_npu

# Import comparison utilities
sys.path.insert(0, os.path.dirname(__file__))
from result_compare import compare_indices, compare_values

# Import standalone CPU golden reference
from pool_key_indexer_reference import pool_key_indexer_reference

# Import torch extension
from cann_ops_transformer import pool_key_indexer

HEAD_DIM = 128
N2 = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_prefix_seq(seq_lens):
    """构造 TND 前缀和 actual_seq(与用例既有口径一致: 多 batch 均分总长)。"""
    total = sum(seq_lens)
    b = len(seq_lens)
    if b == 1:
        return torch.tensor([total], dtype=torch.int64)
    lens = [total // b] * b
    lens[-1] += total - sum(lens)
    return torch.tensor(lens, dtype=torch.int64).cumsum(0).to(torch.int64)


def _build_bsnd_inputs(batch, q_seq, k_seq, n1, pool_size, dtype, seed=42):
    """Construct BSND-layout inputs on CPU.

    k_seq 可为 int(每 batch 等长) 或 list(每 batch token 数不等, 仅 PA 场景,
    pool_key 返回扁平 (sum_pools, N2, D))。"""
    torch.manual_seed(seed)
    dt = dtype
    query = torch.randn(batch, q_seq, n1, HEAD_DIM, dtype=dt)
    if isinstance(k_seq, (list, tuple)):
        k_pool_lens = [kl // pool_size for kl in k_seq]
        pool_key = torch.randn(sum(k_pool_lens), N2, HEAD_DIM, dtype=dt)
    else:
        k_seq_pools = k_seq // pool_size
        pool_key = torch.randn(batch, k_seq_pools, N2, HEAD_DIM, dtype=dt)
    weights = torch.randn(batch, q_seq, n1, dtype=dt)
    pool_tail_k = torch.randint(0, pool_size, (batch,), dtype=torch.int64)
    return query, pool_key, weights, pool_tail_k


def _build_tnd_inputs(q_lens, k_lens, n1, pool_size, dtype, seed=42):
    """Construct TND-layout inputs on CPU."""
    torch.manual_seed(seed)
    dt = dtype
    total_q = sum(q_lens)
    k_pool_lens = [kl // pool_size for kl in k_lens]
    total_k_pools = sum(k_pool_lens)
    query = torch.randn(total_q, n1, HEAD_DIM, dtype=dt)
    pool_key = torch.randn(total_k_pools, N2, HEAD_DIM, dtype=dt)
    weights = torch.randn(total_q, n1, dtype=dt)
    pool_tail_k = torch.randint(0, pool_size, (len(q_lens),), dtype=torch.int64)
    # actual_seq_q 为 query token 前缀和; actual_seq_k 为累计 pool 数
    # (与 pool_key 的 T 维一致, 亦即本用例的均分口径)
    actual_seq_q = _normalize_prefix_seq(q_lens)
    actual_seq_k = _normalize_prefix_seq(k_pool_lens)
    return query, pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k


def _make_actual_seq_noncum(pool_lens):
    """PA_BBND: actual_seq_k 为每 batch pool 数(非前缀和)。"""
    return torch.tensor(pool_lens, dtype=torch.int64)


def _pack_pa_key(pool_key_bsnd, k_pool_lens, pool_size, block_size=16):
    """将逻辑 (T2_pool, N2, D) 打包为 PA 物理块布局 (blockNum, blockSize, N2, D)。

    与 LIV2 语义一致: PA key 块内每行 = 1 个池(key 不做 token 展开)。
    每 batch 的池独立连续填块(与 _build_block_table 的块数口径一致),
    物理块随机排布, 尾块不足补零。返回 (key_pa, block_size, block_num)。
    block_size 需为 16 的倍数(算子约束), 尾块 padding 行补零。
    """
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    n2, d = pool_key_bsnd.shape[1], pool_key_bsnd.shape[2]
    blocks = torch.zeros(block_num, block_size, n2, d, dtype=pool_key_bsnd.dtype)
    # 逐 batch 填块
    pool_start = 0
    blk_start = 0
    for b, nb in enumerate(blocks_per_batch):
        pools = k_pool_lens[b]
        seg = pool_key_bsnd[pool_start : pool_start + pools]  # (pools, n2, d)
        pad = nb * block_size - pools
        if pad:
            seg = torch.cat([seg, torch.zeros(pad, n2, d, dtype=seg.dtype)], dim=0)
        blocks[blk_start : blk_start + nb] = seg.view(nb, block_size, n2, d)
        pool_start += pools
        blk_start += nb
    perm = torch.randperm(block_num)
    return blocks[perm].contiguous(), block_size, block_num


def _build_block_table(batch, k_pool_lens, pool_size, block_size=16):
    """构造 (B, maxBlocks) int32 物理块映射, 空槽 -1。

    PA key 块内每行=1池(LIV2 语义): 每块容纳 block_size 个池,
    块数 = ceil(pools/block_size)。各 batch 池数不等时, 短 batch 的
    尾部槽位保持 -1(算子按 actual_seq_k 只读有效槽位)。
    返回 (block_table, 物理块总数)。
    """
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    max_blocks = max(blocks_per_batch)
    total_blocks = sum(blocks_per_batch)
    perm = torch.randperm(total_blocks)
    bt = torch.full((batch, max_blocks), -1, dtype=torch.int32)
    cur = 0
    for b, nb in enumerate(blocks_per_batch):
        for i in range(nb):
            bt[b, i] = perm[cur]
            cur += 1
    return bt, total_blocks


def _make_pa_key_non_contig(key_pa, block_size, block_num, dtype):
    """构造 PA key 0 轴非连续输入(keyStride0 运行时 stride 分支)。

    参考 development_blueprint.md §二·补: 物理块嵌入 2 倍缓冲的偶数行,
    视图 stride[0] = 2 * block_size * N2 * D (含 padding)。
    返回 (cpu_view, npu_builder):
      - cpu_view: 供 golden 直接消费(torch 按 stride 索引);
      - npu_builder: NPU 侧先搬大缓冲再切片 —— 直接对非稠密视图 .to(npu)
        会被拍平为连续, 丢失 stride 信息, 无法覆盖 kernel 非连续寻址。
    """
    big = torch.zeros(2 * block_num, block_size, N2, HEAD_DIM, dtype=dtype)
    big[0::2] = key_pa
    view = big[::2]

    def npu_builder(dev, _big=big):
        return _big.to(dev)[::2]

    return view, npu_builder


def _run_npu(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=128,
    pool_size=1,
    mask_mode=0,
    quant_mode=-1,
    return_value=False,
    device_id=0,
    pool_key_npu_fn=None,
):
    """Move inputs to NPU and call the torch extension.

    pool_key_npu_fn: 可选 callable(dev) -> npu tensor, 用于 0 轴非连续
    key 的 NPU 侧构造(大缓冲搬运后切片), 绕过 .to(npu) 的连续化。"""
    dev = f"npu:{device_id}"
    q = query.to(dev)
    k = pool_key_npu_fn(dev) if pool_key_npu_fn is not None else pool_key.to(dev)
    w = weights.to(dev)
    ptk = pool_tail_k.to(dev)
    asq = actual_seq_q.to(dev) if actual_seq_q is not None else None
    ask = actual_seq_k.to(dev) if actual_seq_k is not None else None
    bt = block_table.to(dev) if block_table is not None else None

    indices, values = pool_key_indexer(
        q,
        k,
        w,
        ptk,
        actual_seq_q=asq,
        actual_seq_k=ask,
        block_table=bt,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        return_value=return_value,
    )
    torch.npu.synchronize()
    return indices.cpu(), values.cpu()


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _npu_init():
    torch_npu.npu.set_device(0)


# Test case configurations: (id, layout_q, layout_k, dtype, batch, q_seq, k_seq,
#                            n1, topk, pool_size, mask_mode, return_value,
#                            tail_spec, pa_block_size, key_non_contig)
#   tail_spec:      None=随机生成 / int=所有 batch 定值 / list=每 batch 定值
#                   (取值须在 [0, pool_size-1])
#   pa_block_size:  PA 场景物理块大小(16 的倍数, 上限 1024), 非 PA 忽略
#   key_non_contig: PA key 0 轴非连续输入(keyStride0 分支), 仅 PA 支持
TEST_CASES = [
    # --- Degenerate (pool_size=1) ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        1,
        0,
        False,
        None,
        16,
        False,
        id="bsnd_fp16_ps1_nocausal_noval",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        1,
        3,
        False,
        None,
        16,
        False,
        id="bsnd_fp16_ps1_causal_noval",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.bfloat16,
        2,
        32,
        128,
        4,
        64,
        1,
        0,
        True,
        None,
        16,
        False,
        id="bsnd_bf16_ps1_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        2,
        48,
        192,
        16,
        128,
        1,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps1_causal_val",
    ),
    # --- Expanded (pool_size>1) ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        16,
        0,
        False,
        None,
        16,
        False,
        id="bsnd_fp16_ps16_nocausal_noval",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        64,
        256,
        8,
        128,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps16_causal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.bfloat16,
        2,
        32,
        256,
        16,
        128,
        8,
        0,
        True,
        None,
        16,
        False,
        id="bsnd_bf16_ps8_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        32,
        512,
        32,
        256,
        32,
        3,
        False,
        None,
        16,
        False,
        id="bsnd_fp16_ps32_causal_noval",
    ),
    # --- TND ---
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [32, 32],
        [64, 64],
        [128, 128],
        8,
        128,
        1,
        0,
        False,
        None,
        16,
        False,
        id="tnd_fp16_ps1_nocausal_noval",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [32, 32],
        [64, 64],
        [128, 128],
        8,
        128,
        16,
        3,
        True,
        None,
        16,
        False,
        id="tnd_fp16_ps16_causal_val",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.bfloat16,
        [16, 48],
        [32, 32],
        [64, 192],
        4,
        64,
        8,
        0,
        True,
        None,
        16,
        False,
        id="tnd_bf16_ps8_nocausal_val",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [16, 16, 16],
        [48, 48, 48],
        [96, 96, 96],
        16,
        128,
        16,
        3,
        False,
        None,
        16,
        False,
        id="tnd_fp16_ps16_causal_noval_3batch",
    ),
    # --- PA (layout_k=PA_BBND) ---
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        2,
        32,
        256,
        8,
        128,
        8,
        0,
        True,
        None,
        16,
        False,
        id="bsnd_pa_fp16_ps8_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        2,
        32,
        256,
        8,
        128,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_pa_fp16_ps16_causal_val",
    ),
    pytest.param(
        "TND",
        "PA_BBND",
        torch.float16,
        [16, 16],
        32,
        [128, 128],
        8,
        128,
        8,
        0,
        False,
        None,
        16,
        False,
        id="tnd_pa_fp16_ps8_nocausal_noval",
    ),
    pytest.param(
        "TND",
        "PA_BBND",
        torch.bfloat16,
        [16, 16],
        32,
        [128, 128],
        8,
        128,
        16,
        3,
        True,
        None,
        16,
        False,
        id="tnd_pa_bf16_ps16_causal_val",
    ),
    # --- sparseCount=1 边界(topk == pool_size): 尾区越写/清理缺陷回归 ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        48,
        800,
        8,
        16,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps16_topk16_causal_sparse1",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        48,
        48,
        8,
        16,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps16_topk16_causal_L48",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [16, 32],
        48,
        48,
        8,
        16,
        16,
        3,
        True,
        None,
        16,
        False,
        id="tnd_fp16_ps16_topk16_causal_L48",
    ),
    # --- 大 topk(含建议值 2048 与特殊值 3072/4096/8192) ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        32,
        2048,
        8,
        512,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps16_topk512_causal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        32,
        8192,
        8,
        2048,
        16,
        0,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps16_topk2048_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        8,
        4096,
        8,
        3072,
        1,
        3,
        False,
        None,
        16,
        False,
        id="bsnd_fp16_ps1_topk3072_causal_sparse3k",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        16,
        8192,
        8,
        4096,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps16_topk4096_causal_val",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [8, 8],
        [8, 8],
        [16384, 16384],
        4,
        8192,
        16,
        3,
        False,
        None,
        16,
        False,
        id="tnd_fp16_ps16_topk8192_causal_noval",
    ),
    # NOTE 已知缺陷(arch22/910B, 2026-08-24 二分定位): pool_size>=32 且
    # topk>=6144 的组合(如 TND ps32/topk6144、BSND ps64/topk8192, k=16384)
    # 触发 aicore exception(EE9999/507015), 且崩溃会污染进程内后续用例,
    # 因此不作为 xfail 用例纳入(避免 CI 雪崩)。
    # 2026-08-26 arch35(950) 复测: 该组合已由
    # test_pool_key_indexer_known_defects.py 覆盖且全部通过(arch35 TopK 为
    # VF 直通路径, 无 arch22 的 LD 跨核归并路径); arch22 侧待责任人修复,
    # 修复后可补充:
    #   - tnd_fp16_ps32_topk6144_causal
    #   - bsnd_fp16_ps64_topk8192_causal
    # --- pool_size 极值(64 / 上限 128, sparseCount 退化为 1~4) ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        32,
        2048,
        8,
        256,
        64,
        0,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_ps64_topk256_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        2,
        256,
        2048,
        8,
        128,
        128,
        3,
        True,
        127,
        16,
        False,
        id="bsnd_fp16_ps128_topk128_causal_tail127_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        16,
        8192,
        8,
        256,
        128,
        0,
        False,
        None,
        16,
        False,
        id="bsnd_fp16_ps128_topk256_nocausal_noval",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.bfloat16,
        [16, 32],
        [16, 32],
        [1024, 2048],
        4,
        256,
        64,
        3,
        True,
        None,
        16,
        False,
        id="tnd_bf16_ps64_topk256_causal_val",
    ),
    # --- pool_tail_k 确定性边界(0 / pool_size-1 / 每 batch 混合) ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        2,
        48,
        800,
        8,
        128,
        16,
        3,
        True,
        0,
        16,
        False,
        id="bsnd_fp16_ps16_tail0_causal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        2,
        48,
        800,
        8,
        128,
        16,
        0,
        True,
        15,
        16,
        False,
        id="bsnd_fp16_ps16_tail15_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        3,
        32,
        512,
        8,
        128,
        8,
        3,
        True,
        [0, 4, 7],
        16,
        False,
        id="bsnd_fp16_ps8_tail_mixed_causal_val",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.float16,
        [32, 32],
        [32, 32],
        [512, 512],
        8,
        128,
        16,
        3,
        False,
        15,
        16,
        False,
        id="tnd_fp16_ps16_tail15_causal_noval",
    ),
    # --- n1 极值(1 / 上限 64)与 S1=1 最小 shape ---
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        2,
        32,
        256,
        1,
        128,
        16,
        0,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_n1_1_ps16_nocausal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        16,
        512,
        64,
        128,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_n1_64_ps16_causal_val",
    ),
    pytest.param(
        "BSND",
        "BSND",
        torch.float16,
        1,
        1,
        256,
        8,
        128,
        16,
        3,
        True,
        None,
        16,
        False,
        id="bsnd_fp16_s1_1_ps16_causal_val",
    ),
    pytest.param(
        "TND",
        "TND",
        torch.bfloat16,
        [32],
        [32],
        [512],
        64,
        128,
        8,
        0,
        False,
        None,
        16,
        False,
        id="tnd_bf16_n1_64_ps8_nocausal_noval",
    ),
    # --- PA 细节: 不等池数(-1 空槽+尾块 padding)、block_size 32/64、B=4 ---
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        3,
        32,
        [200, 1000, 520],
        8,
        128,
        16,
        0,
        True,
        None,
        16,
        False,
        id="bsnd_pa_fp16_ps16_unequal_pools_val",
    ),
    pytest.param(
        "TND",
        "PA_BBND",
        torch.float16,
        [32, 16],
        32,
        [768, 256],
        8,
        128,
        16,
        3,
        True,
        None,
        16,
        False,
        id="tnd_pa_fp16_ps16_unequal_pools_causal_val",
    ),
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        2,
        32,
        [512, 256],
        8,
        128,
        16,
        0,
        True,
        None,
        32,
        False,
        id="bsnd_pa_fp16_ps16_blocksize32_val",
    ),
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        4,
        16,
        [1024, 512, 256, 768],
        8,
        128,
        16,
        3,
        True,
        None,
        64,
        False,
        id="bsnd_pa_fp16_ps16_blocksize64_b4_causal_val",
    ),
    # --- PA key 0 轴非连续(keyStride0 属性路径, 参考 compressor 方案) ---
    # 已修复(2026-08-24): torch_extension csrc 层直读 poolKey.stride(0) 作为
    # key_stride0 属性下传, tiling 属性优先消费; eager/aclnn 路径 0 轴非连续
    # 寻址正确(strided view 与 contiguous 同数据输出一致)。
    pytest.param(
        "BSND",
        "PA_BBND",
        torch.float16,
        2,
        32,
        256,
        8,
        128,
        16,
        0,
        True,
        None,
        16,
        True,
        id="bsnd_pa_fp16_ps16_key_noncontig_val",
    ),
    pytest.param(
        "TND",
        "PA_BBND",
        torch.bfloat16,
        [16, 16],
        16,
        [256, 256],
        8,
        128,
        16,
        3,
        True,
        None,
        32,
        True,
        id="tnd_pa_bf16_ps16_bs32_key_noncontig_causal_val",
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout_q,layout_k,dtype,batch,q_seq,k_seq,n1,topk,pool_size,mask_mode,"
    "return_value,tail_spec,pa_block_size,key_non_contig",
    TEST_CASES,
)
def test_pool_key_indexer(
    layout_q,
    layout_k,
    dtype,
    batch,
    q_seq,
    k_seq,
    n1,
    topk,
    pool_size,
    mask_mode,
    return_value,
    tail_spec,
    pa_block_size,
    key_non_contig,
):
    """End-to-end test: CPU golden reference vs NPU torch extension."""
    assert not key_non_contig or layout_k == "PA_BBND", (
        "0-axis non-contiguous pool_key is only supported in PA_BBND layout"
    )

    # ---- Construct inputs ----
    block_table = None
    pool_key_npu_fn = None
    if layout_q == "BSND" and layout_k == "PA_BBND":
        # PA + BSND: actual_seq_q 必须为 None; actual_seq_k 为每 batch pool 数(非前缀和)
        # k_seq 支持每 batch 不等长(产生 -1 空槽与尾块 padding)
        k_lens = k_seq if isinstance(k_seq, (list, tuple)) else [k_seq] * batch
        k_pool_lens = [kl // pool_size for kl in k_lens]
        query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
            batch, q_seq, k_seq, n1, pool_size, dtype
        )
        actual_seq_q = None
        actual_seq_k = _make_actual_seq_noncum(k_pool_lens)
        # pool_key: (B, pools, N2, D) -> (B*pools, N2, D) 再打包为物理块布局
        pool_key = pool_key.reshape(-1, pool_key.shape[-2], pool_key.shape[-1])
        pool_key, pa_block_size, pa_block_num = _pack_pa_key(
            pool_key, k_pool_lens, pool_size, pa_block_size
        )
        block_table, _ = _build_block_table(
            batch, k_pool_lens, pool_size, pa_block_size
        )
        k_seq = sum(k_lens)
    elif layout_q == "TND" and layout_k == "PA_BBND":
        # PA + TND: actual_seq_q 前缀和; actual_seq_k 为每 batch pool 数(非前缀和)
        q_lens = batch if isinstance(batch, list) else [q_seq] * batch
        k_lens = k_seq if isinstance(k_seq, list) else [k_seq] * len(q_lens)
        query, _, weights, pool_tail_k, actual_seq_q, _ = _build_tnd_inputs(
            q_lens, k_lens, n1, pool_size, dtype
        )
        k_pool_lens = [kl // pool_size for kl in k_lens]
        actual_seq_k = _make_actual_seq_noncum(k_pool_lens)
        pool_key, pa_block_size, pa_block_num = _pack_pa_key(
            torch.randn(sum(k_pool_lens), N2, HEAD_DIM, dtype=dtype),
            k_pool_lens,
            pool_size,
            pa_block_size,
        )
        q_seq = sum(q_lens)
        k_seq = sum(k_lens)
        batch = len(q_lens)
        block_table, _ = _build_block_table(
            batch, k_pool_lens, pool_size, pa_block_size
        )
    elif layout_q == "BSND":
        assert layout_k == "BSND"
        query, pool_key, weights, pool_tail_k = _build_bsnd_inputs(
            batch, q_seq, k_seq, n1, pool_size, dtype
        )
        actual_seq_q = None
        actual_seq_k = None
    elif layout_q == "TND":
        assert layout_k == "TND"
        if isinstance(batch, list):
            q_lens = batch
            k_lens = k_seq if isinstance(k_seq, list) else [k_seq] * len(q_lens)
            b_size = len(q_lens)
            total_q = sum(q_lens)
            total_k = sum(k_lens)
        else:
            q_lens = [q_seq] * batch
            k_lens = [k_seq] * batch
            b_size = batch
            total_q = q_seq * batch
            total_k = k_seq * batch
        query, pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k = (
            _build_tnd_inputs(q_lens, k_lens, n1, pool_size, dtype)
        )
        q_seq = total_q
        k_seq = total_k
        batch = b_size
    else:
        pytest.skip(f"Unsupported layout_q={layout_q}")

    # ---- PA key 0 轴非连续: 大缓冲偶数行视图(NPU 侧先搬大缓冲再切片) ----
    if key_non_contig:
        pool_key, pool_key_npu_fn = _make_pa_key_non_contig(
            pool_key, pa_block_size, pa_block_num, dtype
        )

    # ---- pool_tail_k 确定性覆写(默认随机) ----
    if tail_spec is not None:
        if isinstance(tail_spec, (list, tuple)):
            pool_tail_k = torch.tensor(list(tail_spec), dtype=torch.int64)
        else:
            pool_tail_k = torch.full((batch,), int(tail_spec), dtype=torch.int64)
        assert bool((pool_tail_k >= 0).all()) and bool(
            (pool_tail_k < pool_size).all()
        ), f"tail_spec {tail_spec} out of [0, pool_size={pool_size})"

    print(
        f"\n=== Test: layout_q={layout_q} layout_k={layout_k} dtype={dtype} batch={batch} "
        f"q_seq={q_seq} k_seq={k_seq} n1={n1} topk={topk} "
        f"pool_size={pool_size} mask_mode={mask_mode} return_value={return_value} "
        f"tail_spec={tail_spec} pa_bs={pa_block_size} key_non_contig={key_non_contig} ==="
    )

    # ---- Run CPU golden reference (independent, operator-consistent) ----
    g_indices, g_values = pool_key_indexer_reference(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )

    # ---- Run NPU ----
    npu_indices, npu_values = _run_npu(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=return_value,
        pool_key_npu_fn=pool_key_npu_fn,
    )

    # ---- 比对(集合比对策略, 无归一化后处理) ----
    # 1) indices 集合比对: 唯一精度门禁(严格 100% 标准)。
    #    逐行有效索引(>=0)排序后要求集合完全相等, 所有行必须全部通过
    #    (允许 TopK 输出顺序不同, 但选中索引集合不得有任何行差异)。
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"Indices set comparison failed for {layout_q}/{layout_k} ps={pool_size} "
        f"mask={mask_mode}"
    )

    # 2) values 参考比对: 仅打印日志, 不作为判定依据。
    #    逐行有效值(有限值)排序后按 rtol/atol 统计数值接近程度。
    if return_value:
        assert npu_values.shape == g_values.shape, (
            f"values shape mismatch: golden {g_values.shape} vs npu {npu_values.shape}"
        )
        compare_values(g_values, npu_values, dtype, label="values")
    else:
        # return_value=False: NPU values 应为空 tensor
        assert npu_values.numel() == 0 or npu_values.shape[-1] == 0, (
            f"Expected empty values when return_value=False, got shape {npu_values.shape}"
        )

    print(
        f"=== PASSED: {layout_q} dtype={dtype} ps={pool_size} mask={mask_mode} val={return_value} ==="
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
