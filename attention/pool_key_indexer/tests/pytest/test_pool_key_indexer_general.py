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
PoolKeyIndexer 泛化验证测试 (generalization tests).

目的:
  在 test_pool_key_indexer.py(定点用例)基础上, 用"随机生成 + 定向边界"两层用例对算子
  已实现功能做泛化验证, 尽可能覆盖所有场景与参数组合, 并以覆盖率断言保证泛化广度。

用例生成(确定性, MASTER_SEED 固定, 可复现):
  1. 随机层(40 例): 4 个场景桶(BSND/BSND, TND/TND, BSND/PA, TND/PA)各 10 例,
     随机采样 dtype / mask_mode / pool_size / topk / n1 / batch / 每batch长度 /
     pool_tail_k / PA block_size(轮转覆盖 16~1024) / 0长度query batch /
     0池batch / 池数<sparseCount 等;
  2. 定向层(20 例): 最小/最大 topk、sparseCount=1、pool_size 极值(1/128 与
     非2次幂8倍数)、特殊 topk(3072/4096/8192)、S1=1、单池、池数截断、
     0池batch(含 ps=128 回归)、TND零长度batch、PA key 0轴非连续、
     block_size 1024 等。

比对策略(与 test_pool_key_indexer.py 一致):
  - indices: 逐行有效索引集合比对, 行通过率 >= 0.995, 唯一精度门禁;
  - values:  逐行有效值排序数值比对, 仅打印参考日志;
  - 另增加输出 dtype/shape 与接口规格的显式断言(int32 / float32 / 空values)。

覆盖率断言(test_zz_generalization_coverage_summary):
  全部用例执行完后校验各参数轴的覆盖桶均被命中(布局x4 / dtypex2 / maskx2 /
  return_valuex2 / pool_size 分桶 / topk 分桶 / n1{1,mid,64} / batch{1,2-3,4+} /
  S1=1 / 池数<sparseCount / 0长度query batch / 非连续key /
  block_size 全档 / tail 边界), 未命中即失败, 防止泛化退化为定点复读。

未纳入泛化的功能与已知缺陷(详见文末打印的分析报告, 供排定修复优先级):
  A. quant_mode=0/1(FP8/mxFP8): 仅 Ascend 950(DAV_3510)支持, 本机 910B3 无法验证;
  B. BSND + actual_seq_q/actual_seq_k: 接口文档称可选, tiling 实际拒绝(EZ1008);
  C. pool_size ∈ [2,7] 及非 8 的倍数: aicore 崩溃(EE9999/507015), 任意 shape 复现;
  D. topk ∈ {5120, 6144, 7168}: shape 相关崩溃(如 ps=16/s2pool∈{384,512} 崩,
     s2pool=640 正常), 崩溃面不稳定;
  E. pool_size>=32 且 topk>=6144: 已知缺陷(test_pool_key_indexer.py 已记录);
  F. 0 池 batch(actual_seq_k 含 0): 非确定性数据污染 —— 0 池 batch 的首个输出
     行偶发残留脏索引(脏分配器下 17/30 复现, TND/PA 与 TND/TND 均命中),
     规格声称"有效 pool 数不小于 0"允许 0 池; 实测 0 长度 query batch 安全。
"""

import os
import random
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
MASTER_SEED = 20260824

# ---------------------------------------------------------------------------
# 已实测验证可用的参数包络 (arch22 / Atlas A2 910B3, 2026-08-24)
# ---------------------------------------------------------------------------
# pool_size: 规格允许 [1,128] 任意整数, 但实测仅 {1} ∪ {8 的倍数} 可用;
#            其余值(2/3/4/5/7/12/100 等)任意 shape 均触发 aicore 崩溃。
POOL_SIZES_VALID = [
    1,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
]
POOL_SIZE_WEIGHTS = [4, 3, 4, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2]
# topk 特殊值: 3072/4096 已多 shape 验证稳定; 8192 仅纳入已验证形态(定向用例);
#             5120/6144/7168 存在 shape 相关崩溃, 不纳入(见模块 docstring D/E)。
TOPK_SPECIAL_VALID = [3072, 4096]
# PA block_size: 16 的倍数且 <= 1024, 全档轮转覆盖(含非 2 次幂 48)。
BLOCK_SIZES_VALID = [16, 32, 48, 64, 128, 256, 1024]

N1_CHOICES = [1, 2, 3, 4, 5, 8, 8, 12, 16, 16, 32, 32, 64, 64]
S1_CHOICES = [1, 2, 4, 8, 16, 16, 32, 32, 64, 128, 256]

# ---------------------------------------------------------------------------
# 覆盖率跟踪(模块级, 全部用例共享)
# ---------------------------------------------------------------------------
_COVERAGE = {
    "layout": set(),  # (layout_q, layout_k)
    "dtype_layout": set(),  # (dtype, layout_q, layout_k)
    "mask_mode": set(),
    "return_value": set(),
    "pool_size": set(),
    "pool_size_bucket": set(),  # ps1 / small(8-32) / mid(40-64) / large(72-128)
    "topk_bucket": set(),  # t1 / small / mid / near2048 / 3072 / 4096 / 8192
    "n1": set(),
    "batch_bucket": set(),  # b1 / b2_3 / b4plus
    "pa_block_size": set(),
    "tail_bound": set(),  # zero / max / mixed / random
    "s1_eq_1": False,
    "pools_lt_sparse": False,
    "zero_pool_batch": False,
    "zero_q_batch": False,
    "key_non_contig": False,
    "cases_executed": 0,
}


def _topk_bucket(topk):
    if topk == 1:
        return "t1"
    if topk <= 128:
        return "small"
    if topk <= 1024:
        return "mid"
    if topk <= 2048:
        return "near2048"
    return str(topk)


def _ps_bucket(ps):
    if ps == 1:
        return "ps1"
    if ps <= 32:
        return "small(8-32)"
    if ps <= 64:
        return "mid(40-64)"
    return "large(72-128)"


def _update_coverage(cfg, k_pools, q_lens):
    _COVERAGE["cases_executed"] += 1
    _COVERAGE["layout"].add((cfg["layout_q"], cfg["layout_k"]))
    _COVERAGE["dtype_layout"].add(
        (str(cfg["dtype"]).replace("torch.", ""), cfg["layout_q"], cfg["layout_k"])
    )
    _COVERAGE["mask_mode"].add(cfg["mask_mode"])
    _COVERAGE["return_value"].add(cfg["return_value"])
    _COVERAGE["pool_size"].add(cfg["pool_size"])
    _COVERAGE["pool_size_bucket"].add(_ps_bucket(cfg["pool_size"]))
    _COVERAGE["topk_bucket"].add(_topk_bucket(cfg["topk"]))
    _COVERAGE["n1"].add(cfg["n1"])
    batch = len(q_lens)
    if batch == 1:
        _COVERAGE["batch_bucket"].add("b1")
    elif batch <= 3:
        _COVERAGE["batch_bucket"].add("b2_3")
    else:
        _COVERAGE["batch_bucket"].add("b4plus")
    if cfg["layout_k"] == "PA_BBND":
        _COVERAGE["pa_block_size"].add(cfg["pa_block_size"])
    if any(q == 1 for q in q_lens):
        _COVERAGE["s1_eq_1"] = True
    sparse = cfg["topk"] // cfg["pool_size"]
    if any(0 < p < sparse for p in k_pools):
        _COVERAGE["pools_lt_sparse"] = True
    if any(p == 0 for p in k_pools):
        _COVERAGE["zero_pool_batch"] = True
    if any(q == 0 for q in q_lens):
        _COVERAGE["zero_q_batch"] = True
    if cfg["key_non_contig"]:
        _COVERAGE["key_non_contig"] = True
    tail = cfg.get("tail")
    if tail is None:
        _COVERAGE["tail_bound"].add("random")
    elif isinstance(tail, (list, tuple)):
        _COVERAGE["tail_bound"].add("mixed" if len(set(tail)) > 1 else "fixed")
        if all(t == 0 for t in tail):
            _COVERAGE["tail_bound"].add("zero")
        if all(t == cfg["pool_size"] - 1 for t in tail):
            _COVERAGE["tail_bound"].add("max")


# ---------------------------------------------------------------------------
# 输入构造
# ---------------------------------------------------------------------------


def _pack_pa_key(pool_key_bsnd, k_pool_lens, block_size):
    """将逻辑 (sum_pools, N2, D) 打包为 PA 物理块布局 (blockNum, blockSize, N2, D)。

    每 batch 的池独立连续填块, 物理块随机排布, 尾块不足补零;
    0 池 batch 不占任何物理块。返回 (key_pa, block_size, block_num)。
    """
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    n2, d = pool_key_bsnd.shape[1], pool_key_bsnd.shape[2]
    blocks = torch.zeros(block_num, block_size, n2, d, dtype=pool_key_bsnd.dtype)
    pool_start = blk_start = 0
    for pools, nb in zip(k_pool_lens, blocks_per_batch):
        if nb == 0:
            continue
        seg = pool_key_bsnd[pool_start : pool_start + pools]
        pad = nb * block_size - pools
        if pad:
            seg = torch.cat([seg, torch.zeros(pad, n2, d, dtype=seg.dtype)], dim=0)
        blocks[blk_start : blk_start + nb] = seg.view(nb, block_size, n2, d)
        pool_start += pools
        blk_start += nb
    perm = torch.randperm(block_num)
    return blocks[perm].contiguous(), block_size, block_num


def _build_block_table(batch, k_pool_lens, block_size):
    """构造 (B, maxBlocks) int32 物理块映射, 空槽 -1。"""
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

    物理块嵌入 2 倍缓冲的偶数行, 视图 stride[0] = 2 * block_size * N2 * D。
    返回 (cpu_view, npu_builder): NPU 侧先搬大缓冲再切片, 绕过 .to(npu)
    的连续化。"""
    big = torch.zeros(2 * block_num, block_size, N2, HEAD_DIM, dtype=dtype)
    big[0::2] = key_pa
    view = big[::2]

    def npu_builder(dev, _big=big):
        return _big.to(dev)[::2]

    return view, npu_builder


def _build_case_inputs(cfg):
    """根据 case 配置构造全部 CPU 输入 + 可选 NPU key 构造器。"""
    torch.manual_seed(cfg["seed"])
    dtype = cfg["dtype"]
    ps = cfg["pool_size"]
    q_lens = cfg["q_lens"]
    k_pools = cfg["k_pools"]
    batch = len(q_lens)
    n1 = cfg["n1"]

    actual_seq_q = None
    actual_seq_k = None
    block_table = None
    npu_key_fn = None

    # ---- query / weights ----
    if cfg["layout_q"] == "BSND":
        s1 = q_lens[0]
        assert all(q == s1 for q in q_lens), "BSND 要求各 batch S1 相同"
        query = torch.randn(batch, s1, n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(batch, s1, n1, dtype=dtype)
    else:  # TND
        t1 = sum(q_lens)
        assert t1 > 0, "query 不允许空 tensor"
        query = torch.randn(t1, n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(t1, n1, dtype=dtype)
        actual_seq_q = torch.tensor(q_lens, dtype=torch.int64).cumsum(0)

    # ---- pool_key ----
    if cfg["layout_k"] == "PA_BBND":
        assert sum(k_pools) > 0, "PA 全部 batch 0 池时 pool_key 为空, 不允许"
        logical = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        pool_key, block_size, block_num = _pack_pa_key(
            logical, k_pools, cfg["pa_block_size"]
        )
        block_table, _ = _build_block_table(batch, k_pools, cfg["pa_block_size"])
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64)  # 非前缀和
        if cfg["key_non_contig"]:
            pool_key, npu_key_fn = _make_pa_key_non_contig(
                pool_key, block_size, block_num, dtype
            )
    elif cfg["layout_q"] == "TND":  # TND/TND
        assert sum(k_pools) > 0, "pool_key 不允许空 tensor"
        pool_key = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64).cumsum(0)  # 前缀和
    else:  # BSND/BSND: 不允许 S2 pad, 各 batch 池数必须相同
        pools = k_pools[0]
        assert all(p == pools for p in k_pools), "BSND 要求各 batch 池数相同"
        assert pools > 0
        pool_key = torch.randn(batch, pools, N2, HEAD_DIM, dtype=dtype)

    # ---- pool_tail_k ----
    tail = cfg.get("tail")
    if tail is None:
        pool_tail_k = torch.randint(0, ps, (batch,), dtype=torch.int64)
    else:
        pool_tail_k = torch.tensor(list(tail), dtype=torch.int64)
        assert bool((pool_tail_k >= 0).all()) and bool((pool_tail_k < ps).all())

    return {
        "query": query,
        "pool_key": pool_key,
        "weights": weights,
        "pool_tail_k": pool_tail_k,
        "actual_seq_q": actual_seq_q,
        "actual_seq_k": actual_seq_k,
        "block_table": block_table,
        "npu_key_fn": npu_key_fn,
    }


# ---------------------------------------------------------------------------
# 用例配置生成
# ---------------------------------------------------------------------------

BUCKETS = {
    "dense_bsnd": ("BSND", "BSND"),
    "dense_tnd": ("TND", "TND"),
    "pa_bsnd": ("BSND", "PA_BBND"),
    "pa_tnd": ("TND", "PA_BBND"),
}

N_RANDOM_PER_BUCKET = 10


def _sample_topk(rng, ps):
    """采样 topk: 18% 概率尝试特殊值(需被 ps 整除), 其余 sparseCount*ps <= 2048。"""
    if rng.random() < 0.18:
        special = rng.choice(TOPK_SPECIAL_VALID)
        if special % ps == 0:
            return special
    sparse = rng.randint(1, 24) if rng.random() < 0.75 else rng.randint(25, 96)
    topk = sparse * ps
    while topk > 2048:
        sparse = max(1, sparse // 2)
        topk = sparse * ps
    return topk


def _sample_tail(rng, batch, ps):
    """25% 概率确定性边界(全 0 / 全 ps-1 / 每 batch 混合), 否则随机(None)。"""
    r = rng.random()
    if r < 0.08:
        return [0] * batch
    if r < 0.17:
        return [ps - 1] * batch
    if r < 0.25:
        return [rng.randrange(ps) for _ in range(batch)]
    return None


def _gen_random_case(case_idx, bucket, seed, pa_bs_cycle):
    rng = random.Random(seed)
    layout_q, layout_k = BUCKETS[bucket]
    ps = rng.choices(POOL_SIZES_VALID, weights=POOL_SIZE_WEIGHTS, k=1)[0]
    topk = _sample_topk(rng, ps)
    sparse = topk // ps
    n1 = rng.choice(N1_CHOICES)
    batch = rng.randint(1, 6)

    if layout_q == "BSND":
        s1 = rng.choice(S1_CHOICES)
        q_lens = [s1] * batch
    else:
        q_lens = [rng.randint(1, 64) for _ in range(batch)]
        if rng.random() < 0.10:  # TND 零长度 query batch(前缀和允许相等)
            q_lens[rng.randrange(batch)] = 0
        if sum(q_lens) == 0:
            q_lens[0] = 1

    if layout_k == "BSND":
        pools = rng.randint(1, min(256, max(2, sparse * 2)))
        k_pools = [pools] * batch
    else:
        # 10% 概率 0 池 batch(TND/PA 允许, 规格有效池数>=0);
        # 曾因 CleanInvalidOutput 缺 V->MTE3 跨流水同步偶发残留脏索引,
        # 已修复(2026-08-25)后纳入泛化。
        k_pools = []
        for _ in range(batch):
            if rng.random() < 0.10:
                k_pools.append(0)
            else:
                k_pools.append(
                    rng.randint(1, min(160, max(2, sparse + rng.randint(1, 64))))
                )
        if sum(k_pools) == 0:
            k_pools[0] = rng.randint(1, 16)

    return {
        "id": f"gen_{bucket}_{case_idx:02d}",
        "layout_q": layout_q,
        "layout_k": layout_k,
        "dtype": rng.choice([torch.float16, torch.bfloat16]),
        "q_lens": q_lens,
        "k_pools": k_pools,
        "n1": n1,
        "topk": topk,
        "pool_size": ps,
        "mask_mode": rng.choice([0, 3]),
        "return_value": rng.choice([True, False]),
        "pa_block_size": BLOCK_SIZES_VALID[pa_bs_cycle % len(BLOCK_SIZES_VALID)],
        "key_non_contig": False,
        "tail": _sample_tail(rng, batch, ps),
        "seed": seed,
    }


FP16 = torch.float16
BF16 = torch.bfloat16

DIRECTED_CASES = [
    # ---- topk 最小值 / 常规上限 ----
    dict(
        id="dir_bsnd_ps1_topk1",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[8],
        k_pools=[8],
        n1=4,
        topk=1,
        pool_size=1,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_bsnd_ps1_topk2048",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[8],
        k_pools=[2048],
        n1=4,
        topk=2048,
        pool_size=1,
        mask_mode=0,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    # ---- sparseCount=1 家族(topk == pool_size) ----
    dict(
        id="dir_bsnd_ps8_topk8_sparse1",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[32],
        k_pools=[16],
        n1=4,
        topk=8,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_bsnd_ps128_topk128_tail127",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[256, 256],
        k_pools=[16, 16],
        n1=8,
        topk=128,
        pool_size=128,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=[127, 127],
    ),
    # ---- 特殊 topk(已验证稳定形态) ----
    dict(
        id="dir_bsnd_ps24_topk3072",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[16],
        k_pools=[256],
        n1=4,
        topk=3072,
        pool_size=24,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_tnd_bf16_ps8_topk4096",
        layout_q="TND",
        layout_k="TND",
        dtype=BF16,
        q_lens=[8, 8],
        k_pools=[512, 512],
        n1=4,
        topk=4096,
        pool_size=8,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_tnd_ps16_topk8192",
        layout_q="TND",
        layout_k="TND",
        dtype=FP16,
        q_lens=[8, 8],
        k_pools=[1024, 1024],
        n1=4,
        topk=8192,
        pool_size=16,
        mask_mode=3,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_bsnd_ps16_topk8192",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[16, 16],
        k_pools=[1024, 1024],
        n1=4,
        topk=8192,
        pool_size=16,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_bsnd_ps64_topk2048",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[32],
        k_pools=[64],
        n1=8,
        topk=2048,
        pool_size=64,
        mask_mode=3,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    # ---- 0 池 batch / 0 长度 query batch(0 池残留脏索引缺陷已修复, 2026-08-25) ----
    dict(
        id="dir_pa_bsnd_zeropool",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[16, 16, 16],
        k_pools=[16, 0, 33],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_tnd_zeroq",
        layout_q="TND",
        layout_k="TND",
        dtype=FP16,
        q_lens=[16, 0, 8],
        k_pools=[16, 16, 16],
        n1=4,
        topk=128,
        pool_size=16,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_tnd_zerok",
        layout_q="TND",
        layout_k="TND",
        dtype=FP16,
        q_lens=[16, 8, 8],
        k_pools=[16, 0, 16],
        n1=4,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_tnd_zeropool_ps128",
        layout_q="TND",
        layout_k="TND",
        dtype=FP16,
        q_lens=[63, 20, 51, 14, 22],
        k_pools=[32, 13, 24, 4, 0],
        n1=8,
        topk=1024,
        pool_size=128,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    # ---- PA key 0 轴非连续 ----
    dict(
        id="dir_pa_tnd_bf16_noncontig_bs32",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=BF16,
        q_lens=[16, 16],
        k_pools=[16, 16],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=32,
        key_non_contig=True,
        tail=None,
    ),
    # ---- 极小 shape / 单池 / 池数 < sparseCount ----
    dict(
        id="dir_bsnd_s1_1",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[1],
        k_pools=[32],
        n1=4,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_bsnd_pools_lt_sparse_n1_1",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[16],
        k_pools=[2],
        n1=1,
        topk=64,
        pool_size=8,
        mask_mode=0,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_bsnd_pools1",
        layout_q="BSND",
        layout_k="BSND",
        dtype=FP16,
        q_lens=[16],
        k_pools=[1],
        n1=4,
        topk=128,
        pool_size=16,
        mask_mode=3,
        return_value=False,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
    # ---- n1=64 / block_size 极值 / 非 2 次幂大 pool_size ----
    dict(
        id="dir_pa_tnd_n1_64",
        layout_q="TND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[16, 16],
        k_pools=[24, 40],
        n1=64,
        topk=128,
        pool_size=16,
        mask_mode=0,
        return_value=True,
        pa_block_size=64,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_pa_bs1024",
        layout_q="BSND",
        layout_k="PA_BBND",
        dtype=FP16,
        q_lens=[8, 8],
        k_pools=[70, 20],
        n1=8,
        topk=128,
        pool_size=16,
        mask_mode=0,
        return_value=False,
        pa_block_size=1024,
        key_non_contig=False,
        tail=None,
    ),
    dict(
        id="dir_tnd_bf16_ps96",
        layout_q="TND",
        layout_k="TND",
        dtype=BF16,
        q_lens=[8, 8],
        k_pools=[12, 24],
        n1=4,
        topk=192,
        pool_size=96,
        mask_mode=3,
        return_value=True,
        pa_block_size=16,
        key_non_contig=False,
        tail=None,
    ),
]


def _generate_all_cases():
    """确定性生成全部泛化用例(随机层 + 定向层)。"""
    cases = []
    rng = random.Random(MASTER_SEED)
    pa_bs_cycle = 0
    case_no = 0
    for bucket in BUCKETS:
        for i in range(N_RANDOM_PER_BUCKET):
            seed = rng.randrange(1 << 31)
            cfg = _gen_random_case(i, bucket, seed, pa_bs_cycle)
            if cfg["layout_k"] == "PA_BBND":
                pa_bs_cycle += 1
            # 每桶第 5 例启用 PA key 非连续(仅 PA 桶生效)
            if bucket.startswith("pa_") and i == 4:
                cfg["key_non_contig"] = True
            cases.append(cfg)
            case_no += 1
    for j, cfg in enumerate(DIRECTED_CASES):
        cfg = dict(cfg)
        cfg["seed"] = MASTER_SEED + 10000 + j
        cases.append(cfg)
    return cases


ALL_CASES = _generate_all_cases()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _npu_init():
    torch_npu.npu.set_device(0)


@pytest.mark.parametrize("cfg", ALL_CASES, ids=[c["id"] for c in ALL_CASES])
def test_pool_key_indexer_general(cfg):
    """泛化用例: CPU golden 标杆 vs NPU torch extension。"""
    inputs = _build_case_inputs(cfg)
    ps = cfg["pool_size"]
    topk = cfg["topk"]
    sparse = topk // ps
    out_len = topk + ps - 1

    print(
        f"\n=== General: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"dtype={cfg['dtype']} batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} "
        f"k_pools={cfg['k_pools']} n1={cfg['n1']} topk={topk} ps={ps} "
        f"mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"pa_bs={cfg['pa_block_size']} noncontig={cfg['key_non_contig']} "
        f"tail={cfg.get('tail')} ==="
    )

    # ---- CPU golden ----
    g_indices, g_values = pool_key_indexer_reference(
        inputs["query"],
        inputs["pool_key"],
        inputs["weights"],
        inputs["pool_tail_k"],
        actual_seq_q=inputs["actual_seq_q"],
        actual_seq_k=inputs["actual_seq_k"],
        block_table=inputs["block_table"],
        layout_q=cfg["layout_q"],
        layout_k=cfg["layout_k"],
        topk=topk,
        pool_size=ps,
        mask_mode=cfg["mask_mode"],
        return_value=cfg["return_value"],
    )

    # ---- NPU ----
    dev = "npu:0"
    key_npu = (
        inputs["npu_key_fn"](dev)
        if inputs["npu_key_fn"] is not None
        else inputs["pool_key"].to(dev)
    )
    npu_indices, npu_values = pool_key_indexer(
        inputs["query"].to(dev),
        key_npu,
        inputs["weights"].to(dev),
        inputs["pool_tail_k"].to(dev),
        actual_seq_q=(
            inputs["actual_seq_q"].to(dev)
            if inputs["actual_seq_q"] is not None
            else None
        ),
        actual_seq_k=(
            inputs["actual_seq_k"].to(dev)
            if inputs["actual_seq_k"] is not None
            else None
        ),
        block_table=(
            inputs["block_table"].to(dev) if inputs["block_table"] is not None else None
        ),
        layout_q=cfg["layout_q"],
        layout_k=cfg["layout_k"],
        topk=topk,
        pool_size=ps,
        mask_mode=cfg["mask_mode"],
        quant_mode=-1,
        return_value=cfg["return_value"],
    )
    torch.npu.synchronize()
    npu_indices = npu_indices.cpu()
    npu_values = npu_values.cpu()

    # ---- 输出规格断言(dtype/shape, 接口文档口径) ----
    assert npu_indices.dtype == torch.int32, (
        f"indices dtype {npu_indices.dtype} != int32"
    )
    if cfg["layout_q"] == "BSND":
        exp_idx = (len(cfg["q_lens"]), cfg["q_lens"][0], out_len)
        exp_val = (len(cfg["q_lens"]), cfg["q_lens"][0], sparse)
    else:
        exp_idx = (sum(cfg["q_lens"]), out_len)
        exp_val = (sum(cfg["q_lens"]), sparse)
    assert tuple(npu_indices.shape) == exp_idx, (
        f"indices shape {tuple(npu_indices.shape)} != {exp_idx}"
    )
    if cfg["return_value"]:
        assert npu_values.dtype == torch.float32, (
            f"values dtype {npu_values.dtype} != float32"
        )
        assert tuple(npu_values.shape) == exp_val, (
            f"values shape {tuple(npu_values.shape)} != {exp_val}"
        )
    else:
        assert npu_values.numel() == 0, (
            f"return_value=False 时 values 应为空, got {tuple(npu_values.shape)}"
        )

    # ---- 比对(集合比对策略) ----
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{cfg['id']}] indices set comparison failed: "
        f"layout=({cfg['layout_q']},{cfg['layout_k']}) ps={ps} topk={topk} "
        f"mask={cfg['mask_mode']}"
    )
    if cfg["return_value"]:
        compare_values(g_values, npu_values, cfg["dtype"], label="values")

    _update_coverage(cfg, cfg["k_pools"], cfg["q_lens"])
    print(f"=== GENERAL PASSED: {cfg['id']} ===")


def test_zz_generalization_coverage_summary():
    """覆盖率收口: 校验泛化广度并打印覆盖矩阵与未覆盖功能分析。"""
    cov = _COVERAGE
    print("\n" + "=" * 72)
    print("PoolKeyIndexer 泛化测试覆盖率汇总")
    print("=" * 72)
    print(f"executed cases : {cov['cases_executed']} / {len(ALL_CASES)}")
    print(f"layout combos  : {sorted(cov['layout'])}")
    print(f"dtype x layout : {sorted(cov['dtype_layout'])}")
    print(f"mask_mode      : {sorted(cov['mask_mode'])}")
    print(f"return_value   : {sorted(cov['return_value'])}")
    print(f"pool_size      : {sorted(cov['pool_size'])}")
    print(f"pool_size bkt  : {sorted(cov['pool_size_bucket'])}")
    print(f"topk buckets   : {sorted(cov['topk_bucket'])}")
    print(f"n1             : {sorted(cov['n1'])}")
    print(f"batch buckets  : {sorted(cov['batch_bucket'])}")
    print(f"pa_block_size  : {sorted(cov['pa_block_size'])}")
    print(f"tail bounds    : {sorted(cov['tail_bound'])}")
    print(
        f"s1==1={cov['s1_eq_1']} pools<sparse={cov['pools_lt_sparse']} "
        f"zero_pool_batch={cov['zero_pool_batch']} zero_q_batch={cov['zero_q_batch']} "
        f"key_non_contig={cov['key_non_contig']}"
    )

    # ---- 覆盖率断言(防泛化退化) ----
    problems = []
    if cov["cases_executed"] != len(ALL_CASES):
        problems.append(f"仅执行 {cov['cases_executed']}/{len(ALL_CASES)} 例")
    expect_layouts = {
        ("BSND", "BSND"),
        ("TND", "TND"),
        ("BSND", "PA_BBND"),
        ("TND", "PA_BBND"),
    }
    if cov["layout"] != expect_layouts:
        problems.append(f"layout 覆盖不全: {cov['layout']}")
    for layout in expect_layouts:
        for dt in ("float16", "bfloat16"):
            if (dt, *layout) not in cov["dtype_layout"]:
                problems.append(f"{layout} 未覆盖 dtype {dt}")
    if cov["mask_mode"] != {0, 3}:
        problems.append(f"mask_mode 覆盖不全: {cov['mask_mode']}")
    if cov["return_value"] != {True, False}:
        problems.append(f"return_value 覆盖不全: {cov['return_value']}")
    expect_ps_buckets = {"ps1", "small(8-32)", "mid(40-64)", "large(72-128)"}
    if not expect_ps_buckets <= cov["pool_size_bucket"]:
        problems.append(f"pool_size 分桶覆盖不全: {cov['pool_size_bucket']}")
    if len(cov["pool_size"]) < 8:
        problems.append(f"pool_size 去重覆盖仅 {len(cov['pool_size'])} 个 (<8)")
    expect_topk_buckets = {"t1", "small", "mid", "near2048", "3072", "4096", "8192"}
    if not expect_topk_buckets <= cov["topk_bucket"]:
        problems.append(f"topk 分桶覆盖不全: {cov['topk_bucket']}")
    if not {1, 64} <= cov["n1"]:
        problems.append(f"n1 未覆盖极值 1/64: {sorted(cov['n1'])}")
    if not {1, 2, 3} & cov["n1"] or not {5, 8, 12, 16, 32} & cov["n1"]:
        problems.append(f"n1 中间值覆盖不足: {sorted(cov['n1'])}")
    if cov["batch_bucket"] != {"b1", "b2_3", "b4plus"}:
        problems.append(f"batch 分桶覆盖不全: {cov['batch_bucket']}")
    if set(BLOCK_SIZES_VALID) - cov["pa_block_size"]:
        problems.append(f"pa_block_size 覆盖不全: {sorted(cov['pa_block_size'])}")
    if not {"zero", "max", "random"} <= cov["tail_bound"]:
        problems.append(f"tail 边界覆盖不全: {cov['tail_bound']}")
    for key, desc in [
        ("s1_eq_1", "S1=1 最小 shape"),
        ("pools_lt_sparse", "池数 < sparseCount 截断"),
        ("zero_pool_batch", "0 池 batch"),
        ("zero_q_batch", "TND 0 长度 query batch"),
        ("key_non_contig", "PA key 0 轴非连续"),
    ]:
        if not cov[key]:
            problems.append(f"未覆盖 {desc}")

    # ---- 未覆盖功能分析报告(供排定优先级, 不影响本用例判定) ----
    print("\n---- 未纳入泛化的功能/缺陷分析(供识别优先级) ----")
    print(
        "A. [平台限制] quant_mode=0(FP8 per-token-head)/1(mxFP8): 仅 Ascend 950\n"
        "   (DAV_3510) 支持, 本机 910B3 tiling 直接拒绝 FP8 输入; golden 亦无\n"
        "   反量化路径, 需 950 环境补充泛化。"
    )
    print(
        "B. [文档-实现不一致] BSND + actual_seq_q/actual_seq_k: 接口文档标注\n"
        "   BSND 可选传入有效长度, tiling 实际强制为 null(EZ1008 报错),\n"
        "   实测确认被拒; 需决策: 补实现 or 修文档。"
    )
    print(
        "C. [kernel 缺陷] pool_size ∈ [2,7] 及非 8 的倍数(如 12/100): 任意 shape\n"
        "   触发 aicore exception(EE9999/507015)且污染进程; 规格声称 [1,128]\n"
        "   任意整数, 实测仅 {1} ∪ {8 的倍数} 可用。"
    )
    print(
        "D. [kernel 缺陷] topk ∈ {5120, 6144, 7168}: shape 相关崩溃\n"
        "   (ps=16/topk=5120 在 s2pool=384/512 崩、640 正常; ps=32/topk>=6144 崩;\n"
        "   ps=1/8 全部正常), 崩溃面不稳定, 已整体排除出泛化用例。"
    )
    print(
        "E. [已知缺陷] pool_size>=32 且 topk>=6144 崩溃(test_pool_key_indexer.py\n"
        "   已记录, 与 D 部分重叠)。"
    )
    print(
        "F. [已修复 2026-08-25] 0 池 batch(actual_seq_k 含 0)曾非确定性残留脏\n"
        "   索引: 根因为 CleanInvalidOutput(poolSize>1 分支)在 Duplicate 写\n"
        "   expandOutLocal_ 后缺 V->MTE3 跨流水同步(仅 PipeBarrier<PIPE_V>),\n"
        "   MTE3 DataCopyPad 竞争读到上次 ExpandAndAppendIndices 残留的有效\n"
        "   token 索引; 补 SetFlag/WaitFlag<HardEvent::V_MTE3> 后修复(脏分配器\n"
        "   压力 17/30 复现 -> 0/30), 0 池用例已回补进泛化套件。"
    )
    print(
        "G. [规格内边界] topk=8192 仅以已验证形态(TND/BSND, ps=16, 大 k)纳入;\n"
        "   未做全 shape 扫描, 其余形态待 D/E 修复后补测。"
    )
    print(
        "H. [arch35 移植待验证] arch22 的 4 处 poolSize>1 行宽修复(批次基址/\n"
        "   清理偏移/values 行宽换算/ProcessInvalid 越界)已按同模式移植到\n"
        "   arch35(950), 但本机无 950 硬件且 arch35 VF 路径存在既有 fp8 编译\n"
        "   阻塞(见 docs/build_fix_record.md), 编译与运行时验证待 950 环境完成。"
    )

    assert not problems, "泛化覆盖率不足: " + "; ".join(problems)
    print("\n[SUMMARY] 泛化覆盖率检查通过。")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
