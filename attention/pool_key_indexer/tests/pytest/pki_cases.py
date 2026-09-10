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
PoolKeyIndexer 用例配置(精度套件 / GE 图套件共用)。

用例来源与分类:
  1. 定点用例 FIXED_CASES(42 例): 原 test_pool_key_indexer.py 的 TEST_CASES,
     覆盖 BSND/TND/PA 布局、fp16/bf16、mask 0/3、pool_size 1~128、
     topk 16~8192、tail 边界、n1 极值、PA 非连续等定点形态;
  2. 泛化用例 ALL_GENERAL_CASES(60 例): 原 test_pool_key_indexer_general.py,
     确定性 MASTER_SEED 生成(随机层 40 例 + 定向层 20 例), 含覆盖率统计口径;
  3. 已知缺陷回归 ALL_DEFECT_CASES(25 例): known_defect_cases.py 注册表
     (缺陷家族 C/D/E, 由 _defect_case_runner.py 子进程隔离执行);
  4. 因果边界计数 BOUNDARY_COUNT_CASES: 原 boundary_causal_test.py 的
     每行有效计数解析公式校验场景(pool_size=16/topk=16/mask=3, tail 全扫)。

统一 cfg dict 字段见 pki_common.build_case_inputs。
"""

import random

import torch

from known_defect_cases import ALL_DEFECT_CASES  # noqa: F401  (缺陷回归用例)

FP16 = torch.float16
BF16 = torch.bfloat16

# ---------------------------------------------------------------------------
# 1. 定点用例(原 test_pool_key_indexer.py TEST_CASES)
# ---------------------------------------------------------------------------
# 元组: (id, layout_q, layout_k, dtype, batch, q_seq, k_seq, n1, topk,
#        pool_size, mask_mode, return_value, tail_spec, pa_block_size,
#        key_non_contig)
#   batch:    int(BSND/TND 等长) 或 list(TND 每 batch q_lens)
#   q_seq:    每 batch query token 数(batch 为 list 时忽略)
#   k_seq:    池化前 key token 数(int 等长 / list 每 batch), 池数 = k_seq // pool_size
#   tail_spec: None=随机 / int=所有 batch 定值 / list=每 batch 定值
_FIXED_TUPLES = [
    # --- 退化 pool_size=1 ---
    (
        "bsnd_fp16_ps1_nocausal_noval",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps1_causal_noval",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_bf16_ps1_nocausal_val",
        "BSND",
        "BSND",
        BF16,
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
    ),
    (
        "bsnd_fp16_ps1_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    # --- 池化展开 pool_size>1 ---
    (
        "bsnd_fp16_ps16_nocausal_noval",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps16_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_bf16_ps8_nocausal_val",
        "BSND",
        "BSND",
        BF16,
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
    ),
    (
        "bsnd_fp16_ps32_causal_noval",
        "BSND",
        "BSND",
        FP16,
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
    ),
    # --- TND ---
    (
        "tnd_fp16_ps1_nocausal_noval",
        "TND",
        "TND",
        FP16,
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
    ),
    (
        "tnd_fp16_ps16_causal_val",
        "TND",
        "TND",
        FP16,
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
    ),
    (
        "tnd_bf16_ps8_nocausal_val",
        "TND",
        "TND",
        BF16,
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
    ),
    (
        "tnd_fp16_ps16_causal_noval_3batch",
        "TND",
        "TND",
        FP16,
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
    ),
    # --- PA (layout_k=PA_BBND) ---
    (
        "bsnd_pa_fp16_ps8_nocausal_val",
        "BSND",
        "PA_BBND",
        FP16,
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
    ),
    (
        "bsnd_pa_fp16_ps16_causal_val",
        "BSND",
        "PA_BBND",
        FP16,
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
    ),
    (
        "tnd_pa_fp16_ps8_nocausal_noval",
        "TND",
        "PA_BBND",
        FP16,
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
    ),
    (
        "tnd_pa_bf16_ps16_causal_val",
        "TND",
        "PA_BBND",
        BF16,
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
    ),
    # --- sparseCount=1 边界(topk == pool_size) ---
    (
        "bsnd_fp16_ps16_topk16_causal_sparse1",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps16_topk16_causal_L48",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "tnd_fp16_ps16_topk16_causal_L48",
        "TND",
        "TND",
        FP16,
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
    ),
    # --- 大 topk(建议值 512/2048 与特殊值 3072/4096/8192) ---
    (
        "bsnd_fp16_ps16_topk512_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps16_topk2048_nocausal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps1_topk3072_causal_sparse3k",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps16_topk4096_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "tnd_fp16_ps16_topk8192_causal_noval",
        "TND",
        "TND",
        FP16,
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
    ),
    # --- pool_size 极值(64 / 上限 128) ---
    (
        "bsnd_fp16_ps64_topk256_nocausal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps128_topk128_causal_tail127_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps128_topk256_nocausal_noval",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "tnd_bf16_ps64_topk256_causal_val",
        "TND",
        "TND",
        BF16,
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
    ),
    # --- pool_tail_k 确定性边界(0 / pool_size-1 / 每 batch 混合) ---
    (
        "bsnd_fp16_ps16_tail0_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps16_tail15_nocausal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_ps8_tail_mixed_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "tnd_fp16_ps16_tail15_causal_noval",
        "TND",
        "TND",
        FP16,
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
    ),
    # --- n1 极值(1 / 上限 64)与 S1=1 最小 shape ---
    (
        "bsnd_fp16_n1_1_ps16_nocausal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_n1_64_ps16_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "bsnd_fp16_s1_1_ps16_causal_val",
        "BSND",
        "BSND",
        FP16,
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
    ),
    (
        "tnd_bf16_n1_64_ps8_nocausal_noval",
        "TND",
        "TND",
        BF16,
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
    ),
    # --- PA 细节: 不等池数(-1 空槽+尾块 padding)、block_size 32/64、B=4 ---
    (
        "bsnd_pa_fp16_ps16_unequal_pools_val",
        "BSND",
        "PA_BBND",
        FP16,
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
    ),
    (
        "tnd_pa_fp16_ps16_unequal_pools_causal_val",
        "TND",
        "PA_BBND",
        FP16,
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
    ),
    (
        "bsnd_pa_fp16_ps16_blocksize32_val",
        "BSND",
        "PA_BBND",
        FP16,
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
    ),
    (
        "bsnd_pa_fp16_ps16_blocksize64_b4_causal_val",
        "BSND",
        "PA_BBND",
        FP16,
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
    ),
    # --- PA key 0 轴非连续(keyStride0 属性路径) ---
    (
        "bsnd_pa_fp16_ps16_key_noncontig_val",
        "BSND",
        "PA_BBND",
        FP16,
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
    ),
    (
        "tnd_pa_bf16_ps16_bs32_key_noncontig_causal_val",
        "TND",
        "PA_BBND",
        BF16,
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
    ),
    # NOTE 已知缺陷(arch22/910B, 2026-08-24 二分定位): pool_size>=32 且
    # topk>=6144 的组合(如 TND ps32/topk6144、BSND ps64/topk8192, k=16384)
    # 触发 aicore exception(EE9999/507015), 且崩溃会污染进程内后续用例,
    # 因此不纳入定点用例(避免 CI 雪崩); arch35(950) 复测不复现(2026-08-26),
    # 已由 known_defect_cases.py 缺陷家族 E 覆盖回归; arch22 侧待责任人修复。
]


def _fixed_to_cfg(t):
    """定点元组 -> 统一 cfg dict。"""
    (
        cid,
        lq,
        lk,
        dt,
        batch,
        q_seq,
        k_seq,
        n1,
        topk,
        ps,
        mask,
        rval,
        tail,
        pa_bs,
        noncontig,
    ) = t
    q_lens = list(batch) if isinstance(batch, (list, tuple)) else [q_seq] * batch
    b = len(q_lens)
    k_lens = list(k_seq) if isinstance(k_seq, (list, tuple)) else [k_seq] * b
    k_pools = [kl // ps for kl in k_lens]
    if tail is not None and not isinstance(tail, (list, tuple)):
        tail = [int(tail)] * b
    return dict(
        id=cid,
        layout_q=lq,
        layout_k=lk,
        dtype=dt,
        q_lens=q_lens,
        k_pools=k_pools,
        n1=n1,
        topk=topk,
        pool_size=ps,
        mask_mode=mask,
        return_value=rval,
        pa_block_size=pa_bs,
        key_non_contig=noncontig,
        tail=tail,
        seed=42,
    )


FIXED_CASES = [_fixed_to_cfg(t) for t in _FIXED_TUPLES]

# ---------------------------------------------------------------------------
# 2. 泛化用例(原 test_pool_key_indexer_general.py, 确定性生成)
# ---------------------------------------------------------------------------
# 已实测验证可用的参数包络 (arch22 / Atlas A2 910B3, 2026-08-24):
#   pool_size 规格允许 [1,128] 任意整数; arch22 实测仅 {1} ∪ {8 的倍数} 可用
#   (其余值触发 aicore 崩溃, 根因为向量写非 32B 对齐); arch35(950) 已于
#   2026-08-26 修复全部 pool_size(见 docs/build_fix_record.md), 非 8 倍数由
#   known_defect_cases.py 缺陷家族 C 覆盖; 本随机层采样包络保持不变
#   (泛化基准稳定性)。
MASTER_SEED = 20260824

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
# 5120/6144/7168 存在 arch22 shape 相关崩溃, 不纳入随机层(见缺陷家族 D)。
TOPK_SPECIAL_VALID = [3072, 4096]
# PA block_size: 16 的倍数且 <= 2048, 全档轮转覆盖(含非 2 次幂 48)。
BLOCK_SIZES_VALID = [16, 32, 48, 64, 128, 256, 1024, 2048]

N1_CHOICES = [1, 2, 3, 4, 5, 8, 8, 12, 16, 16, 32, 32, 64, 64]
S1_CHOICES = [1, 2, 4, 8, 16, 16, 32, 32, 64, 128, 256]

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
    for j, cfg in enumerate(DIRECTED_CASES):
        cfg = dict(cfg)
        cfg["seed"] = MASTER_SEED + 10000 + j
        cases.append(cfg)
    return cases


ALL_GENERAL_CASES = _generate_all_cases()

# ---------------------------------------------------------------------------
# 泛化覆盖率跟踪(精度套件 / GE 图套件共享, 模块级状态)
# ---------------------------------------------------------------------------

COVERAGE = {
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


def update_coverage(cfg, k_pools, q_lens):
    """泛化用例执行后更新覆盖统计。"""
    cov = COVERAGE
    cov["cases_executed"] += 1
    cov["layout"].add((cfg["layout_q"], cfg["layout_k"]))
    cov["dtype_layout"].add(
        (str(cfg["dtype"]).replace("torch.", ""), cfg["layout_q"], cfg["layout_k"])
    )
    cov["mask_mode"].add(cfg["mask_mode"])
    cov["return_value"].add(cfg["return_value"])
    cov["pool_size"].add(cfg["pool_size"])
    cov["pool_size_bucket"].add(_ps_bucket(cfg["pool_size"]))
    cov["topk_bucket"].add(_topk_bucket(cfg["topk"]))
    cov["n1"].add(cfg["n1"])
    batch = len(q_lens)
    if batch == 1:
        cov["batch_bucket"].add("b1")
    elif batch <= 3:
        cov["batch_bucket"].add("b2_3")
    else:
        cov["batch_bucket"].add("b4plus")
    if cfg["layout_k"] == "PA_BBND":
        cov["pa_block_size"].add(cfg["pa_block_size"])
    if any(q == 1 for q in q_lens):
        cov["s1_eq_1"] = True
    sparse = cfg["topk"] // cfg["pool_size"]
    if any(0 < p < sparse for p in k_pools):
        cov["pools_lt_sparse"] = True
    if any(p == 0 for p in k_pools):
        cov["zero_pool_batch"] = True
    if any(q == 0 for q in q_lens):
        cov["zero_q_batch"] = True
    if cfg["key_non_contig"]:
        cov["key_non_contig"] = True
    tail = cfg.get("tail")
    if tail is None:
        cov["tail_bound"].add("random")
    elif isinstance(tail, (list, tuple)):
        cov["tail_bound"].add("mixed" if len(set(tail)) > 1 else "fixed")
        if all(t == 0 for t in tail):
            cov["tail_bound"].add("zero")
        if all(t == cfg["pool_size"] - 1 for t in tail):
            cov["tail_bound"].add("max")


# ---------------------------------------------------------------------------
# 3. 因果边界计数场景(原 boundary_causal_test.py 解析公式校验)
# ---------------------------------------------------------------------------
# pool_size=16, topk=16, mask_mode=3, S1=48:
#   场景1: S2=50 池 (L_orig=800+t >> S1) -> 每行恒 16+t 个有效索引
#   场景2: S2=3 池  (L_orig=48+t ≈ S1)  -> 分三段: [0,14-t]:0 |
#           [15-t,14]:16+(i+t-15) | [15,47]:16+t
# tailK 扫描 t=0..15, BSND/TND 双布局。
BOUNDARY_COUNT_PARAMS = [
    (50, "BSND"),
    (50, "TND"),
    (3, "BSND"),
    (3, "TND"),
]
