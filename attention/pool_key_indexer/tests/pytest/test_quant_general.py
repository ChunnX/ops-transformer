# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and/or it is redistribute it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You may obtain a copy of the License at
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer 量化场景泛化精度测试(quantMode=0/1, 仅 Ascend 950 / arch35)。

与非量化泛化层(pki_cases.ALL_GENERAL_CASES)同构的确定性随机生成 + 定向
边界层, 输出比对口径与 test_quant.py 一致(CPU 金标 vs eager, 集合比对)。

用例构成:
  1. 随机层: 4 布局桶(dense_bsnd/dense_tnd/pa_bsnd/pa_tnd) × 2 quant_mode
     × 每桶 N 例确定性随机(MASTER_SEED 派生);
  2. 定向边界层:
     - S2 跨 s2BaseSize 边界(128/129/256 池, 覆盖 k_descale 多轮乒乓)
     - 0 池 batch / 零长度 query batch(TND)
     - topk 极值(=pool_size / 2048 上限 / 3072 特殊值)
     - n1 极值(1 / 64)与 gSize 非 splitM 形态
      - PA 0 轴非连续 key + k_descale 同步非连续
      - PA bs=2048 + 大 maxBlockNumPerBatch(16 块): scoreGm workspace 布局
        (s2Size=32768) 与 mxFP8/FP8 descale PA 大块分块搬运
     - E8M0 编码边界(raw=0 下溢 / raw=254 大 scale / raw=255 NaN 已由
       test_quant.py NaN 专项覆盖, 此处补 raw 全 127(=1.0) 常量)
     - descale 极值(scale 大幅差: 0.25 vs 3.25, 拉开分数区分度)

运行:  pytest test_quant_general.py            (默认全量)
       pytest test_quant_general.py -k mode0    (仅 quantMode=0)
       PKI_DEVICE_ID=2 pytest test_quant_general.py
"""

import os
import random
import sys

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

if not torch.npu.is_available():
    pytest.skip("NPU device is not available", allow_module_level=True)

if "950" not in torch.npu.get_device_properties(0).name:
    pytest.skip("quant_mode 仅 Ascend950(arch35) 支持", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from pki_common import (  # noqa: E402
    assert_output_spec,
    build_case_inputs,
    run_golden,
    run_op_eager,
)
from result_compare import compare_indices, compare_values  # noqa: E402

FP8 = torch.float8_e4m3fn

# ---------------------------------------------------------------------------
# 随机层生成(确定性, MASTER_SEED 派生)
# ---------------------------------------------------------------------------

QUANT_MASTER_SEED = 20260828

BUCKETS = {
    "dense_bsnd": ("BSND", "BSND"),
    "dense_tnd": ("TND", "TND"),
    "pa_bsnd": ("BSND", "PA_BBND"),
    "pa_tnd": ("TND", "PA_BBND"),
}

N_RANDOM_PER_BUCKET = 8

# 量化随机层采样包络(收敛于已验证形态):
#   pool_size: 1/8/16/32(8 的倍数主档 + 退化档, 避开家族 C 标量路径的
#              随机形态——量化近并列分数叠加非对齐路径, 集合比对噪声大)
POOL_SIZES_QUANT = [1, 8, 16, 32]
POOL_SIZE_WEIGHTS_QUANT = [3, 4, 4, 2]

TOPK_SPECIAL_QUANT = [3072]

PA_BLOCK_SIZES_QUANT = [16, 32, 48, 64, 128, 256]

N1_CHOICES_QUANT = [1, 2, 4, 8, 12, 16, 32, 40, 48, 64]
S1_CHOICES_QUANT = [1, 2, 4, 8, 16, 32, 64]


def _sample_topk(rng, ps):
    """采样 topk: 15% 特殊值(需被 ps 整除), 其余 sparseCount*ps <= 2048。"""
    if rng.random() < 0.15:
        special = rng.choice(TOPK_SPECIAL_QUANT)
        if special % ps == 0:
            return special
    sparse = rng.randint(1, 16) if rng.random() < 0.7 else rng.randint(17, 64)
    topk = sparse * ps
    while topk > 2048:
        sparse = max(1, sparse // 2)
        topk = sparse * ps
    return topk


def _gen_quant_random_case(case_idx, bucket, quant_mode, seed, pa_bs_cycle):
    rng = random.Random(seed)
    layout_q, layout_k = BUCKETS[bucket]
    ps = rng.choices(POOL_SIZES_QUANT, weights=POOL_SIZE_WEIGHTS_QUANT, k=1)[0]
    topk = _sample_topk(rng, ps)
    sparse = topk // ps
    n1 = rng.choice(N1_CHOICES_QUANT)
    batch = rng.randint(1, 5)

    if layout_q == "BSND":
        s1 = rng.choice(S1_CHOICES_QUANT)
        q_lens = [s1] * batch
    else:
        q_lens = [rng.randint(1, 48) for _ in range(batch)]
        if rng.random() < 0.10:  # TND 零长度 query batch
            q_lens[rng.randrange(batch)] = 0
        if sum(q_lens) == 0:
            q_lens[0] = 1

    if layout_k == "BSND":
        pools = rng.randint(1, min(320, max(2, sparse * 2)))
        k_pools = [pools] * batch
    else:
        k_pools = []
        for _ in range(batch):
            if rng.random() < 0.10:
                k_pools.append(0)  # 0 池 batch(TND/PA 允许)
            else:
                k_pools.append(
                    rng.randint(1, min(200, max(2, sparse + rng.randint(1, 48))))
                )
        if sum(k_pools) == 0:
            k_pools[0] = rng.randint(1, 16)

    tail = None
    r = rng.random()
    if r < 0.08:
        tail = [0] * batch
    elif r < 0.16:
        tail = [ps - 1] * batch
    elif r < 0.24:
        tail = [rng.randrange(ps) for _ in range(batch)]

    return {
        "id": f"qgen_{bucket}_m{quant_mode}_{case_idx:02d}",
        "layout_q": layout_q,
        "layout_k": layout_k,
        "dtype": FP8,
        "q_lens": q_lens,
        "k_pools": k_pools,
        "n1": n1,
        "topk": topk,
        "pool_size": ps,
        "mask_mode": rng.choice([0, 3]),
        "return_value": rng.choice([True, False]),
        "pa_block_size": PA_BLOCK_SIZES_QUANT[pa_bs_cycle % len(PA_BLOCK_SIZES_QUANT)],
        "key_non_contig": False,
        "tail": tail,
        "seed": seed,
        "quant_mode": quant_mode,
        # weights dtype 随机(FP16/BF16, 量化场景规格均允许; P1-2 BF16 支持)
        "weights_dtype": rng.choice([torch.float16, torch.bfloat16]),
    }


def _generate_quant_random_cases():
    cases = []
    rng = random.Random(QUANT_MASTER_SEED)
    pa_bs_cycle = 0
    for quant_mode in (0, 1):
        for bucket in BUCKETS:
            for i in range(N_RANDOM_PER_BUCKET):
                seed = rng.randrange(1 << 31)
                cfg = _gen_quant_random_case(i, bucket, quant_mode, seed, pa_bs_cycle)
                if cfg["layout_k"] == "PA_BBND":
                    pa_bs_cycle += 1
                # 每桶第 3 例启用 PA key 非连续
                if bucket.startswith("pa_") and i == 2:
                    cfg["key_non_contig"] = True
                cases.append(cfg)
    return cases


# ---------------------------------------------------------------------------
# 定向边界层
# ---------------------------------------------------------------------------


def _qc(
    cid,
    layout_q,
    layout_k,
    q_lens,
    k_pools,
    n1,
    topk,
    ps,
    mask,
    rval,
    pa_bs=16,
    noncontig=False,
    tail=None,
    quant_mode=0,
    seed=42,
    **kw,
):
    cfg = dict(
        id=cid,
        layout_q=layout_q,
        layout_k=layout_k,
        dtype=FP8,
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
        seed=seed,
        quant_mode=quant_mode,
    )
    cfg.update(kw)
    return cfg


_QUANT_DIRECTED_CASES = [
    # ---- S2 跨 s2BaseSize(128) 边界: k_descale 多轮乒乓刷新 ----
    _qc(
        "qdir_s2_128_m0",
        "BSND",
        "BSND",
        [8],
        [128],
        8,
        256,
        16,
        3,
        True,
        quant_mode=0,
        seed=301,
    ),
    _qc(
        "qdir_s2_129_m0",
        "BSND",
        "BSND",
        [8],
        [129],
        8,
        256,
        16,
        3,
        True,
        quant_mode=0,
        seed=302,
    ),
    _qc(
        "qdir_s2_256_m0",
        "BSND",
        "BSND",
        [8],
        [256],
        8,
        2048,
        16,
        0,
        False,
        quant_mode=0,
        seed=303,
    ),
    _qc(
        "qdir_s2_300_m1",
        "BSND",
        "BSND",
        [8],
        [300],
        8,
        2048,
        16,
        3,
        True,
        quant_mode=1,
        seed=304,
    ),
    _qc(
        "qdir_s2_256_pa_m0",
        "TND",
        "PA_BBND",
        [8, 8],
        [256, 100],
        8,
        512,
        16,
        3,
        True,
        pa_bs=64,
        quant_mode=0,
        seed=305,
    ),
    _qc(
        "qdir_s2_256_pa_m1",
        "TND",
        "PA_BBND",
        [8, 8],
        [200, 256],
        8,
        512,
        16,
        3,
        True,
        pa_bs=64,
        quant_mode=1,
        seed=306,
    ),
    # ---- 0 池 batch / 零长度 query ----
    _qc(
        "qdir_zeropool_tnd_m0",
        "TND",
        "TND",
        [8, 4],
        [0, 32],
        8,
        64,
        8,
        3,
        True,
        quant_mode=0,
        seed=307,
    ),
    _qc(
        "qdir_zeropool_tnd_m1",
        "TND",
        "TND",
        [4, 8],
        [32, 0],
        8,
        64,
        8,
        0,
        True,
        quant_mode=1,
        seed=308,
    ),
    _qc(
        "qdir_zeropool_pa_m0",
        "TND",
        "PA_BBND",
        [8, 8, 8],
        [0, 48, 0],
        8,
        128,
        8,
        3,
        True,
        quant_mode=0,
        seed=309,
    ),
    _qc(
        "qdir_zeroq_tnd_m0",
        "TND",
        "TND",
        [0, 8, 0],
        [32, 32, 32],
        8,
        64,
        8,
        3,
        True,
        quant_mode=0,
        seed=310,
    ),
    _qc(
        "qdir_zeroq_tnd_m1",
        "TND",
        "TND",
        [0, 8],
        [48, 16],
        8,
        64,
        8,
        0,
        False,
        quant_mode=1,
        seed=311,
    ),
    # ---- topk 极值 ----
    _qc(
        "qdir_topk_eq_ps_m0",
        "BSND",
        "BSND",
        [8],
        [64],
        8,
        16,
        16,
        3,
        True,
        quant_mode=0,
        seed=312,
    ),
    _qc(
        "qdir_topk_eq_ps_m1",
        "BSND",
        "BSND",
        [8],
        [64],
        8,
        32,
        32,
        3,
        True,
        quant_mode=1,
        seed=313,
    ),
    _qc(
        "qdir_topk_2048_m0",
        "BSND",
        "BSND",
        [4],
        [160],
        8,
        2048,
        8,
        3,
        False,
        quant_mode=0,
        seed=314,
    ),
    _qc(
        "qdir_topk_3072_m1",
        "BSND",
        "BSND",
        [4],
        [240],
        8,
        3072,
        8,
        3,
        True,
        quant_mode=1,
        seed=315,
    ),
    # ---- n1 极值 ----
    _qc(
        "qdir_n1_1_m0",
        "BSND",
        "BSND",
        [16],
        [64],
        1,
        128,
        8,
        3,
        True,
        quant_mode=0,
        seed=316,
    ),
    _qc(
        "qdir_n1_64_m0",
        "BSND",
        "BSND",
        [16],
        [64],
        64,
        128,
        1,
        3,
        True,
        quant_mode=0,
        seed=317,
    ),
    _qc(
        "qdir_n1_64_m1",
        "BSND",
        "BSND",
        [16],
        [64],
        64,
        128,
        8,
        3,
        True,
        quant_mode=1,
        seed=318,
    ),
    # ---- PA 非连续(key + k_descale 同步 0 轴非连续) ----
    _qc(
        "qdir_pa_noncontig_m0",
        "TND",
        "PA_BBND",
        [8, 8],
        [48, 32],
        8,
        64,
        8,
        3,
        True,
        noncontig=True,
        quant_mode=0,
        seed=319,
    ),
    _qc(
        "qdir_pa_noncontig_m1",
        "TND",
        "PA_BBND",
        [8, 8],
        [48, 32],
        8,
        64,
        8,
        3,
        True,
        noncontig=True,
        quant_mode=1,
        seed=320,
    ),
    _qc(
        "qdir_pa_bs1024_m0",
        "BSND",
        "PA_BBND",
        [8],
        [100],
        8,
        128,
        8,
        3,
        True,
        pa_bs=1024,
        quant_mode=0,
        seed=321,
    ),
    _qc(
        "qdir_pa_bs1024_m1",
        "BSND",
        "PA_BBND",
        [8],
        [100],
        8,
        128,
        8,
        3,
        True,
        pa_bs=1024,
        quant_mode=1,
        seed=322,
    ),
    # ---- PA bs=2048 + 大 maxBlockNumPerBatch(≥16 块): scoreGm workspace 布局 ----
    # s2Size = maxBlockNumPerBatch * blockSize = 16 * 2048 = 32768, 直接放大
    # arch35 scoreGm 区域(kernel Init: aicNum * s1BaseSize * Align(s2Size,128)
    # * 4B, kernel_arch35.h) 与 host CalcWorkspaceSize 的 scoreGmSize35 口径
    # 一致性; 同时覆盖 mxFP8 LoadKScaleToL1 / FP8 GetKeyScale 的 PA 分块搬运
    # (kCacheBlockSize 除模寻址)在大块下的正确性。k_pools=[32000, 2049]:
    # batch0 16 块(15 满 + 2720 尾池), batch1 2 块, block_table 宽 16 且
    # batch1 含 -1 空槽(混合块数 + 空槽查表)。
    _qc(
        "qdir_pa_bs2048_blk16_m1",
        "BSND",
        "PA_BBND",
        [8, 8],
        [32000, 2049],
        8,
        128,
        8,
        3,
        True,
        pa_bs=2048,
        quant_mode=1,
        seed=328,
    ),
    _qc(
        "qdir_pa_bs2048_blk16_m0",
        "BSND",
        "PA_BBND",
        [8, 8],
        [32000, 2049],
        8,
        128,
        8,
        3,
        True,
        pa_bs=2048,
        quant_mode=0,
        seed=329,
    ),
    # ---- tail 边界 ----
    _qc(
        "qdir_tail_max_m0",
        "BSND",
        "BSND",
        [16],
        [32],
        8,
        128,
        16,
        3,
        True,
        tail=[15],
        quant_mode=0,
        seed=323,
    ),
    _qc(
        "qdir_tail_max_m1",
        "BSND",
        "BSND",
        [16],
        [32],
        8,
        128,
        32,
        3,
        True,
        tail=[31],
        quant_mode=1,
        seed=324,
    ),
    _qc(
        "qdir_tail_mixed_m0",
        "TND",
        "TND",
        [8, 8, 8],
        [32, 32, 32],
        8,
        128,
        16,
        3,
        True,
        tail=[0, 15, 7],
        quant_mode=0,
        seed=325,
    ),
    # ---- batch 维度 ----
    _qc(
        "qdir_batch6_m0",
        "BSND",
        "BSND",
        [8] * 6,
        [64] * 6,
        8,
        128,
        8,
        3,
        True,
        quant_mode=0,
        seed=326,
    ),
    _qc(
        "qdir_batch5_tnd_m1",
        "TND",
        "TND",
        [4, 8, 12, 16, 20],
        [24, 48, 12, 0, 36],
        8,
        128,
        8,
        0,
        True,
        quant_mode=1,
        seed=327,
    ),
]

QUANT_RANDOM_CASES = _generate_quant_random_cases()
ALL_QUANT_GENERAL_CASES = QUANT_RANDOM_CASES + _QUANT_DIRECTED_CASES


# ---------------------------------------------------------------------------
# 执行路径(与 test_quant.py 同口径)
# ---------------------------------------------------------------------------


def _run_quant_case(cfg):
    inputs = build_case_inputs(cfg)
    g_indices, g_values = run_golden(inputs, cfg, ref_device="cpu")
    npu_indices, npu_values = run_op_eager(inputs, cfg)
    assert_output_spec(npu_indices, npu_values, cfg)
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{cfg['id']}] indices set comparison failed: "
        f"layout=({cfg['layout_q']},{cfg['layout_k']}) ps={cfg['pool_size']} "
        f"topk={cfg['topk']} mask={cfg['mask_mode']} quant={cfg['quant_mode']} "
        f"q_lens={cfg['q_lens']} k_pools={cfg['k_pools']}"
    )
    if cfg["return_value"]:
        compare_values(g_values, npu_values, cfg["dtype"], label="values")


@pytest.mark.parametrize(
    "cfg", ALL_QUANT_GENERAL_CASES, ids=[c["id"] for c in ALL_QUANT_GENERAL_CASES]
)
def test_quant_general(cfg):
    """量化泛化用例(随机层 + 定向边界层): CPU 金标 vs NPU eager。"""
    print(
        f"\n=== QuantGen: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} k_pools={cfg['k_pools']} "
        f"n1={cfg['n1']} topk={cfg['topk']} ps={cfg['pool_size']} "
        f"mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"quant={cfg['quant_mode']} pa_bs={cfg['pa_block_size']} "
        f"noncontig={cfg['key_non_contig']} tail={cfg.get('tail')} ==="
    )
    _run_quant_case(cfg)
    print(f"=== QUANT GEN PASSED: {cfg['id']} ===")
