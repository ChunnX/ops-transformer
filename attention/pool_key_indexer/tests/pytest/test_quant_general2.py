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
PoolKeyIndexer 量化场景泛化测试第二层(扩展包络, quantMode=0/1, 仅 Ascend 950)。

与 test_quant_general.py(第一层)互补, 扩大采样包络:
  - pool_size: 24/64/128(非 8 倍数 24 走 arch35 标量展开路径; 64/128 大池)
  - topk: 4096 特殊值档
  - S1/S2/batch 更大规模(S2 最高 ~700, 跨 5+ 个 s2 基本块)
  - E8M0 编码宽分布(raw 100..160, scale 跨 2^-27..2^33, 拉大分数分离度)
  - descale 数值形态: 大 scale 差(0.01 与 100 混布)拉开池间区分度,
    压缩 tie-break 噪声对集合比对的干扰

随机层: 4 布局桶 × 2 mode × 6 例(独立 MASTER_SEED);
定向层: 大 S2 多块(s2Idx>1 乒乓滚动)、ps=24 标量展开、ps=128、topk=4096、
  E8M0 极端编码(全 100 / 全 160 / 混合)、S1 尾块(gS1 非整块)、深 batch。

运行:  pytest test_quant_general2.py
       pytest test_quant_general2.py -k m1
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
# 随机层(第二层包络, 独立 MASTER_SEED)
# ---------------------------------------------------------------------------

QUANT2_MASTER_SEED = 20260829

BUCKETS2 = {
    "dense_bsnd": ("BSND", "BSND"),
    "dense_tnd": ("TND", "TND"),
    "pa_bsnd": ("BSND", "PA_BBND"),
    "pa_tnd": ("TND", "PA_BBND"),
}

N_RANDOM_PER_BUCKET2 = 6

# 第二层包络: 大 ps(64/128) + 非 8 倍数(24, arch35 标量展开) + 常规档
POOL_SIZES2 = [1, 8, 16, 24, 32, 64, 128]
POOL_SIZE_WEIGHTS2 = [1, 2, 2, 2, 2, 2, 1]

TOPK_SPECIAL2 = [3072, 4096]

PA_BLOCK_SIZES2 = [16, 32, 48, 64, 128, 256, 512]

N1_CHOICES2 = [1, 2, 4, 8, 16, 32, 48, 64]
S1_CHOICES2 = [1, 2, 8, 16, 32, 64, 128]
# S2 上限抬到 ~700(跨 5+ s2 基本块, 深度覆盖 k_descale 乒乓滚动)
S2_MAX2 = 700


def _is_valid_topk(topk):
    """规格合法档位: [1, 2048] 或 {3072, 4096, 5120, 6144, 7168, 8192}。"""
    return (1 <= topk <= 2048) or topk in (3072, 4096, 5120, 6144, 7168, 8192)


def _sample_topk2(rng, ps):
    """topk: 25% 特殊值(需被 ps 整除), 其余 sparse*ps; 必须落在合法档位内。"""
    for _ in range(64):  # 重试直到合法(sparse*ps 常落在非法档位间隙)
        if rng.random() < 0.25:
            special = rng.choice(TOPK_SPECIAL2)
            if special % ps == 0:
                return special
        else:
            sparse = rng.randint(1, 20) if rng.random() < 0.7 else rng.randint(21, 128)
            topk = sparse * ps
            if _is_valid_topk(topk):
                return topk
    # 兜底: 特殊值整除档(3072 = ps*? 仅特定 ps 可整除), 否则 ps 本身(sparse=1 合法)
    for special in (4096, 3072):
        if special % ps == 0:
            return special
    return ps


def _gen_quant2_random_case(case_idx, bucket, quant_mode, seed, pa_bs_cycle):
    rng = random.Random(seed)
    layout_q, layout_k = BUCKETS2[bucket]
    ps = rng.choices(POOL_SIZES2, weights=POOL_SIZE_WEIGHTS2, k=1)[0]
    topk = _sample_topk2(rng, ps)
    sparse = topk // ps
    n1 = rng.choice(N1_CHOICES2)
    batch = rng.randint(1, 6)

    if layout_q == "BSND":
        s1 = rng.choice(S1_CHOICES2)
        q_lens = [s1] * batch
    else:
        q_lens = [rng.randint(1, 96) for _ in range(batch)]
        if rng.random() < 0.10:
            q_lens[rng.randrange(batch)] = 0
        if sum(q_lens) == 0:
            q_lens[0] = 1

    if layout_k == "BSND":
        pools = rng.randint(1, min(S2_MAX2, max(2, sparse * 3)))
        k_pools = [pools] * batch
    else:
        k_pools = []
        for _ in range(batch):
            if rng.random() < 0.08:
                k_pools.append(0)
            else:
                k_pools.append(
                    rng.randint(1, min(500, max(2, sparse * 2 + rng.randint(1, 96))))
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
        "id": f"q2gen_{bucket}_m{quant_mode}_{case_idx:02d}",
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
        "pa_block_size": PA_BLOCK_SIZES2[pa_bs_cycle % len(PA_BLOCK_SIZES2)],
        "key_non_contig": False,
        "tail": tail,
        "seed": seed,
        "quant_mode": quant_mode,
        # weights dtype 随机(FP16/BF16, 量化场景规格均允许; P1-2 BF16 支持)
        "weights_dtype": rng.choice([torch.float16, torch.bfloat16]),
    }


def _generate_quant2_random_cases():
    cases = []
    rng = random.Random(QUANT2_MASTER_SEED)
    pa_bs_cycle = 0
    for quant_mode in (0, 1):
        for bucket in BUCKETS2:
            for i in range(N_RANDOM_PER_BUCKET2):
                seed = rng.randrange(1 << 31)
                cfg = _gen_quant2_random_case(i, bucket, quant_mode, seed, pa_bs_cycle)
                if cfg["layout_k"] == "PA_BBND":
                    pa_bs_cycle += 1
                if bucket.startswith("pa_") and i == 1:
                    cfg["key_non_contig"] = True
                cases.append(cfg)
    return cases


# ---------------------------------------------------------------------------
# 定向边界层(第二层)
# ---------------------------------------------------------------------------


def _qc2(
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


_QUANT2_DIRECTED_CASES = [
    # ---- 大 S2 多块(s2Idx 深滚动, k_descale 乒乓第 2 轮+) ----
    _qc2(
        "q2dir_s2_512_m0",
        "BSND",
        "BSND",
        [8],
        [512],
        8,
        2048,
        16,
        3,
        True,
        quant_mode=0,
        seed=401,
    ),
    _qc2(
        "q2dir_s2_700_m0",
        "BSND",
        "BSND",
        [4],
        [700],
        8,
        2048,
        16,
        0,
        False,
        quant_mode=0,
        seed=402,
    ),
    _qc2(
        "q2dir_s2_600_pa_m1",
        "TND",
        "PA_BBND",
        [8, 8],
        [300, 300],
        8,
        2048,
        16,
        3,
        True,
        pa_bs=128,
        quant_mode=1,
        seed=403,
    ),
    _qc2(
        "q2dir_s2_512_tnd_m1",
        "TND",
        "TND",
        [8, 12],
        [256, 256],
        8,
        4096,
        32,
        3,
        True,
        quant_mode=1,
        seed=404,
    ),
    # ---- ps=24(非 8 倍数, arch35 标量展开路径) ----
    _qc2(
        "q2dir_ps24_m0",
        "BSND",
        "BSND",
        [16],
        [64],
        8,
        384,
        24,
        3,
        True,
        quant_mode=0,
        seed=405,
    ),
    _qc2(
        "q2dir_ps24_m1",
        "BSND",
        "BSND",
        [16],
        [64],
        8,
        384,
        24,
        3,
        True,
        quant_mode=1,
        seed=406,
    ),
    _qc2(
        "q2dir_ps24_causal_m0",
        "TND",
        "TND",
        [8, 8],
        [32, 24],
        8,
        240,
        24,
        3,
        True,
        tail=[7, 23],
        quant_mode=0,
        seed=407,
    ),
    # ---- ps=128(最大池) ----
    _qc2(
        "q2dir_ps128_m0",
        "BSND",
        "BSND",
        [8],
        [16],
        8,
        2048,
        128,
        3,
        True,
        tail=[100],
        quant_mode=0,
        seed=408,
    ),
    _qc2(
        "q2dir_ps128_m1",
        "BSND",
        "BSND",
        [8],
        [16],
        8,
        1024,
        128,
        3,
        True,
        tail=[127],
        quant_mode=1,
        seed=409,
    ),
    # ---- topk=4096(量化档上限) ----
    _qc2(
        "q2dir_topk4096_m0",
        "BSND",
        "BSND",
        [4],
        [256],
        8,
        4096,
        16,
        3,
        False,
        quant_mode=0,
        seed=410,
    ),
    _qc2(
        "q2dir_topk4096_m1",
        "BSND",
        "BSND",
        [4],
        [256],
        8,
        4096,
        32,
        3,
        True,
        quant_mode=1,
        seed=411,
    ),
    # ---- E8M0 编码极端形态(数值分离度拉满) ----
    _qc2(
        "q2dir_e8m0_wide_m1",
        "BSND",
        "BSND",
        [8],
        [64],
        8,
        128,
        8,
        3,
        True,
        quant_mode=1,
        seed=412,
        mx_raw_range=(100, 160),
    ),
    _qc2(
        "q2dir_e8m0_min_m1",
        "BSND",
        "BSND",
        [8],
        [64],
        8,
        128,
        8,
        3,
        True,
        quant_mode=1,
        seed=413,
        mx_raw_range=(0, 0),
    ),
    # ---- mode=0 descale 大幅差(0.01 与 100 混布) ----
    _qc2(
        "q2dir_ds_spread_m0",
        "BSND",
        "BSND",
        [8],
        [64],
        8,
        128,
        8,
        3,
        True,
        quant_mode=0,
        seed=414,
        ds_spread=True,
    ),
    _qc2(
        "q2dir_ds_spread_tnd_m0",
        "TND",
        "TND",
        [8, 16],
        [48, 32],
        8,
        128,
        8,
        0,
        True,
        quant_mode=0,
        seed=415,
        ds_spread=True,
    ),
    # ---- S1 尾块(gS1 非整块, mBasicSizeTail) ----
    _qc2(
        "q2dir_s1tail_m0",
        "BSND",
        "BSND",
        [6],
        [64],
        8,
        128,
        8,
        3,
        True,
        quant_mode=0,
        seed=416,
    ),
    _qc2(
        "q2dir_s1tail_m1",
        "TND",
        "TND",
        [3, 5, 7],
        [32, 32, 32],
        8,
        128,
        8,
        3,
        True,
        quant_mode=1,
        seed=417,
    ),
    # ---- 深 batch ----
    _qc2(
        "q2dir_batch8_pa_m0",
        "TND",
        "PA_BBND",
        [4] * 8,
        [16, 0, 32, 8, 0, 24, 16, 40],
        8,
        256,
        8,
        3,
        True,
        quant_mode=0,
        seed=418,
    ),
    _qc2(
        "q2dir_batch8_tnd_m1",
        "TND",
        "TND",
        [2, 4, 6, 8, 10, 12, 14, 16],
        [8, 16, 24, 0, 32, 40, 48, 56],
        8,
        256,
        8,
        0,
        True,
        quant_mode=1,
        seed=419,
    ),
    # ---- PA 块尺寸边界(512/非 2 幂 48) ----
    _qc2(
        "q2dir_pa_bs512_m0",
        "BSND",
        "PA_BBND",
        [8],
        [200],
        8,
        256,
        16,
        3,
        True,
        pa_bs=512,
        quant_mode=0,
        seed=420,
    ),
    _qc2(
        "q2dir_pa_bs48_m1",
        "TND",
        "PA_BBND",
        [8, 8],
        [48, 96],
        8,
        256,
        16,
        3,
        True,
        pa_bs=48,
        quant_mode=1,
        seed=421,
    ),
    # ---- n1=48(非 2 幂头数) ----
    _qc2(
        "q2dir_n1_48_m0",
        "BSND",
        "BSND",
        [16],
        [64],
        48,
        128,
        8,
        3,
        True,
        quant_mode=0,
        seed=422,
    ),
    _qc2(
        "q2dir_n1_48_m1",
        "TND",
        "TND",
        [8, 8],
        [32, 32],
        48,
        128,
        8,
        3,
        True,
        quant_mode=1,
        seed=423,
    ),
    # ---- BF16 weights(P1-2 支持, 定向覆盖 mode=0/1 × PA/非 PA) ----
    _qc2(
        "q2dir_bf16w_m0_pa",
        "TND",
        "PA_BBND",
        [8, 8],
        [48, 32],
        8,
        64,
        8,
        3,
        True,
        quant_mode=0,
        seed=424,
        weights_dtype=torch.bfloat16,
    ),
    _qc2(
        "q2dir_bf16w_m1_s2multi",
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
        seed=425,
        weights_dtype=torch.bfloat16,
    ),
]

QUANT2_RANDOM_CASES = _generate_quant2_random_cases()
ALL_QUANT2_GENERAL_CASES = QUANT2_RANDOM_CASES + _QUANT2_DIRECTED_CASES


# ---------------------------------------------------------------------------
# 执行路径(与第一层一致; 支持 mx_raw_range / ds_spread 注入)
# ---------------------------------------------------------------------------


def _inject_descale_variants(inputs, cfg):
    """按 cfg 的扩展字段重生成 descale(E8M0 编码范围 / mode=0 大幅差)。"""
    mx_range = cfg.get("mx_raw_range")
    if mx_range is not None and inputs.get("k_descale") is not None:
        import torch as _t

        lo, hi = mx_range
        for key in ("q_descale", "k_descale"):
            if inputs.get(key) is not None:
                shape = inputs[key].shape
                inputs[key] = _t.randint(lo, hi + 1, shape, dtype=_t.uint8)
    if cfg.get("ds_spread") and inputs.get("k_descale") is not None:
        import torch as _t

        # mode=0: scale 大幅差混布(0.01 / 100), 拉开分数区分度
        for key in ("q_descale", "k_descale"):
            if inputs.get(key) is not None:
                shape = inputs[key].shape
                mask = _t.rand(shape) < 0.5
                inputs[key] = _t.where(
                    mask, _t.full(shape, 0.01), _t.full(shape, 100.0)
                )
    return inputs


def _run_quant2_case(cfg):
    inputs = build_case_inputs(cfg)
    inputs = _inject_descale_variants(inputs, cfg)
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
    "cfg", ALL_QUANT2_GENERAL_CASES, ids=[c["id"] for c in ALL_QUANT2_GENERAL_CASES]
)
def test_quant_general2(cfg):
    """量化泛化第二层(扩展包络): CPU 金标 vs NPU eager。"""
    print(
        f"\n=== QuantGen2: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} k_pools={cfg['k_pools']} "
        f"n1={cfg['n1']} topk={cfg['topk']} ps={cfg['pool_size']} "
        f"mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"quant={cfg['quant_mode']} pa_bs={cfg['pa_block_size']} "
        f"noncontig={cfg['key_non_contig']} tail={cfg.get('tail')} "
        f"mx_raw={cfg.get('mx_raw_range')} ds_spread={cfg.get('ds_spread')} ==="
    )
    _run_quant2_case(cfg)
    print(f"=== QUANT GEN2 PASSED: {cfg['id']} ===")
