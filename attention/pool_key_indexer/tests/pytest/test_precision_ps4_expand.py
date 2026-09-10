# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer ps=4 展开路径专项精度用例(Plan-A 前置补测, precision.md §3-A/B)。

背景: 泛化/缺陷套件中 pool_size=4(非 8 倍数, 走 ExpandAndAppendIndices 标量
展开路径)仅 1 例, 且无目标形态 pytest 金标、无 ps=4 因果边界计数扫描 —— 凡
触碰 EXPAND/-1 预填/尾块语义的优化(Plan-A vgather 向量化)存在验证盲区。

用例构成:
  1. ps=4 因果边界计数扫描(§3-A): ps=4/topk=16(sc=4)/S1=48, 池数 {3,50,300}
     (覆盖 L_orig≈S1 / L_orig>>S1 / 多 s2 块), tail t=0..3 全扫 × BSND/TND;
     双口径 = 每行有效计数解析公式 + 标杆集合比对(100% 行 + -1 精确)。
  2. 目标形态 pytest 金标(§3-B): B=1,S1=4096,S2pool=1024,N1=32,ps=4,topk=2048,
     mask=3,BF16,BSND/BSND,return_value=True,pool_tail_k=3, 固定 seed;
     把 perf harness 门禁升级为 pytest 金标口径。
  3. 重复运行稳定性: 目标形态 NPU 侧 3 次重复(同步类改动防偶发, 禁 printf 判
     同步), 每次独立比对金标 + 脏分配器压力(随机 tensor 预分配/释放)。
  4. ps=4 快速回归 3 次重复(d1_bsnd_ps4_topk128 形态, 换 seed 区分数据)。
"""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

import os  # noqa: E402
import sys  # noqa: E402

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

# ---------------------------------------------------------------------------
# 1. ps=4 因果边界计数扫描(§3-A)
# ---------------------------------------------------------------------------

_PS4, _TOPK16 = 4, 16  # sparseCount = topk/ps = 4
_S1_48 = 48
_PS4_BC_POOLS = [3, 50, 300]  # L_orig=12+t(≈S1/4) / 200+t(>>S1 相关) / 1200+t(多 s2 块)
_PS4_BC_LAYOUTS = ["BSND", "TND"]


def _expected_row_counts_ps4(s2_pools, s1, t):
    """ps=4/topk=16/mask=3 每行有效索引数解析公式(独立于 golden 的推导)。

    行 i(0-based): globalPosQ = l_orig - s1 + i;
      可见池数 vis_pools = clamp((globalPosQ+1)//ps, 0, min(总池, topk/ps));
      尾块可见 vis_tail = clamp(min(t, globalPosQ - topk + 1), 0, .) (t>0 时)。
    """
    ps = _PS4
    l_orig = s2_pools * ps + t
    counts = []
    for i in range(s1):
        pools = min(l_orig // ps if l_orig >= 0 else 0, s2_pools)
        cutoff = (l_orig - s1 + i + 1) // ps
        vis_pools = max(0, min(cutoff, pools, _TOPK16 // ps))
        vis_tail = max(0, min(t, (l_orig - s1 + i) - _TOPK16 + 1)) if t > 0 else 0
        counts.append(vis_pools * ps + vis_tail)
    return counts


@pytest.mark.parametrize(
    "s2_pools", _PS4_BC_POOLS, ids=[f"pools{p}" for p in _PS4_BC_POOLS]
)
@pytest.mark.parametrize("layout", _PS4_BC_LAYOUTS)
def test_ps4_boundary_causal_counts(s2_pools, layout, ref_device):
    """ps=4 因果边界计数: tail t=0..3 全扫, 解析公式 + 集合比对双口径。"""
    for t in range(_PS4):
        cfg = dict(
            id=f"ps4bc_{layout}_pools{s2_pools}_t{t}",
            layout_q=layout,
            layout_k=layout,
            dtype=torch.float16,
            q_lens=[_S1_48],
            k_pools=[s2_pools],
            n1=8,
            topk=_TOPK16,
            pool_size=_PS4,
            mask_mode=3,
            return_value=True,
            pa_block_size=16,
            key_non_contig=False,
            tail=[t],
            seed=456 + t,
        )
        inputs = build_case_inputs(cfg)
        g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)
        npu_indices, npu_values = run_op_eager(inputs, cfg)

        assert_output_spec(npu_indices, npu_values, cfg)

        # 1) 每行有效计数 vs 解析推导(-1 预填破坏/截断错/尾块错位最敏感)
        exp = _expected_row_counts_ps4(s2_pools, _S1_48, t)
        rows = npu_indices[0] if layout == "BSND" else npu_indices
        got = [int((r >= 0).sum()) for r in rows]
        assert got == exp, (
            f"[{cfg['id']}] row counts mismatch: "
            f"diff {[(i, e, g) for i, (e, g) in enumerate(zip(exp, got)) if e != g][:4]}"
        )

        # 2) 标杆集合比对(标准门禁: 100% 行 + -1 精确校验)
        ok_idx = compare_indices(g_indices, npu_indices)
        assert ok_idx, f"[{cfg['id']}] indices set comparison failed"
        compare_values(g_values, npu_values, cfg["dtype"], label=cfg["id"])
        print(
            f"[ps4bc] {layout} pools={s2_pools} t={t}: "
            f"rows0/11/47={got[0]}/{got[11]}/{got[47]} PASS"
        )


# ---------------------------------------------------------------------------
# 2. 目标形态 pytest 金标(§3-B)
# ---------------------------------------------------------------------------

TARGET_CFG = dict(
    id="ps4_target_shape_golden",
    layout_q="BSND",
    layout_k="BSND",
    dtype=torch.bfloat16,
    q_lens=[4096],
    k_pools=[1024],  # 池化前 S2=4096
    n1=32,
    topk=2048,
    pool_size=4,
    mask_mode=3,
    return_value=True,
    pa_block_size=16,
    key_non_contig=False,
    tail=[3],  # pool_tail_k=3(perf_compare 因果场景约定, 尾块可见)
    seed=20260910,
)


def _run_target_once(cfg, ref_device):
    inputs = build_case_inputs(cfg)
    g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)
    npu_indices, npu_values = run_op_eager(inputs, cfg)
    assert_output_spec(npu_indices, npu_values, cfg)
    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, f"[{cfg['id']}] indices set comparison failed"
    stats = compare_values(g_values, npu_values, cfg["dtype"], label=cfg["id"])
    return g_indices, npu_indices, stats


def test_ps4_target_shape_golden(ref_device):
    """目标形态金标: B=1,S1=4096,S2pool=1024,N1=32,ps=4,topk=2048,mask3,BF16。"""
    _, _, stats = _run_target_once(TARGET_CFG, ref_device)
    print(
        f"[target] values ref: {stats['pass_rows']}/{stats['total_rows']} rows, "
        f"max_abs_err={stats['max_abs_err']:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. 重复运行稳定性(同步类改动防偶发; 禁 printf 判同步)
# ---------------------------------------------------------------------------

REPEAT_N = 3


def test_ps4_target_shape_repeat_stability(ref_device):
    """目标形态 NPU 侧 3 次重复 + 脏分配器压力, 每次独立金标比对。"""
    # 金标一次(CPU), NPU 重复跑(数据重建保确定性: 同 seed)
    inputs = build_case_inputs(TARGET_CFG)
    g_indices, g_values = run_golden(inputs, TARGET_CFG, ref_device=ref_device)

    for rep in range(REPEAT_N):
        # 脏分配器压力: 随机形状预分配/释放, 让 NPU 内存分配器状态脏化
        for _ in range(4):
            junk = torch.randn(
                torch.randint(1, 512, (1,)).item(),
                1024,
                dtype=torch.float16,
                device="npu",
            )
            del junk
        npu_indices, npu_values = run_op_eager(inputs, TARGET_CFG)
        assert_output_spec(npu_indices, npu_values, TARGET_CFG)
        ok_idx = compare_indices(g_indices, npu_indices)
        assert ok_idx, f"[rep{rep}] indices set comparison failed"
        compare_values(
            g_values, npu_values, TARGET_CFG["dtype"], label=f"target-rep{rep}"
        )
        print(f"[target-rep{rep}] PASS")


def test_ps4_fast_smoke_repeat(ref_device):
    """ps=4 快速回归 3 次重复(d1_bsnd_ps4_topk128 形态, 变 seed)。"""
    for rep in range(REPEAT_N):
        cfg = dict(
            id=f"ps4_fast_rep{rep}",
            layout_q="BSND",
            layout_k="BSND",
            dtype=torch.float16,
            q_lens=[300],
            k_pools=[300],
            n1=8,
            topk=128,
            pool_size=4,
            mask_mode=3,
            return_value=False,
            pa_block_size=16,
            key_non_contig=False,
            tail=[3],
            seed=777 + rep,
        )
        inputs = build_case_inputs(cfg)
        g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)
        npu_indices, npu_values = run_op_eager(inputs, cfg)
        assert_output_spec(npu_indices, npu_values, cfg)
        ok_idx = compare_indices(g_indices, npu_indices)
        assert ok_idx, f"[{cfg['id']}] indices set comparison failed"
        print(f"[fast-rep{rep}] PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
