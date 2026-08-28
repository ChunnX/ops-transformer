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
PoolKeyIndexer torch 单算子(eager)精度测试套件(套件2)。

统一执行路径: CPU/NPU 标杆(pool_key_indexer_reference, 经 --ref-device 切换)
vs NPU torch extension, 输出规格断言 + 集合比对。

用例构成(统一 cfg 格式, 详见 pki_cases.py):
  1. 定点用例(42 例): 布局/dtype/mask/pool_size/topk/tail/n1/PA 非连续定点形态;
  2. 泛化用例(60 例): 确定性随机层(40) + 定向边界层(20), 带覆盖率收口断言;
  3. 已知缺陷回归(25 例): 缺陷家族 C/D/E, 独立子进程执行(EE9999 崩溃只污染
     子进程, 由 _defect_case_runner.py 按输出协议判定);
  4. 因果边界计数(4 组 x tail 全扫): 每行有效索引计数解析公式独立校验
     (golden 之外的独立推导, 交叉验证标杆与算子)。

比对策略(集合比对, 与既有口径一致):
  - indices(第一输出): 逐行有效索引(>=0)排序集合比对, 100% 行通过,
    唯一精度门禁(允许 TopK 输出顺序不同, 选中索引集合不得有差异);
  - values(第二输出): 逐行有效值(有限值)排序数值比对, 仅打印参考日志;
  - 输出 dtype/shape 与接口规格的显式断言(int32 / float32 / 空 values)。

标杆切换:
  pytest test_precision.py                      # 默认 CPU 金标
  pytest test_precision.py --ref-device npu     # NPU 组合标杆(验证两标杆一致性)
"""

import os
import subprocess
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
    get_device_id,
    run_golden,
    run_op_eager,
)
from pki_cases import (  # noqa: E402
    ALL_DEFECT_CASES,
    ALL_GENERAL_CASES,
    BLOCK_SIZES_VALID,
    BOUNDARY_COUNT_PARAMS,
    COVERAGE as _COVERAGE,
    FIXED_CASES,
    update_coverage as _update_coverage,
)
from result_compare import compare_indices, compare_values  # noqa: E402

RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_defect_case_runner.py"
)
CASE_TIMEOUT_SEC = 600  # 首次 JIT 编译兜底; 扩展缓存命中时单用例 ~10-60s


# ---------------------------------------------------------------------------
# 统一用例执行路径
# ---------------------------------------------------------------------------


def _run_precision_case(cfg, ref_device):
    """构造输入 -> 标杆(--ref-device) -> eager 算子 -> 规格断言 + 集合比对。"""
    inputs = build_case_inputs(cfg)

    g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)
    npu_indices, npu_values = run_op_eager(inputs, cfg)

    assert_output_spec(npu_indices, npu_values, cfg)

    ok_idx = compare_indices(g_indices, npu_indices)
    assert ok_idx, (
        f"[{cfg['id']}] indices set comparison failed: "
        f"layout=({cfg['layout_q']},{cfg['layout_k']}) ps={cfg['pool_size']} "
        f"topk={cfg['topk']} mask={cfg['mask_mode']}"
    )
    if cfg["return_value"]:
        compare_values(g_values, npu_values, cfg["dtype"], label="values")


# ---------------------------------------------------------------------------
# 1. 定点用例
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", FIXED_CASES, ids=[c["id"] for c in FIXED_CASES])
def test_precision_fixed(cfg, ref_device):
    """定点用例: 标杆 vs NPU torch extension。"""
    print(
        f"\n=== Fixed: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"dtype={cfg['dtype']} batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} "
        f"k_pools={cfg['k_pools']} n1={cfg['n1']} topk={cfg['topk']} "
        f"ps={cfg['pool_size']} mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"pa_bs={cfg['pa_block_size']} noncontig={cfg['key_non_contig']} "
        f"tail={cfg.get('tail')} ref={ref_device} ==="
    )
    _run_precision_case(cfg, ref_device)
    print(f"=== FIXED PASSED: {cfg['id']} ===")


# ---------------------------------------------------------------------------
# 2. 泛化用例(随机层 + 定向层)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg", ALL_GENERAL_CASES, ids=[c["id"] for c in ALL_GENERAL_CASES]
)
def test_precision_general(cfg, ref_device):
    """泛化用例: 标杆 vs NPU torch extension。"""
    print(
        f"\n=== General: {cfg['id']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
        f"dtype={cfg['dtype']} batch={len(cfg['q_lens'])} q_lens={cfg['q_lens']} "
        f"k_pools={cfg['k_pools']} n1={cfg['n1']} topk={cfg['topk']} ps={cfg['pool_size']} "
        f"mask={cfg['mask_mode']} val={cfg['return_value']} "
        f"pa_bs={cfg['pa_block_size']} noncontig={cfg['key_non_contig']} "
        f"tail={cfg.get('tail')} ref={ref_device} ==="
    )
    _run_precision_case(cfg, ref_device)
    _update_coverage(cfg, cfg["k_pools"], cfg["q_lens"])
    print(f"=== GENERAL PASSED: {cfg['id']} ===")


def test_zz_precision_coverage_summary():
    """覆盖率收口: 校验泛化广度并打印覆盖矩阵与未覆盖功能分析。"""
    cov = _COVERAGE
    print("\n" + "=" * 72)
    print("PoolKeyIndexer 泛化测试覆盖率汇总")
    print("=" * 72)
    print(f"executed cases : {cov['cases_executed']} / {len(ALL_GENERAL_CASES)}")
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
    if cov["cases_executed"] != len(ALL_GENERAL_CASES):
        problems.append(f"仅执行 {cov['cases_executed']}/{len(ALL_GENERAL_CASES)} 例")
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
        "A. [平台限制+实现缺失] quant_mode=0(FP8 per-token-head)/1(mxFP8): 仅\n"
        "   Ascend 950(DAV_3510) 支持; kernel 侧反量化(q_descale/k_descale 消费)\n"
        "   未实现, host 已 fail-fast 拒绝(不会静默出错), 实现并验证后需补泛化。"
    )
    print(
        "B. [文档-实现不一致] BSND + actual_seq_q/actual_seq_k: 接口文档标注\n"
        "   BSND 可选传入有效长度, tiling 实际强制为 null(EZ1008 报错),\n"
        "   实测确认被拒; 需决策: 补实现 or 修文档(标杆已支持该语义)。"
    )
    print(
        "C. [kernel 缺陷, arch35 已修复/arch22 待修] pool_size ∈ [2,7] 及非 8 的\n"
        "   倍数(如 12/100): arch22 任意 shape 触发 aicore exception(EE9999/507015)\n"
        "   且污染进程; 根因为 ExpandAndAppendIndices 向量写偏移(outOff=k*ps /\n"
        "   tailPos=topk / validExpand=v*ps)非 32B 操作数地址对齐。arch35(950)\n"
        "   已于 2026-08-26 修复(ps%8!=0 退化标量精确写路径), 用例由已知缺陷\n"
        "   回归(缺陷家族 C)覆盖。"
    )
    print(
        "D. [kernel 缺陷, arch22 待修] topk ∈ {5120, 6144, 7168}: arch22 shape\n"
        "   相关崩溃(ps=16/topk=5120 在 s2pool=384/512 崩、640 正常; ps=32/\n"
        "   topk>=6144 崩; ps=1/8 全部正常), 崩溃面不稳定; arch35(950) 复测\n"
        "   不复现(2026-08-26, 缺陷家族 D 回归通过, 含原崩溃形态)。"
    )
    print(
        "E. [arch22 待修] pool_size>=32 且 topk>=6144 崩溃(定点用例 NOTE 已记录,\n"
        "   与 D 部分重叠); arch35(950) 复测不复现(2026-08-26, 缺陷家族 E 回归\n"
        "   通过, 含 TND ps32/topk6144 与 BSND ps64/topk8192 原始崩溃形态)。"
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
        "H. [arch35 已闭环] arch22 的 4 处 poolSize>1 行宽修复(批次基址/\n"
        "   清理偏移/values 行宽换算/ProcessInvalid 越界)已按同模式移植到\n"
        "   arch35(950), 并于 2026-08-25 在 950 上板验证 42/42 通过;\n"
        "   2026-08-26 缺陷 C/D/E 在 arch35 复测并修复/确认不复现(见\n"
        "   docs/build_fix_record.md 与已知缺陷回归用例), arch22 侧待责任人处理。"
    )

    assert not problems, "泛化覆盖率不足: " + "; ".join(problems)
    print("\n[SUMMARY] 泛化覆盖率检查通过。")


# ---------------------------------------------------------------------------
# 3. 已知缺陷回归(独立子进程执行)
# ---------------------------------------------------------------------------


def _run_defect_case(case_id, ref_device):
    """在独立子进程中执行单用例, 返回 (verdict, detail)。

    verdict: PASS / FAIL / CRASH / TIMEOUT
    """
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", os.path.dirname(os.path.abspath(__file__)))
    env["PKI_REF_DEVICE"] = ref_device
    try:
        proc = subprocess.run(
            [sys.executable, RUNNER, case_id, "--device", str(get_device_id())],
            capture_output=True,
            text=True,
            timeout=CASE_TIMEOUT_SEC,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"subprocess exceeded {CASE_TIMEOUT_SEC}s"

    # 优先解析子进程输出协议行
    verdict, detail = None, ""
    for line in proc.stdout.splitlines():
        if line.startswith("DEFECT_CASE_RESULT:"):
            tokens = line[len("DEFECT_CASE_RESULT:") :].strip().split(" ", 2)
            verdict = tokens[0]
            detail = tokens[2] if len(tokens) > 2 else ""
            break

    if verdict is None:
        # 协议行缺失: 子进程被信号杀死/解释器级崩溃, 按退出码兜底判定
        if proc.returncode < 0:
            verdict = "CRASH"
            detail = f"killed by signal {-proc.returncode}"
        else:
            verdict = "CRASH"
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            detail = "no result line; last output: " + (
                tail[-1][:300] if tail else "<empty>"
            )
    return verdict, detail


@pytest.mark.parametrize(
    "case", ALL_DEFECT_CASES, ids=[c["id"] for c in ALL_DEFECT_CASES]
)
def test_precision_known_defect(case, ref_device):
    """已知缺陷回归: 独立子进程执行, 期望修复后 PASS。

    缺陷家族(arch22/910B3, 2026-08-24 实测崩溃 EE9999/507015 且崩溃污染
    进程内后续用例; arch35/950 已修复或复测不复现, 本回归保护修复成果):
      - C: pool_size ∈ [2,7] 及非 8 的倍数
      - D: topk ∈ {5120, 6144, 7168} shape 相关崩溃
      - E: pool_size>=32 且 topk>=6144 确定性崩溃
    """
    verdict, detail = _run_defect_case(case["id"], ref_device)
    assert verdict == "PASS", (
        f"[{case['id']}] expected PASS, got {verdict}: {detail} "
        f"(note: {case.get('note', '')})"
    )


def test_defect_case_inventory():
    """缺陷用例清单完整性: 三家族齐备且 id 唯一。"""
    ids = [c["id"] for c in ALL_DEFECT_CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    groups = {}
    for c in ALL_DEFECT_CASES:
        groups.setdefault(c["group"], []).append(c)
    assert len(groups.get("group_pool_size", [])) >= 10
    assert len(groups.get("group_topk_special", [])) >= 8
    assert len(groups.get("group_ps32_topk6k", [])) >= 5
    for c in ALL_DEFECT_CASES:
        assert c["topk"] % c["pool_size"] == 0, (
            f"{c['id']}: topk % pool_size must be 0 (host tiling constraint)"
        )
        assert 1 <= c["pool_size"] <= 128


# ---------------------------------------------------------------------------
# 4. 因果边界计数(解析公式独立校验)
# ---------------------------------------------------------------------------

_PS, _TOPK = 16, 16
_S1 = 48


def _expected_row_counts(s2_pools, s1, t):
    """按推导公式给出每行有效索引数(golden 之外的独立推导)。

    场景: pool_size=16, topk=16, mask_mode=3(rightDownCausal):
      每行有效索引数 = 可见池数 * pool_size + 可见尾 token 数。
    """
    ps = _PS
    l_orig = s2_pools * ps + t
    counts = []
    for i in range(s1):
        pools = min(l_orig // ps if l_orig >= 0 else 0, s2_pools)
        cutoff = (l_orig - s1 + i + 1) // ps
        vis_pools = max(0, min(cutoff, pools, _TOPK // ps))
        vis_tail = max(0, min(t, (l_orig - s1 + i) - _TOPK + 1)) if t > 0 else 0
        counts.append(vis_pools * ps + vis_tail)
    return counts


@pytest.mark.parametrize(
    "s2_pools,layout",
    BOUNDARY_COUNT_PARAMS,
    ids=[f"pools{p}_{l}" for p, l in BOUNDARY_COUNT_PARAMS],
)
def test_precision_boundary_causal_counts(s2_pools, layout, ref_device):
    """因果边界计数推导验证: tail t=0..15 全扫。

    场景1: S2=50 池 (L_orig=800+t >> S1) -> 每行恒 16+t 个有效索引;
    场景2: S2=3 池  (L_orig=48+t ≈ S1)  -> 分三段:
           [0,14-t]: 0 | [15-t,14]: 16+(i+t-15) | [15,47]: 16+t。
    双重校验: 解析公式(行计数) + 标杆集合比对。
    """
    for t in range(_PS):
        cfg = dict(
            id=f"boundary_{layout}_pools{s2_pools}_t{t}",
            layout_q=layout,
            layout_k=layout,
            dtype=torch.float16,
            q_lens=[_S1],
            k_pools=[s2_pools],
            n1=8,
            topk=_TOPK,
            pool_size=_PS,
            mask_mode=3,
            return_value=True,
            pa_block_size=16,
            key_non_contig=False,
            tail=[t],
            seed=123 + t,
        )
        inputs = build_case_inputs(cfg)
        g_indices, g_values = run_golden(inputs, cfg, ref_device=ref_device)
        npu_indices, npu_values = run_op_eager(inputs, cfg)

        # 1) 每行有效计数 vs 解析推导
        exp = _expected_row_counts(s2_pools, _S1, t)
        rows = npu_indices[0] if layout == "BSND" else npu_indices
        got = [int((r >= 0).sum()) for r in rows]
        assert got == exp, (
            f"[{cfg['id']}] row counts mismatch: "
            f"first diff {[(i, e, g) for i, (e, g) in enumerate(zip(exp, got)) if e != g][:4]}"
        )

        # 2) 标杆集合比对(标准门禁)
        ok_idx = compare_indices(g_indices, npu_indices)
        assert ok_idx, f"[{cfg['id']}] indices set comparison failed"
        compare_values(
            g_values, npu_values, cfg["dtype"], label=f"boundary-{cfg['id']}"
        )
        print(
            f"[boundary] {layout} pools={s2_pools} t={t}: "
            f"rows0/15/47={got[0]}/{got[15]}/{got[47]} PASS"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
