#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""AllGatherMatmulV2 (non-quant) canonical case table.

Single source of truth for the L0/L1/L2 case matrix of test plan 2.1 §5
(修订版 2026-09-19 CP2.1). Every case carries its plan id so the simulation
driver (tests/sim/run_sim_cases.sh), the ST json generator
(tests/sim/gen_st_json.py) and the UT assets stay 1:1 traceable to the plan.

Factor baseline (2.1 §5 缺省参数约定): cases that do not override a factor
use bias=False, trans_b=False, gather_out=True, comm_mode="ai_cpu",
rank_size=2, value_spec="clean", repeat_launch=1.

Levels:
  L0 门槛 (4)   -> channel A (cannsim simulation)
  L1 功能 (31)  -> channel A (cannsim simulation); L1-142 conditional
  L2 异常 (22)  -> channel B (op_host tiling UT csv) / C (op_api UT)
ST mirror (28) = L0(4) + L1-S1(8) + S2(5) + S3(4) + S4(4) + S7(3), see
gen_st_json.py; card count is a runner (WS) config, not a case dimension.
"""

CLEAN = "clean"                 # [-5, 5] layered sampling (incl. small band)
SIGNED_ZERO = "signed_zero"     # +/-0 mixed into x1
INF_NAN = "inf_nan"             # +/-inf/nan mixed into x1
MAG_BANDS = "magnitude_bands"   # layered magnitude bands (small/negative mirror)


def _case(cid, desc, *, dtype="fp16", rank_size=2, comm_mode="ai_cpu",
          m=128, k=768, n=128, bias=False, trans_b=False, gather_out=True,
          value_spec=CLEAN, repeat_launch=1, conditional=False,
          x2_shape=None, level="L1"):
    """Build one functional case dict (channel A / ST mirror factor domain)."""
    return {
        "id": cid,
        "level": level,
        "desc": desc,
        "dtype": dtype,
        "rank_size": rank_size,
        "comm_mode": comm_mode,
        "m": m,
        "k": k,
        "n": n,
        "bias": bias,
        "trans_b": trans_b,
        "gather_out": gather_out,
        "value_spec": value_spec,
        "repeat_launch": repeat_launch,
        "conditional": conditional,
        # x2 API-view shape override (L1-162 n=0: x2 = (k, 0))
        "x2_shape": x2_shape or ([n, k] if trans_b else [k, n]),
    }


# ---------------------------------------------------------------------------
# L0 门槛用例 (2.1 §5.1) — dtype × comm_mode 基础正例
# ---------------------------------------------------------------------------

L0_CASES = [
    _case("L0-01", "FP16 基础正例（ai_cpu）", dtype="fp16", comm_mode="ai_cpu",
          level="L0"),
    _case("L0-02", "BF16 基础正例（ai_cpu）", dtype="bf16", comm_mode="ai_cpu",
          level="L0"),
    _case("L0-03", "FP16 基础正例（ccu）", dtype="fp16", comm_mode="ccu",
          level="L0"),
    _case("L0-04", "BF16 基础正例（ccu）", dtype="bf16", comm_mode="ccu",
          level="L0"),
]

# ---------------------------------------------------------------------------
# L1 功能用例 (2.1 §5.2) — 31 例（含 1 条件例 L1-142）
# ---------------------------------------------------------------------------

L1_CASES = [
    # S1 参数域 pairwise：与 L0-01~04 合计 12 例实现 5 因子全两两组合
    _case("L1-101", "fp16/ai_cpu/bias 有/transB/gatherOut",
          dtype="fp16", comm_mode="ai_cpu", bias=True, trans_b=True,
          gather_out=True),
    _case("L1-102", "fp16/ai_cpu/无 bias/transB/无 gatherOut",
          dtype="fp16", comm_mode="ai_cpu", bias=False, trans_b=True,
          gather_out=False),
    _case("L1-103", "fp16/ccu/bias 有/不转置/无 gatherOut",
          dtype="fp16", comm_mode="ccu", bias=True, trans_b=False,
          gather_out=False),
    _case("L1-104", "bf16/ai_cpu/bias 有/transB/无 gatherOut",
          dtype="bf16", comm_mode="ai_cpu", bias=True, trans_b=True,
          gather_out=False),
    _case("L1-105", "bf16/ccu/bias 有/transB/gatherOut",
          dtype="bf16", comm_mode="ccu", bias=True, trans_b=True,
          gather_out=True),
    _case("L1-106", "bf16/ccu/无 bias/transB/无 gatherOut",
          dtype="bf16", comm_mode="ccu", bias=False, trans_b=True,
          gather_out=False),
    _case("L1-107", "fp16/ccu/bias 有/transB/无 gatherOut",
          dtype="fp16", comm_mode="ccu", bias=True, trans_b=True,
          gather_out=False),
    _case("L1-108", "bf16/ai_cpu/无 bias/不转置/无 gatherOut",
          dtype="bf16", comm_mode="ai_cpu", bias=False, trans_b=False,
          gather_out=False),
    # S2 k 边界（R=2, ai_cpu, m=128, n=128, transB=false, gatherOut=true）
    _case("L1-110", "k 下界 256", k=256),
    _case("L1-111", "k 非对齐 1001", k=1001),
    _case("L1-112", "k 上界内最大 65534（K≥SPLIT_K_THRES=27392 切K触发域）",
          m=137, k=65534, n=128),
    _case("L1-113", "k 下界 bf16", dtype="bf16", k=256),
    _case("L1-114", "k 上界 × transB × bias 组合",
          dtype="bf16", m=137, k=65534, n=128, bias=True, trans_b=True),
    # S3 m 尾块与非对齐（k=768, n=128, transB=false, gatherOut=true）
    _case("L1-120", "m 尾块 1200（触发尾块 tiling/计算分支）", m=1200),
    _case("L1-121", "m 非对齐 137", m=137),
    _case("L1-122", "m 最小 1", m=1, bias=True),
    _case("L1-123", "m 尾块 × transB", dtype="bf16", m=1200, trans_b=True),
    # S4 n 边界与大 N 切 K 触发域（k=768, m=128）
    _case("L1-130", "n 最小 1", n=1, bias=True),
    _case("L1-131", "n 非对齐 777", n=777, bias=True),
    _case("L1-132", "n 非对齐 × transB", dtype="bf16", n=777, bias=True,
          trans_b=True),
    _case("L1-133",
          "大 N(≥1024) 大 M 切 K 触发域（L2 cache tile 失败支；"
          "与 L1-112/114 共同覆盖 2.2 F4 两类切K触发条件）",
          m=2048, k=4096, n=2048, bias=True),
    # S5 卡数规模
    _case("L1-140", "4 卡 ai_cpu", rank_size=4),
    _case("L1-141", "4 卡 ccu 组合", dtype="bf16", rank_size=4,
          comm_mode="ccu", bias=True, trans_b=True),
    _case("L1-142", "（条件）8 卡——仿真代价可接受时执行（A5 条件项）",
          rank_size=8, conditional=True),
    # S6 特殊值（R=2, ai_cpu, m=128, k=768, n=128）
    _case("L1-150", "x1 含 ±0（gatherOut bitwise 符号零保持硬门槛）",
          value_spec=SIGNED_ZERO),
    _case("L1-151", "x1 含 ±inf/nan（gatherOut bitwise 硬门槛；"
                    "output 走传播语义核对）",
          value_spec=INF_NAN),
    _case("L1-152", "量级带覆盖（小值/负镜像分层采样）",
          dtype="bf16", value_spec=MAG_BANDS),
    # S7 空 tensor 与 n=0 分支（R=2）
    _case("L1-160", "x1 空（m=0）短路：op_api DealWithX1Empty 路径",
          m=0, gather_out=True),
    _case("L1-161", "x1 空 + gatherOut 空传入", m=0, gather_out=True),
    _case("L1-162", "x2 n=0 只通信分支：kernel rankN==0，"
                    "output=(m*R,0) 空，gatherOut 仍全量通信",
          n=0, x2_shape=[768, 0]),
    # S8 确定性回归
    _case("L1-170", "连续重复 launch：同参数连续 2 次调用输出 bitwise 一致",
          repeat_launch=2),
]

# ---------------------------------------------------------------------------
# L2 异常用例 (2.1 §5.3) — 22 例，通道 B（tiling UT csv）/ C（op_api UT）
# asset 字段：实现该用例的 UT 资产（csv 行名 / gtest 用例名），status:
#   kept    既有资产核对保留
#   added   本次补全新增
#   pending 行为对齐后冻结（当前无在用例——L2-213 已随 T-A5 落地转 added）
# ---------------------------------------------------------------------------


def _l2(cid, desc, params, expect, channel, asset, status, note=""):
    return {
        "id": cid,
        "level": "L2",
        "desc": desc,
        "params": params,
        "expect": expect,
        "channel": channel,
        "asset": asset,
        "status": status,
        "note": note,
    }


L2_CASES = [
    _l2("L2-201", "amaxOut 非空拒绝", "amaxOut=(1,) fp32 非空",
        "ACLNN_ERR_PARAM_INVALID", "C", "TestAmaxOutWithFP16 等", "kept"),
    _l2("L2-202", "scale 非空拒绝（非量化）",
        "x1Scale/x2Scale/quantScale 分别非空", "拒绝（tiling CheckInput）",
        "B+C", "ut_quant_scale_must_be_none_fp16_950 / "
               "ut_fp16_x2_scale_must_be_none_950 / ut_fp16_x1_scale_must_be_none_950；"
               "op_api TestAllGatherCheckScale 系列", "kept"),
    _l2("L2-203", "x1/x2 dtype 不一致", "x1=fp16, x2=bf16", "拒绝", "C",
        "TestDtypeMismatchX1X2", "kept"),
    _l2("L2-204", "bias dtype 不一致", "x1=fp16, bias=bf16", "拒绝", "C",
        "TestBiasDtypeMismatch", "kept"),
    _l2("L2-205", "k=255 拒绝", "k=255（< 256 下界）", "拒绝", "C",
        "TestKAxisTooSmall", "kept"),
    _l2("L2-206", "k=65535 拒绝", "k=65535（≥ 65535 上界）", "拒绝", "C",
        "TestKAxisTooLarge", "kept"),
    _l2("L2-207", "k 轴不匹配", "x1.k=768, x2.k=512", "拒绝", "C",
        "TestKAxisMismatch", "kept"),
    _l2("L2-208", "n 轴不匹配", "bias.n≠n / output.n≠n", "拒绝", "B+C",
        "TestNAxisMismatch；ut_bias_dim0_mismatch_fp16_950", "kept"),
    _l2("L2-209", "x1 转置 view 拒绝", "x1 以 (k,m) 转置 view 传入", "拒绝",
        "C", "TestX1TransposedRejected", "kept"),
    _l2("L2-210", "streamMode≠1 拒绝", "streamMode=2", "拒绝", "C",
        "TestStreamModeInvalid", "kept"),
    _l2("L2-211", "commMode 非法值", "commMode=\"foo\"", "拒绝", "C",
        "TestCommModeInvalid", "kept"),
    _l2("L2-212", "commMode 空串（两通道行为一致均拒绝，T8 闭环）",
        "commMode=\"\"", "ACLNN_ERR_PARAM_INVALID（op_api "
        "CheckAndHandleCommMode 空串不匹配 ccu/ai_cpu → 拒绝；tiling "
        "AnalyzeAttrs 同样拒绝）", "B+C",
        "TestCommModeEmptyRejected（原 TestCommModeEmptyAutoDetect，随 T8 "
        "重命名并加注释，名称与语义对齐——既有用例修正，非新增）；"
        "ut_comm_mode_empty_auto_ccu_fp16_r8_950",
        "kept",
        note="原用例名误导（断言实为 EXPECT_EQ 拒绝），T8 闭环后统一期望为拒绝"),
    _l2("L2-213", "非 transB 非连续 x2（950 CCU）",
        "x2 (k,n) shape 但 stride 非行主序且非转置", "接口层不拒绝"
        "（非 ACLNN_ERR_PARAM_INVALID；mock 无卡实测返回 361001=nnopbase "
        "context 依赖、套件放行型用例同此；数值行为按 stride 由 kernel 解释，"
        "端到端待有卡）", "C",
        "TestCCUNonContiguousX2WithoutTranspose（T-A5 落地，op_api UT）", "added",
        note="期望已按 2.2 V2-B4 冻结（源码实证：拒绝校验仅 DAV_2201/AIV 分支"
             " L634-640；950/CCU 无显式校验）+ mock 平台实测回填（实测 361001，"
             "根因=nnopbase NnopbaseCreateExecutorSpace→aclrtCtxGetFloatOverflowAddr"
             " 无卡 context 依赖，与 strided x2 无关）；冻结 ≤ CP3；stride 语义"
             "端到端验证随 B9-⑤ 有卡待办"),
    _l2("L2-214", "非 2 维拒绝", "x1=(8,8,256) 3 维", "拒绝", "C",
        "TestX1ThreeDimRejected（本次新增）", "added"),
    _l2("L2-215", "output shape 不符", "output=(m,n) 而非 (m*R,n)", "拒绝",
        "B", "ut_output_y_dim0_mismatch_fp16_950", "kept"),
    _l2("L2-216", "gatherOut shape 不符", "gatherOut dim0/dim1 与 (m*R,k) 不符",
        "拒绝", "B",
        "ut_gather_out_dim0_mismatch_950 / ut_gather_out_dim1_k_mismatch_950",
        "kept"),
    _l2("L2-217", "bias 形状非 (n,)", "bias=(n-1,)", "拒绝", "B",
        "ut_bias_dim0_mismatch_fp16_950", "kept"),
    _l2("L2-218", "rank_size=3 拒绝", "rank_size=3（不在 {2,4,8,16,32,64}）",
        "拒绝", "B", "ut_rank_size_3_unsupported_fp16_950", "kept"),
    _l2("L2-219", "x2 空 tensor（k≠0）", "x2=(0,n) 而 x1.k≠0", "拒绝"
        "（shape 不匹配）", "C", "TestEmptyX2WithNonZeroK（本次新增）", "added",
        note="现有空 x2 例为 FP8 量化场景，非量化缺"),
    _l2("L2-220", "gatherIndex≠0 拒绝", "gather_index=1",
        "拒绝（tiling AnalyzeAttrs 强校验 gather_index==0）", "B",
        "ut_gather_index_nonzero_should_fail_fp16_950（本次新增 csv 行）",
        "added", note="原 csv 全量 122 行无 gather_index≠0 行——2.1 §5.3 实证缺"),
    _l2("L2-221", "commTurn≠0 拒绝", "comm_turn=1",
        "拒绝（tiling AnalyzeAttrs 强校验 comm_turn==0）", "B",
        "ut_comm_turn_must_be_zero_fp16_950", "kept"),
    _l2("L2-222", "blockSize≠0 拒绝（非量化）", "block_size=1",
        "拒绝（非量化 blockSize 必须为 0）", "B",
        "ut_block_size_nonzero_should_fail_fp16_950", "kept"),
]

# ---------------------------------------------------------------------------
# 汇总访问接口
# ---------------------------------------------------------------------------

ALL_FUNCTIONAL_CASES = L0_CASES + L1_CASES
ALL_L2_CASES = L2_CASES

# 2.1 §9 汇总：L0=4 / L1=31（含 1 条件例）/ L2=22 / ST 镜像=28
assert len(L0_CASES) == 4, "L0 case count must be 4 (2.1 §5.1)"
assert len(L1_CASES) == 31, "L1 case count must be 31 (2.1 §5.2)"
assert len(L2_CASES) == 22, "L2 case count must be 22 (2.1 §5.3)"


def get_case(cid):
    for c in ALL_FUNCTIONAL_CASES:
        if c["id"] == cid:
            return c
    raise KeyError(f"unknown functional case id: {cid}")


def cases_by_level(level):
    if level == "L0":
        return L0_CASES
    if level == "L1":
        return L1_CASES
    if level == "L2":
        return L2_CASES
    if level == "all":
        return ALL_FUNCTIONAL_CASES
    raise ValueError(f"unknown level: {level}")


def env_lines(case, dump_dir, seed=20260919):
    """Render one case as example env-var lines (interface contract with the
    parameterized example; see tests/sim/README.md §dump 契约)."""
    return [
        f"AGM_V2_CASE_ID={case['id']}",
        f"AGM_V2_DUMP_DIR={dump_dir}",
        f"AGM_V2_DTYPE={case['dtype']}",
        f"AGM_V2_M={case['m']}",
        f"AGM_V2_K={case['k']}",
        f"AGM_V2_N={case['n']}",
        f"AGM_V2_RANK_SIZE={case['rank_size']}",
        f"AGM_V2_COMM_MODE={case['comm_mode']}",
        f"AGM_V2_BIAS={1 if case['bias'] else 0}",
        f"AGM_V2_TRANS_B={1 if case['trans_b'] else 0}",
        f"AGM_V2_GATHER_OUT={1 if case['gather_out'] else 0}",
        f"AGM_V2_VALUE_SPEC={case['value_spec']}",
        f"AGM_V2_REPEAT_LAUNCH={case['repeat_launch']}",
        f"AGM_V2_SEED={seed + int(case['id'].split('-')[-1])}",
    ]


if __name__ == "__main__":
    import json
    import sys

    level = sys.argv[1] if len(sys.argv) > 1 else "all"
    as_env = "--env" in sys.argv
    include_conditional = "--include-conditional" in sys.argv
    dump_root = "."
    for arg in sys.argv:
        if arg.startswith("--dump-root="):
            dump_root = arg.split("=", 1)[1]
    cases = cases_by_level(level)
    if as_env:
        for case in cases:
            if case.get("conditional") and not include_conditional:
                continue
            dump_dir = f"{dump_root}/{case['id']}"
            print("\n".join(env_lines(case, dump_dir)))
            print("---")
    else:
        print(json.dumps({"L0": len(L0_CASES), "L1": len(L1_CASES),
                          "L2": len(L2_CASES),
                          "st_mirror": 4 + 8 + 5 + 4 + 4 + 3},
                         indent=1))
