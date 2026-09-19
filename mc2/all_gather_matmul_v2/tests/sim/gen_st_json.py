#!/usr/bin/env python3
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
"""Generate the arch35 non-quant ST case json (2.1 §7 T4) from sim_cases.

28 例 = L0(4) + S1(8) + S2(5) + S3(4) + S4(4，含 L1-133) + S7(3，含 L1-162)。
卡数为 ST 运行侧配置项（runner WS），非用例维度；S6 特殊值与 S8 重复
launch 为仿真/BI 专属，不镜像（2.1 §7 T4）。

P1 缩放镜像（2026-09-20 拍板延伸，3.4 上卡补验报告 v1.2 §10）：sim_cases
中 value_scale≠1.0 的用例（21 例 ×0.1，其中 17 例落在 ST 镜像域内）将
x1/x2/bias 的 range_values 同系数缩放（±5→±0.5），对齐 sim_cases.py 缩放
口径——输出幅值压回 float_compute.md §3 固定 abs 上限适用域；shape/dtype/
bias/transB/gatherOut 因子矩阵不变。scale 占位张量（x1Scale/x2Scale/
quantScale）不缩放：executor 对非 int8 输入将实参置 null，占位值域不产生
数据。

P6 补充镜像（2026-09-20 拍板，3.4 报告 v1.3 §11.7-P6 / v1.4）：L1-130
额外纳入 ST 侧缩放集（_ST_EXTRA_SCALED）——sim 通道 L1-130（R=2）v1.1
即 PASS（maxAbs 0.002）不在 P1 的 21 例缩放域；但 ST runner 的 WS=4
（R=4）变体输出行数×2、极值元素落入 [128,256) 幅值带（1-ULP=0.125>0.1，
§2.3 口径 FAIL、逐元素 matched=1.0、框架口径 PASS，非算子缺陷）。WS 为
ST 运行侧配置维度（同一 json 条目驱动 WS2/WS4 两次运行），json 粒度无法
按 WS 区分——按 P1 拍板口径对 L1-130 整例缩放（系数同
sim_cases.V2_VALUE_SCALE_P1，ST 镜像缩放例 17→18）；WS=2 变体同受缩放
（值域缩放对已 PASS 例无判定影响，与 21 例缩放重跑同水位）。

informational 标记机制（镜像 sim_cases INFO_CASES 机制）：默认输出 28 例
（不含 informational 变体）；--include-informational 追加 INFO_CASES（大幅
值原幅值变体，remark 增 `informational=1`、用例对象携带
`"informational": true` 字段）。ST 侧 informational 用例的判定同样不计入
通过门禁，按「ST 框架权威口径 + §2.3 容差口径」双指标记录（判定侧识别
informational 标记后剔除出门禁统计）。

每个用例带 remark 标注 is_trans_b/is_bias/gather_output 供共享 golden 解析
（golden_utils.golden_all_gather_compare 的 remark 文法）；bias dtype 与 x1
一致、无 scale（x1Scale/x2Scale/quantScale 声明为占位——executor 对非 int8
输入会将其实参置 null，golden 非量化路径不消费 scale）。

executor 非量化扩展已落地（bias 透传 / transB 视图构造 / gatherOut 空
tensor，tests/st/aclnnAllGatherMatmulV2/executor_aclnnAllGatherMatmulV2.py）；
历史依赖描述见 tests/sim/README.md §ST 依赖。

Usage:
  python3 gen_st_json.py [--out FILE] [--include-informational]
  默认输出 tests/st/aclnnAllGatherMatmulV2/arch35_atk_aclnnAllGatherMatmulV2.json
"""

import argparse
import json
import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_DIR.parent / "assets"))
import sim_cases  # noqa: E402

_ST_DTYPE = {"fp16": "fp16", "bf16": "bf16"}
_TORCH_DTYPE_STR = {"fp16": "float16", "bf16": "bfloat16"}
# P1 镜像前 ST 侧基准值域（对应 sim_cases clean 语义的 ±5 主带）
_BASE_RANGE = [-5, 5]

# P6（2026-09-20 拍板，3.4 报告 v1.3 §11.7-P6）：ST 侧补充缩放集——
# L1-130 的 WS=4（R=4）ST 变体复现 §4 既知「固定 abs 上限 × 大幅值输出域」
# 1-ULP 现象（maxAbs 0.125>0.1、matched=1.0，非缺陷），按 P1 口径纳入缩放
# 集；sim_cases 中该例 value_scale=1.0（sim 通道 R=2 域 v1.1 即 PASS，21 例
# 缩放集不含它），故在 ST 镜像侧补充定义，不动 sim_cases（sim 通道口径与
# v1.1/v1.2 登记结果保持不变）。
_ST_EXTRA_SCALED = {"L1-130"}


def _st_scale(case):
    """ST 镜像有效缩放系数：sim_cases value_scale ∪ P6 补充缩放集。"""
    scale = float(case.get("value_scale", 1.0))
    if case["id"] in _ST_EXTRA_SCALED:
        scale = float(sim_cases.V2_VALUE_SCALE_P1)
    return scale


def _mirror_ids(include_informational=False):
    """ST 镜像构成（2.1 §7 T4；informational 变体默认排除）。"""
    ids = [c["id"] for c in sim_cases.L0_CASES]
    ids += [
        c["id"]
        for c in sim_cases.L1_CASES
        if c["id"].startswith(("L1-10", "L1-11", "L1-12", "L1-13"))
        and c["id"] not in ("L1-142",)
    ]
    # S1(101-108) + S2(110-114) + S3(120-123) + S4(130-133) 已含于上式；
    # 排除 S5(140-142)/S6(150-152)/S8(170)，补 S7(160-162)
    ids += [c["id"] for c in sim_cases.L1_CASES if c["id"].startswith("L1-16")]
    if include_informational:
        ids += [c["id"] for c in sim_cases.INFO_CASES]
    return ids


def _scaled_range(range_values, scale):
    """Mirror sim_cases value_scale onto json range_values（P1 处置镜像）。

    仅缩放数值型 [lo, hi] 值域；非数值端点（"-inf"/"inf" 等）原样返回。
    """
    if scale == 1.0:
        return range_values
    if (
        isinstance(range_values, list)
        and len(range_values) == 2
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in range_values
        )
    ):
        return [round(range_values[0] * scale, 10), round(range_values[1] * scale, 10)]
    return range_values


def _tensor(name, dtype, shape, range_values):
    return {
        "name": name,
        "type": "tensor",
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": range_values,
        "backward": True,
        "align_32B": None,
        "outlier_values": None,
    }


def _attr(name, dtype, range_values):
    return {
        "name": name,
        "type": "attr",
        "required": True,
        "dtype": dtype,
        "shape": None,
        "range_values": range_values,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def case_to_st(idx, case):
    dtype = _ST_DTYPE[case["dtype"]]
    m, k, n = case["m"], case["k"], case["n"]
    trans_b = bool(case["trans_b"])
    has_bias = bool(case["bias"])
    gather_out = bool(case["gather_out"])
    # transB：x2 以 (n,k) API view 形态声明（与既有 json 约定一致，
    # executor 的 CPU 参考侧按 x1.k==x2.shape[1] 判定并 transpose）
    x2_shape = [n, k] if trans_b else [k, n]

    # P1 缩放镜像：value_scale 同系数作用于 x1/x2/bias 值域（±5→±0.5）；
    # P6 补充：L1-130 经 _st_scale 纳入（见 _ST_EXTRA_SCALED 注释）；
    # scale 占位张量不缩放（非 int8 实参置 null，不产生数据）
    scale = _st_scale(case)
    data_range = _scaled_range(list(_BASE_RANGE), scale)
    bias_range = _scaled_range(list(_BASE_RANGE), scale) if has_bias else [0]

    inputs = [
        _tensor("x1", dtype, [m, k], data_range),
        _tensor("x2", dtype, x2_shape, data_range),
        # bias：有用例声明真实值域；无用例声明 0 值域（executor 将 bias
        # 实参置 null，golden 侧加 0 —— 两侧均为 no-op，保持一致）
        _tensor("bias", dtype, [max(n, 1)], bias_range),
        # scale 占位：非量化实参必须 nullptr（executor 对非 int8 置 null）
        _tensor("x1Scale", "fp32", [1], [-5, 5]),
        _tensor("x2Scale", "fp32", [1], [-5, 5]),
        _tensor("quantScale", "fp32", [1], ["-inf", "inf"]),
        _attr("blockSize", "int64_t", 0),
        _attr("group", "string", [-7, 7]),
        _attr("gatherIndex", "int64_t", 0),
        _attr("commTurn", "int64_t", 0),
        _attr("streamMode", "int64_t", 1),
        _attr("groupSize", "int64_t", 0),
        _attr("commMode", "string", case["comm_mode"]),
        _attr("outputDtype", "torch_dtype", _TORCH_DTYPE_STR[case["dtype"]]),
        _attr("isFormatNz", "bool", False),
        _attr("useX1Scale", "bool", False),
        _attr("useInt64", "bool", False),
    ]

    remark_parts = []
    if trans_b:
        remark_parts.append("is_trans_b=1")
    if has_bias:
        remark_parts.append("is_bias=1")
    remark_parts.append(f"gather_output={1 if gather_out else 0}")
    remark_parts.append(f"plan_case_id={case['id']}")
    if case.get("informational"):
        remark_parts.append("informational=1")

    st_case = {
        "id": idx,
        "name": f"allgather_matmul_v2_{case['id'].lower()}",
        "aclnn_name": "AllGatherMatmulV2",
        "api_type": "function_allgather_matmul_v2",
        "aclnn_api_type": "aclnn_allgather_matmul_v2",
        "dist_api_type": "function_allgather_matmul_v2",
        "standard": {"acc": "cv_fused_double_benchmark", "perf": [0.95, 0.95]},
        "remark": ",".join(remark_parts),
        "inputs": inputs,
    }
    # informational 标记：仅 informational 用例携带该字段（默认 28 例不含），
    # 判定侧识别后不计入门禁，按双指标记录
    if case.get("informational"):
        st_case["informational"] = True
    return st_case


def main():
    default_out = (
        SIM_DIR.parent
        / "st"
        / "aclnnAllGatherMatmulV2"
        / "arch35_atk_aclnnAllGatherMatmulV2.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument(
        "--include-informational",
        action="store_true",
        help="追加 INFO_CASES（informational 用例，判定不计"
        "门禁，双指标记录；默认排除，输出 28 例）",
    )
    args = parser.parse_args()

    ids = _mirror_ids(include_informational=args.include_informational)
    expected = 31 if args.include_informational else 28
    assert len(ids) == expected, (
        f"ST mirror must be {expected} cases per 2.1 §7 T4 "
        f"(include_informational={args.include_informational}), "
        f"got {len(ids)}: {ids}"
    )
    by_id = {c["id"]: c for c in sim_cases.ALL_FUNCTIONAL_CASES}
    st_cases = [case_to_st(i, by_id[cid]) for i, cid in enumerate(ids)]
    scaled = [cid for cid in ids if _st_scale(by_id[cid]) != 1.0]
    info = [cid for cid in ids if by_id[cid].get("informational")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(st_cases, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[gen_st_json] wrote {len(st_cases)} cases -> {out}")
    print(f"[gen_st_json] mirror ids: {','.join(ids)}")
    print(f"[gen_st_json] P1+P6 scaled range cases ({len(scaled)}): {','.join(scaled)}")
    print(f"[gen_st_json] informational cases ({len(info)}): {','.join(info)}")


if __name__ == "__main__":
    main()
