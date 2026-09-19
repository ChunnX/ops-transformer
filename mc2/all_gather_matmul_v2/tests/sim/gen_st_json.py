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

每个用例带 remark 标注 is_trans_b/is_bias/gather_output 供共享 golden 解析
（golden_utils.golden_all_gather_compare 的 remark 文法）；bias dtype 与 x1
一致、无 scale（x1Scale/x2Scale/quantScale 声明为占位——executor 对非 int8
输入会将其实参置 null，golden 非量化路径不消费 scale）。

已知依赖（有卡执行前须解决，见 tests/sim/README.md §ST 依赖）：
  executor_aclnnAllGatherMatmulV2.py 当前无条件将 bias 实参置 null
  （input_args[2]），且未构造 transB 非连续 view 与 gatherOut 空 tensor——
  bias/transB/gather_out=false 用例需 executor 非量化扩展后方可执行。

Usage:
  python3 gen_st_json.py [--out FILE]
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


def _mirror_ids():
    """ST 镜像构成（2.1 §7 T4）。"""
    ids = [c["id"] for c in sim_cases.L0_CASES]
    ids += [c["id"] for c in sim_cases.L1_CASES
            if c["id"].startswith(("L1-10", "L1-11", "L1-12", "L1-13"))
            and c["id"] not in ("L1-142",)]
    # S1(101-108) + S2(110-114) + S3(120-123) + S4(130-133) 已含于上式；
    # 排除 S5(140-142)/S6(150-152)/S8(170)，补 S7(160-162)
    ids += [c["id"] for c in sim_cases.L1_CASES if c["id"].startswith("L1-16")]
    return ids


def _tensor(name, dtype, shape, range_values):
    return {
        "name": name, "type": "tensor", "required": True,
        "dtype": dtype, "shape": shape, "range_values": range_values,
        "backward": True, "align_32B": None, "outlier_values": None,
    }


def _attr(name, dtype, range_values):
    return {
        "name": name, "type": "attr", "required": True,
        "dtype": dtype, "shape": None, "range_values": range_values,
        "backward": False, "align_32B": None, "outlier_values": None,
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

    inputs = [
        _tensor("x1", dtype, [m, k], [-5, 5]),
        _tensor("x2", dtype, x2_shape, [-5, 5]),
        # bias：有用例声明真实值域；无用例声明 0 值域（executor 将 bias
        # 实参置 null，golden 侧加 0 —— 两侧均为 no-op，保持一致）
        _tensor("bias", dtype, [max(n, 1)],
                [-5, 5] if has_bias else [0]),
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
        _attr("outputDtype", "torch_dtype",
              _TORCH_DTYPE_STR[case["dtype"]]),
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

    st_case = {
        "id": idx,
        "name": f"allgather_matmul_v2_{case['id'].lower()}",
        "aclnn_name": "AllGatherMatmulV2",
        "api_type": "function_allgather_matmul_v2",
        "aclnn_api_type": "aclnn_allgather_matmul_v2",
        "dist_api_type": "function_allgather_matmul_v2",
        "standard": {"acc": "cv_fused_double_benchmark",
                     "perf": [0.95, 0.95]},
        "remark": ",".join(remark_parts),
        "inputs": inputs,
    }
    return st_case


def main():
    default_out = (SIM_DIR.parent / "st" / "aclnnAllGatherMatmulV2"
                   / "arch35_atk_aclnnAllGatherMatmulV2.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(default_out))
    args = parser.parse_args()

    ids = _mirror_ids()
    assert len(ids) == 28, (
        f"ST mirror must be 28 cases per 2.1 §7 T4, got {len(ids)}: {ids}")
    by_id = {c["id"]: c for c in sim_cases.ALL_FUNCTIONAL_CASES}
    st_cases = [case_to_st(i, by_id[cid]) for i, cid in enumerate(ids)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(st_cases, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[gen_st_json] wrote {len(st_cases)} cases -> {out}")
    print(f"[gen_st_json] mirror ids: {','.join(ids)}")


if __name__ == "__main__":
    main()
