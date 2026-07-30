#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
  python3 gen_csv_case_store.py [--paramset qfa_mxfp4_paramset_func_rdv]
                                [--output qfa_mxfp4.csv]

Tensor 顺序与 wrapper 函数签名一致 (q, k, v, q_descale, k_descale, v_descale, block_table).
MXFP4:
  - q/k/v: uint8 (FP4 packed, last dim = D // 2)
  - q_descale/k_descale/v_descale: uint8 (E8M0 scale)
  - block_table: int32 (PA 模式才需要, 当前固定 (0,))
"""

import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

API_NAME = "qfa_mxfp4_wrapper.npu_qfa_mxfp4"


def _compute_tensor_shapes(case):
    """根据 case 字段计算 7 个 tensor 的 shape (TTK 占位 shape).

    顺序与 wrapper 签名一致: q, k, v, q_descale, k_descale, v_descale, block_table.
    TTK 用这个 shape 生成占位 tensor, 真实数据由 customize_inputs 缓存.
    所以 shape 只需大小匹配, dtype 必须正确.
    """
    B = case["B"]
    N_q = case["N_q"]
    N_kv = case["N_kv"]
    G = case["G"]
    D = case["D"]
    V_D = case["V_D"] if case.get("V_D") is not None else D
    s1 = max(case["act_seq_lens_q"])
    s2 = max(case["act_seq_lens_kv"])
    n2 = N_kv
    input_layout = case["input_layout"]

    # MXFP4 packed: last dim = D // 2 (q/k), V_D // 2 (v)
    qk_d_packed = D // 2
    v_d_packed = V_D // 2

    # scale 维度: 每 32 个 element 一组 (E8M0 1 byte)
    # q_descale: 每 D 维度 ceil(D/32) 个 scale, 但 MXFP4 还要做 (D//2, 2) packing
    #             所以 descale 末维 = D (即 ceil(D/32) * 32, 与 packed D 对齐)
    # 简化: descale shape 与 mxfp8 类似, 用 packed 后的 D 维度
    # 实际 descale shape 取决于 layout, 这里给一个占位 shape (TTK 会替换)
    q_descale_d = D  # q scale 沿 D 维量化, 每组 32 个 -> ceil(D/32) 个 scale, packed 后还原成 D 维
    k_descale_d = D
    v_descale_s = math.ceil(s2 / 32)   # v scale 沿 S 维量化
    v_descale_d = V_D

    if input_layout == "BNSD":
        q_shape = (B, N_q, s1, qk_d_packed)
        k_shape = (B, N_kv, s2, qk_d_packed)
        v_shape = (B, N_kv, s2, v_d_packed)
        q_descale_shape = (B, N_q, s1, q_descale_d // 2, 2)  # transpose_qscale 后 BNSD 形式
        k_descale_shape = (B, N_kv, s2, k_descale_d // 2, 2)
        v_descale_shape = (B, N_kv, v_descale_s, v_descale_d, 2)
    elif input_layout == "BSND":
        q_shape = (B, s1, N_q, qk_d_packed)
        k_shape = (B, s2, N_kv, qk_d_packed)
        v_shape = (B, s2, N_kv, v_d_packed)
        # transpose_qscale 后 BSND 输出是 (n2, B, S, G, D//2, 2)
        G_size = N_q // n2
        q_descale_shape = (n2, B, s1, G_size, q_descale_d // 2, 2)
        k_descale_shape = (B, s2, N_kv, k_descale_d // 2, 2)
        v_descale_shape = (B, v_descale_s, N_kv, v_descale_d, 2)
    else:  # TND 等其他 layout 暂用 BNSD 占位
        q_shape = (B, N_q, s1, qk_d_packed)
        k_shape = (B, N_kv, s2, qk_d_packed)
        v_shape = (B, N_kv, s2, v_d_packed)
        q_descale_shape = (B, N_q, s1, q_descale_d // 2, 2)
        k_descale_shape = (B, N_kv, s2, k_descale_d // 2, 2)
        v_descale_shape = (B, N_kv, v_descale_s, v_descale_d, 2)

    # block_table: PA 模式才需要, 当前固定 (0,)
    block_table_shape = (0,) if case.get("kv_storage_mode", "continue") == "continue" else (B, 0)

    return [q_shape, k_shape, v_shape,
            q_descale_shape, k_descale_shape, v_descale_shape,
            block_table_shape]


def case_to_csv_row(case):
    """把一个 case dict 转成 CSV 行."""
    name = case["name"]
    shapes = _compute_tensor_shapes(case)
    tensor_view_shapes = "(" + ",".join(f"({s})" for s in shapes) + ")"

    # MXFP4: q/k/v 都是 uint8 (FP4 packed), descale 都是 uint8 (E8M0), block_table 是 int32
    dtypes = ["'uint8'", "'uint8'", "'uint8'",
              "'uint8'", "'uint8'", "'uint8'",
              "'int32'"]
    tensor_dtypes = "(" + ",".join(dtypes) + ")"

    # softmax_scale=None 时 repr 会输出 'None', TTK 解析时会变 Python None
    softmax_scale = case.get("softmax_scale")
    attrs = {
        "B": case["B"], "N_q": case["N_q"], "N_kv": case["N_kv"], "G": case["G"],
        "D": case["D"], "V_D": case.get("V_D") or case["D"], "Rope_D": case.get("Rope_D", 0),
        "act_seq_lens_q": list(case["act_seq_lens_q"]),
        "act_seq_lens_kv": list(case["act_seq_lens_kv"]),
        "input_layout": case["input_layout"],
        "layout_q_descale": case.get("layout_q_descale", "BSND"),
        "kv_storage_mode": case.get("kv_storage_mode", "continue"),
        "block_size": case.get("block_size", 0),
        "q_dtype": case.get("q_dtype", "fp4_e2m1"),
        "kv_dtype": case.get("kv_dtype", "fp4_e2m1"),
        "out_dtype": case.get("out_dtype", "bfloat16"),
        "q_quant_mode": case.get("q_quant_mode", 3),
        "mask_mode": case.get("mask_mode", 0),
        "pre_tokens": case.get("pre_tokens", 2147483647),
        "next_tokens": case.get("next_tokens", 2147483647),
        "enable_mask": case.get("enable_mask", False),
        "enable_lse": case.get("enable_lse", False),
        "inner_precise": case.get("inner_precise", 0),
        "device_id": case.get("device_id", 0),
        "graph_path": case.get("graph_path", 0),
        "softmax_scale": softmax_scale,
        "data_range_q": case.get("data_range_q", 1.0),
        "data_range_k": case.get("data_range_k", 1.0),
        "data_range_v": case.get("data_range_v", 1.0),
    }
    # TTK process_dict 用 eval() 解析, 需 Python dict 字面量 (单引号)
    attributes = repr(attrs)

    # wrapper 的 7 个 tensor 全是输入, 输出是函数返回值 (非 tensor 参数)
    # 所以 output_tensor_indexes 为空
    output_tensor_indexes = ""

    # 精度容差默认值 (与 spec tolerance / 原版 pytest check_result 一致)
    # wrapper 固定返回 2 个输出 (atten_out, lse_out), 各配一个 (rtol, ptol) 对 + atol
    # 当前 out_dtype=bfloat16: rtol=0.0078125, ptol=0.005, atol=0.0001
    precision_tolerances = "((0.0078125, 0.005), (0.0078125, 0.005))"
    absolute_precision = "(0.0001, 0.0001)"

    return [name, API_NAME, tensor_view_shapes, tensor_dtypes, "",
            attributes, output_tensor_indexes, "",
            precision_tolerances, absolute_precision]


def main():
    parser = argparse.ArgumentParser(description="paramset -> TTK e2e CSV")
    parser.add_argument("--paramset", default="qfa_mxfp4_paramset_func_rdv",
                        help="paramset 模块名 (不含 .py)")
    parser.add_argument("--output", default="qfa_mxfp4.csv",
                        help="输出 CSV 路径")
    args = parser.parse_args()

    paramset_mod = __import__(args.paramset)
    skip = getattr(paramset_mod, "SKIP_CASES", set())

    if hasattr(paramset_mod, "CASES") and paramset_mod.CASES is not None:
        cases = paramset_mod.CASES
    elif hasattr(paramset_mod, "TEST_PARAMS"):
        from qfa_mxfp4_paramset_common import expand_paramset_to_cases
        cases = expand_paramset_to_cases(paramset_mod.TEST_PARAMS)
    else:
        raise ValueError(f"paramset module '{args.paramset}' has neither CASES nor TEST_PARAMS")

    header = ["testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
              "tensor_formats", "attributes", "output_tensor_indexes", "golden_api",
              "precision_tolerances", "absolute_precision"]
    out_path = args.output if os.path.isabs(args.output) else os.path.join(_HERE, args.output)
    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for case in cases:
            if case["name"] in skip:
                skipped += 1
                continue
            writer.writerow(case_to_csv_row(case))
            written += 1

    print(f"生成 {written} 个 case (跳过 {skipped} 个 SKIP) -> {out_path}")


if __name__ == "__main__":
    main()
