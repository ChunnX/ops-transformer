#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""从 pytest func_rdv 全维度矩阵（638 用例）生成 ttk E2E CSV。

用法（建议在目标设备环境执行，metadata 槽形状依赖实际核数）：
    python3 gen_csv.py [-o qfa_mxfp8_sf16.csv] [--cores aic,aiv]

slot 布局（与 impl/inputs.py 一致，共 15 个）：
  0 q  1 k  2 v  3 q_descale  4 k_descale  5 v_descale  6 block_table
  7 p_scale  8 cu_seqlens_q  9 cu_seqlens_kv  10 seqused_q  11 seqused_kv
  12 sinks  13 attn_mask  14 metadata
"""

import argparse
import csv
import math
import os
import sys

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARAMSET_DIR = os.path.join(_ASSETS_DIR, "..", "pytest", "qfa_mxfp8_softmax_fp16_test")
if _PARAMSET_DIR not in sys.path:
    sys.path.insert(0, _PARAMSET_DIR)

import quant_flash_attn_paramset_func_rdv as paramset  # noqa: E402

METADATA_STRIDE = 16
ALIGN_SIZE = 4096
API_NAME = "torch.ops.cann_ops_transformer.quant_flash_attn"
SOFTMAX_SCALE_D128 = 1.0 / math.sqrt(128.0)  # pytest SOFTMAX_SCALE=None 的兜底值

# 双输出容差（atten bf16 + LSE fp32），与 spec.py tolerance / check_result 同源
PRECISION_TOLERANCES = "((0.0078125, 0.005), (0.005, 0.005))"
ABSOLUTE_PRECISION = "(0.0001, 0.000025)"


def _get_core_nums(override):
    if override:
        aic, aiv = (int(x) for x in override.split(","))
        return aic, aiv
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401

        props = torch.npu.get_device_properties()
        return int(props.cube_core_num), int(props.vector_core_num)
    except Exception:
        print(
            "[WARN] 无法获取设备核数，回退 ascend950 默认 (32, 64)；"
            "若目标设备不同请用 --cores aic,aiv 指定",
            file=sys.stderr,
        )
        return 32, 64


def _metadata_slot_shape(batch_size, num_heads_kv, aic_num, aiv_num):
    batch_size = batch_size if batch_size and batch_size > 0 else 1
    fa_size = aic_num * METADATA_STRIDE * batch_size * num_heads_kv
    fd_size = aiv_num * METADATA_STRIDE * batch_size * num_heads_kv
    schedule_size = METADATA_STRIDE + fa_size + fd_size
    return (2, ((schedule_size + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE)


def _build_row(case, aic_num, aiv_num):
    b = case["B"]
    nq = case["N_q"]
    nkv = case["N_kv"]
    d = case["D"]
    sq = max(case["seqused_q"])
    skv = max(case["seqused_kv"])
    lse = bool(case["enable_lse"])
    num_groups_qk = d // 64
    num_groups_v = (skv + 63) // 64

    shapes = (
        (b, nq, sq, d),  # 0 q
        (b, nkv, skv, d),  # 1 k
        (b, nkv, skv, d),  # 2 v
        (b, nq, sq, num_groups_qk, 2),  # 3 q_descale
        (b, nkv, skv, num_groups_qk, 2),  # 4 k_descale
        (b, nkv, num_groups_v, d, 2),  # 5 v_descale
        None,  # 6 block_table
        (1,),  # 7 p_scale
        None,  # 8 cu_seqlens_q
        None,  # 9 cu_seqlens_kv
        (b,),  # 10 seqused_q
        (b,),  # 11 seqused_kv
        None,  # 12 sinks
        None,  # 13 attn_mask
        _metadata_slot_shape(b, nkv, aic_num, aiv_num),  # 14 metadata
    )
    dtypes = (
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e8m0",
        "float8_e8m0",
        "float8_e8m0",
        None,
        "float32",
        None,
        None,
        "int32",
        "int32",
        None,
        None,
        "int32",
    )

    attrs = {
        "B": b,
        "N_q": nq,
        "N_kv": nkv,
        "G": nq // nkv,
        "D": d,
        "V_D": d,
        "Rope_D": 0,
        "seqused_q_values": list(case["seqused_q"]),
        "seqused_kv_values": list(case["seqused_kv"]),
        "cu_seqlens_q_values": [],
        "cu_seqlens_kv_values": [],
        "max_seqlen_q": -1,
        "max_seqlen_kv": -1,
        "layout_q": "BNSD",
        "layout_q_descale": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "kv_storage_mode": "continue",
        "block_size": 0,
        "q_dtype": "fp8_e4m3fn",
        "kv_dtype": "fp8_e4m3fn",
        "out_dtype": "bfloat16",
        "quant_mode": 3,
        "mask_mode": 0,
        "win_left": 2147483647,
        "win_right": 2147483647,
        "enable_mask": False,
        "return_softmax_lse": lse,
        "inner_precise": 0,
        "device_id": 0,
        "graph_path": 0,
        "softmax_scale": SOFTMAX_SCALE_D128,
        "block_table_shape": [],
        "block_table_dtype": None,
        "p_scale_value": 1.0,
        "p_scale_shape": [1],
        "p_scale_dtype": "float32",
        "p_scale_datarange": None,
        "sinks_shape": [],
        "sinks_dtype": None,
        "sinks_datarange": None,
        "attn_mask_shape": [],
        "attn_mask_dtype": None,
        "attn_mask_datarange": None,
        "q_descale_dtype": "float8_e8m0",
        "k_descale_dtype": "float8_e8m0",
        "v_descale_dtype": "float8_e8m0",
        "seqused_q_dtype": "int32",
        "seqused_kv_dtype": "int32",
        "cu_seqlens_q_dtype": None,
        "cu_seqlens_kv_dtype": None,
        "softmax_lse_dtype": "float32" if lse else None,
        "data_range_q": "-1,1",
        "data_range_k": "-1,1",
        "data_range_v": "-1,1",
    }

    return [
        case["name"],
        API_NAME,
        repr(shapes),
        repr(dtypes),
        "",
        repr(attrs),
        "",
        "",
        PRECISION_TOLERANCES,
        ABSOLUTE_PRECISION,
    ]


def main():
    parser = argparse.ArgumentParser(description="func_rdv 638 矩阵 → ttk CSV")
    parser.add_argument(
        "-o", "--output", default=os.path.join(_ASSETS_DIR, "qfa_mxfp8_sf16.csv")
    )
    parser.add_argument(
        "--cores", default=None, help="aic,aiv（缺省自动探测设备，回退 32,64）"
    )
    args = parser.parse_args()

    aic_num, aiv_num = _get_core_nums(args.cores)
    cases = paramset.CASES
    print(f"参数源: func_rdv {len(cases)} 用例；核数 aic={aic_num} aiv={aiv_num}")

    rows = [_build_row(c, aic_num, aiv_num) for c in cases]
    names = [r[0] for r in rows]
    assert len(names) == len(set(names)), "testcase_name 重复"

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "testcase_name",
                "api_name",
                "tensor_view_shapes",
                "tensor_dtypes",
                "tensor_formats",
                "attributes",
                "output_tensor_indexes",
                "golden_api",
                "precision_tolerances",
                "absolute_precision",
            ]
        )
        writer.writerows(rows)

    lse_cnt = sum(1 for c in cases if c["enable_lse"])
    print(
        f"已生成 {args.output}: {len(rows)} 用例（LSE {lse_cnt} / 非 LSE {len(rows) - lse_cnt}）"
    )


if __name__ == "__main__":
    main()
