#!/usr/bin/python
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

"""Convert pytests TestCases dict → TTK E2E CSV（MLA R1 路线）。

与 flash_attn 的 tests/assets/gen_cases.py 同接口/同产物格式，按 MLA(m0070/R1)
适配：布局限定 q=TND / kv=PA_NZ / out=NTD，q/k_cache 最后维 576
（nope512 + rope64，MLA_D_TOTAL），kv head num 恒 1。

Usage:
    # 默认: 读取 functional_stc 用例, 输出到 testcase/flash_mla_with_kvcache_stc.csv
    python3 gen_cases.py

    # 指定输出路径 (相对 assets/ 目录解析, 也支持绝对路径)
    python3 gen_cases.py -o /tmp/result.csv

    # 指定用例模块 (对应 pytests/test_cases/<module>.py, 默认 functional_stc)
    python3 gen_cases.py --case-file functional_stc -o /tmp/result.csv

Tensor order in CSV (matches flash_mla_with_kvcache_ttk signature keyword-only order):
    [0] q              [3] cu_seqlens_q   [6] attn_mask
    [1] k_cache        [4] cache_seqlens
    [2] block_table    [5] seqused_q

Unused positions use None. Small integer tensors (cu_seqlens_q, cache_seqlens,
seqused_q) get their values from attributes dict → override_tensors_from_attributes;
block_table is filled by impl/inputs.py customize_inputs.
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTESTS = os.path.join(_HERE, "..", "pytests")
if _PYTESTS not in sys.path:
    sys.path.insert(0, _PYTESTS)

API_NAME = "flash_mla_with_kvcache_ttk_ops.flash_mla_with_kvcache_ttk"

HEADER = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "golden_api",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
]

DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16"}
INT_DTYPE = "int32"
ABS_PRECISION_DEFAULT = 1e-8

# MLA R1: q/k_cache 576 宽（nope512 + rope64）；NZ 块 = 16 元素
MLA_D_TOTAL = 576
MLA_NZ_BLK_ELEM = 16


def _q_shape(p):
    """layout_q=TND：q = [total_tokens, N1, 576]。"""
    cu_q = p.get("cu_seqlens_q")
    total = int(cu_q[-1]) if cu_q else int(p.get("S1", 1))
    n1 = int(p.get("N1", 1))
    return (total, n1, MLA_D_TOTAL)


def _k_cache_shape(p):
    """layout_kv=PA_NZ：k_cache = [num_blocks, 1, 36, block_size, 16]。"""
    num_blocks = int(p.get("num_blocks", 1))
    bs = int(p.get("block_size", 128))
    d_nz_sub = int(p.get("D_nz_sub", MLA_D_TOTAL // MLA_NZ_BLK_ELEM))
    return (num_blocks, 1, d_nz_sub, bs, MLA_NZ_BLK_ELEM)


def _build_row(case_name, p):
    """构造 TTK CSV 单行（MLA 张量序 + attrs）。"""
    cu_q = p.get("cu_seqlens_q")
    cache_seqlens = p.get("cache_seqlens")
    seqused_q = p.get("seqused_q")

    shapes = [
        _q_shape(p),
        _k_cache_shape(p),
        (int(p.get("B", 1)), int(p.get("max_blocks", 1)))
        if p.get("max_blocks")
        else None,
        (len(cu_q),) if cu_q else None,
        (len(cache_seqlens),) if cache_seqlens else None,
        (len(seqused_q),) if seqused_q else None,
        (2048, 2048) if int(p.get("mask_mode", 0)) in (3, 4) else None,
    ]

    dtype_str = DTYPE_MAP.get(p.get("Dtype", "fp16"), "float16")
    dtypes = [dtype_str, dtype_str]
    for i in range(2, len(shapes)):
        if shapes[i] is None:
            dtypes.append(None)
        elif i == 6:
            dtypes.append("int8")
        else:
            dtypes.append(INT_DTYPE)

    attrs = {
        "softmax_scale": float(p.get("scale", 512.0**-0.5)),
        "mask_mode": int(p.get("mask_mode", 0)),
        "layout_q": p.get("layout_q", "TND"),
        "layout_kv": p.get("layout_kv", "PA_NZ"),
        "layout_out": p.get("layout_out", "NTD"),
        "batch_size": int(p.get("B", 1)),
        "head_dim_v": int(p.get("D", 512)),
        "return_softmax_lse": bool(p.get("return_softmax_lse", False)),
        "block_size": int(p.get("block_size", 128)),
    }
    for key, val in (
        ("cu_seqlens_q", cu_q),
        ("cache_seqlens", cache_seqlens),
        ("seqused_q", seqused_q),
    ):
        if val is not None:
            attrs[f"{key}_values"] = list(val)

    data_ranges = [
        tuple(p.get("q_range", (-5.0, 5.0))),
        tuple(p.get("k_range", (-5.0, 5.0))),
    ]
    data_ranges += [(None, None)] * (len(shapes) - 2)

    prec = ((0.0078125, 0.0001),) if p.get("Dtype") == "bf16" else ((0.005, 0.000025),)

    return [
        case_name,
        API_NAME,
        repr(tuple(shapes)),
        repr(tuple(dtypes)),
        "",
        repr(attrs),
        "",
        "",
        repr(tuple(data_ranges)),
        repr(prec),
        repr(ABS_PRECISION_DEFAULT),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Convert pytests TestCases → TTK E2E CSV (MLA R1)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="testcase/flash_mla_with_kvcache_stc.csv",
        help="Output CSV path relative to assets (default: "
        "testcase/flash_mla_with_kvcache_stc.csv)",
    )
    parser.add_argument(
        "--case-file",
        default="functional_stc",
        help="Test case module (default: functional_stc)",
    )
    args = parser.parse_args()

    from core.case_loader import load_case_modules, normalize_params

    cases = load_case_modules([f"test_cases.{args.case_file}"])

    out_path = os.path.join(_HERE, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for case_id, raw_params in sorted(cases.items()):
            short_name = case_id.rsplit("/", 1)[-1]
            p = normalize_params(raw_params)
            writer.writerow(_build_row(short_name, p))
            written += 1

    print(f"wrote {out_path} ({written} cases)")


if __name__ == "__main__":
    main()
