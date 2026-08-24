#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""由 kernel ST 用例现场生成 E2E（torch）用例 CSV。

E2E CSV 不入库（与 attention/chunk_gated_delta_rule 的 convert_rdv_to_csv.py 同惯例），
用例的唯一真值是同目录 ../st/arch35 的 kernel CSV，避免两份用例各自漂移。

    python3 make_e2e_csv.py [-o <输出路径>]
    python3 -m ttk e2e -i <输出路径> --plugin <此目录的 golden.py>

字段映射（E2E 表头比 aclnn 少 output_tensor_indexes，多 inplace_input_indexes）：
    input_shapes  -> tensor_view_shapes      12 个输入张量，输出不进表
    input_dtypes  -> tensor_dtypes
    attributes / input_data_ranges           原样透传
    inplace_input_indexes = (4, 5)           k_cache / v_cache 由算子原地写回；
                                             框架按下标升序把它们追加到 return 值
                                             之后，于是输出序为 (q, k_cache, v_cache)
kernel CSV 里 precision_tolerances / absolute_precision 两列只服务 isclose 判据，
E2E 这边判据由 spec 自带（cross_check/L1），故不透传。
"""

import argparse
import csv
from pathlib import Path

API_NAME = "cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache"
INPLACE_INPUT_INDEXES = "(4, 5)"
KERNEL_CSV = Path(__file__).resolve().parents[1] / "st" / "arch35" / \
    "ttk_kernel_und_gen_qkv_rms_norm_rope_cache_st.csv"
E2E_HEADER = ["testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
              "attributes", "input_data_ranges", "inplace_input_indexes"]


def convert(kernel_csv: Path, out_csv: Path) -> int:
    with open(kernel_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"kernel CSV 没有用例：{kernel_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=E2E_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "testcase_name": row["testcase_name"],
                "api_name": API_NAME,
                "tensor_view_shapes": row["input_shapes"],
                "tensor_dtypes": row["input_dtypes"],
                "attributes": row["attributes"],
                "input_data_ranges": row["input_data_ranges"],
                "inplace_input_indexes": INPLACE_INPUT_INDEXES,
            })
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=Path, default=KERNEL_CSV, help="kernel ST CSV")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("ttk_e2e_und_gen_qkv_rms_norm_rope_cache_st.csv"),
                        help="生成的 E2E CSV")
    args = parser.parse_args()
    count = convert(args.input, args.output)
    print(f"{count} 条用例 -> {args.output}")


if __name__ == "__main__":
    main()
