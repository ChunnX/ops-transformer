#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""把 fia_fullquant_mxfp8_paramset_func_rdv.py 的 paramset 转成 TTK e2e CSV case store。

用法:
  python3 gen_csv_case_store.py [--paramset fia_fullquant_mxfp8_paramset_func_rdv]
                                [--output fia_mxfp8.csv]
"""

import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

API_NAME = "fia_mxfp8_wrapper.npu_fia_mxfp8"


def case_to_csv_row(case):
    """把一个 case dict 转成 CSV 行(list,由 csv.writer 负责转义)。"""
    name = case["name"]
    B = case["B"]
    N_q = case["N_q"]
    N_kv = case["N_kv"]
    D = case["D"]
    seq_q = list(case["actual_seq_q"])
    seq_kv = list(case["actual_seq_kv"])
    max_sq = max(seq_q) if seq_q else D
    max_skv = max(seq_kv) if seq_kv else D
    enable_pa = case["enable_pa"]
    block_size = case["block_size"]
    enable_lse = case["enable_lse"]

    num_groups = math.ceil(D / 32)
    max_blocks = max(math.ceil(s / block_size) for s in seq_kv) if seq_kv and enable_pa else 0

    shapes = [
        f"({B},{N_q},{max_sq},{D})",
        f"({B},{N_kv},{max_skv},{D})",
        f"({B},{N_kv},{max_skv},{D})",
        f"({B},{N_q},{max_sq},{num_groups})",
        f"({B},{N_kv},{max_skv},{num_groups})",
        f"({B},{N_kv},{math.ceil(max_skv/32)},{D})",
        "(1,)",
        f"({B},{max_blocks})" if enable_pa and max_blocks > 0 else "(0,)",
    ]
    tensor_view_shapes = "(" + ",".join(f"({s})" for s in shapes) + ")"

    dtypes = ["'float16'", "'float16'", "'float16'",
              "'float32'", "'float32'", "'float32'",
              "'float32'", "'int32'"]
    tensor_dtypes = "(" + ",".join(dtypes) + ")"

    attrs = {
        "B": B, "N_q": N_q, "N_kv": N_kv, "D": D,
        "actual_seq_q": seq_q, "actual_seq_kv": seq_kv,
        "enable_pa": enable_pa,
        "kv_cache_layout": case["kv_cache_layout"],
        "block_size": block_size,
        "sparse_mode": case["sparse_mode"],
        "q_scale_layout": case["q_scale_layout"],
        "p_scale_value": case["p_scale"],
        "data_range_q": case["data_range_q"],
        "data_range_k": case["data_range_k"],
        "data_range_v": case["data_range_v"],
        "enable_lse": enable_lse,
        "enable_rope": case.get("enable_rope", False),
        "D_rope": case.get("D_rope", 64),
        "graph_path": case.get("graph_path", 0),
        "input_layout": "TND",
        "is_contiguous": case.get("is_contiguous", True),
        "device_id": case.get("device_id", 0),
    }
    # TTK process_dict 用 eval() 解析,需 Python dict 字面量(单引号)
    attributes = repr(attrs)

    output_tensor_indexes = "0,1" if enable_lse else "0"

    return [name, API_NAME, tensor_view_shapes, tensor_dtypes, "",
            attributes, output_tensor_indexes, ""]


def main():
    parser = argparse.ArgumentParser(description="paramset → TTK e2e CSV")
    parser.add_argument("--paramset", default="fia_fullquant_mxfp8_paramset_func_rdv",
                        help="paramset 模块名(不含.py)")
    parser.add_argument("--output", default="fia_mxfp8.csv",
                        help="输出 CSV 路径")
    args = parser.parse_args()

    paramset_mod = __import__(args.paramset)
    cases = paramset_mod.CASES
    skip = getattr(paramset_mod, "SKIP_CASES", set())

    header = ["testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes",
              "tensor_formats", "attributes", "output_tensor_indexes", "golden_api"]
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

    print(f"生成 {written} 个 case (跳过 {skipped} 个 SKIP) → {out_path}")


if __name__ == "__main__":
    main()
