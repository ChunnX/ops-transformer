#!/usr/bin/env python3
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

"""Convert enabled QBSA MXFP8 Python pytest cases to a TTK E2E CSV."""

import argparse
import csv
import io
import math
import runpy
import warnings
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = CURRENT_DIR.parents[1] / "pytest" / "bsa_fullquant_mxfp8_test"
PYTHON_CASES = LEGACY_DIR / "qbsa_mxfp8_test_cases.py"
DEFAULT_OUTPUT = CURRENT_DIR / "qbsa_mxfp8_cases.csv"

API_NAME = "bsa_ttk_ops.quant_block_sparse_attn"
FIELDNAMES = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
)
TENSOR_FORMATS = ("ND",) * 16
PRECISION_TOLERANCES = ((0.0078125, 0.0001), (0.005, 0.000025))


def _prefix_sums(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return values


def _normalize_range(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"data range must contain [min, max], got {value!r}")
        low, high = float(value[0]), float(value[1])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"invalid data range {value!r}")
        return low, high
    radius = abs(float(value))
    if not math.isfinite(radius):
        raise ValueError(f"invalid data range {value!r}")
    return -radius, radius


def build_case_row(
    *, testcase_name, batch_size, num_heads_q, num_heads_kv,
    head_dim, q_lengths, kv_lengths, blocksize, blocknum, max_block_per_batch,
    sparse_block_size_q, sparse_block_size_kv, mask_mode, p_scale_value,
    softmax_scale, layout_q, layout_kv, layout_out, layout_sparse_indices,
    return_softmax_lse, sparse_mode, data_range_q, data_range_k, data_range_v,
):
    quant_mode = 2
    batch_size = int(batch_size)
    num_heads_q = int(num_heads_q)
    num_heads_kv = int(num_heads_kv)
    head_dim = int(head_dim)
    blocksize = int(blocksize)
    blocknum = int(blocknum)
    max_block_per_batch = int(max_block_per_batch)
    sparse_block_size_q = int(sparse_block_size_q)
    sparse_block_size_kv = int(sparse_block_size_kv)
    q_lengths = [int(value) for value in q_lengths]
    kv_lengths = [int(value) for value in kv_lengths]
    if len(q_lengths) != batch_size or len(kv_lengths) != batch_size:
        raise ValueError(f"{testcase_name}: sequence length count must equal B")
    if any(value < 0 for value in q_lengths + kv_lengths):
        raise ValueError(f"{testcase_name}: sequence lengths must be non-negative")
    if num_heads_q % num_heads_kv != 0 or num_heads_q // num_heads_kv > 16:
        raise ValueError(f"{testcase_name}: N1/N2 must be an integer in [1, 16]")
    if blocksize <= 0 or blocksize > 1024 or blocksize % sparse_block_size_kv != 0:
        raise ValueError(f"{testcase_name}: invalid PA/sparse block sizes")
    if blocknum <= 0:
        raise ValueError(f"{testcase_name}: blocknum must be positive")
    if max_block_per_batch < 0:
        raise ValueError(f"{testcase_name}: max_block_per_batch must be non-negative")
    if layout_kv != "PA_BNSD":
        raise ValueError(f"{testcase_name}: layout_kv must be PA_BNSD")
    if sparse_mode not in ("dense", "random"):
        raise ValueError(f"{testcase_name}: invalid sparse_mode={sparse_mode!r}")
    total_q = sum(q_lengths)
    max_q = max(q_lengths, default=0)
    max_kv = max(kv_lengths, default=0)
    q_blocks = math.ceil(max_q / sparse_block_size_q)
    kv_blocks = math.ceil(max_kv / sparse_block_size_kv)
    required_pages = max((math.ceil(length / blocksize) for length in kv_lengths), default=0)
    logical_pages = max(required_pages, int(max_block_per_batch))
    if layout_q != "TND":
        raise ValueError(f"{testcase_name}: unsupported layout_q={layout_q!r}")
    query_shape = (total_q, num_heads_q, head_dim)
    q_descale_shape = (total_q, num_heads_q, head_dim // 64, 2)
    k_descale_shape = (blocknum, num_heads_kv, blocksize, head_dim // 64, 2)
    v_descale_shape = (blocknum, num_heads_kv, math.ceil(blocksize / 64), head_dim, 2)
    scale_dtype = p_scale_dtype = "float8_e8m0"

    shapes = (
        query_shape,
        (blocknum, num_heads_kv, blocksize, head_dim),
        (blocknum, num_heads_kv, blocksize, head_dim),
        q_descale_shape, k_descale_shape, v_descale_shape,
        (batch_size, num_heads_q, q_blocks, kv_blocks),
        (batch_size, num_heads_q, q_blocks),
        (0,) if p_scale_value is None else (1,),
        (batch_size + 1,), (0,), (0,), (batch_size,),
        (batch_size, logical_pages),
        (2048, 2048) if int(mask_mode) == 3 else (0, 0),
        (0,),
    )
    dtypes = (
        "float8_e4m3fn", "float8_e4m3fn", "float8_e4m3fn",
        scale_dtype, scale_dtype, scale_dtype, "int32", "int32",
        p_scale_dtype, "int32", "int32", "int32", "int32", "int32",
        "uint8", "int32",
    )
    attrs = {
        "p_scale_value": None if p_scale_value is None else float(p_scale_value),
        "cu_seqlens_q_value": _prefix_sums(q_lengths),
        "cu_seqlens_kv_value": [],
        "seqused_q_value": [],
        "seqused_kv_value": kv_lengths,
        "quant_mode": quant_mode,
        "softmax_scale": float(softmax_scale),
        "mask_mode": int(mask_mode),
        "blocksize": blocksize,
        "sparse_block_size_q": sparse_block_size_q,
        "sparse_block_size_kv": sparse_block_size_kv,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
        "layout_sparse_indices": layout_sparse_indices,
        "return_softmax_lse": bool(return_softmax_lse),
        "batch_size": batch_size,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "sparse_mode": sparse_mode,
    }
    q_range, k_range, v_range = map(_normalize_range, (data_range_q, data_range_k, data_range_v))
    p_range = ((None, None) if p_scale_value is None
               else (float(p_scale_value), float(p_scale_value)))
    ranges = (
        q_range, k_range, v_range,
        (None, None), (None, None), (None, None),
        (-1, max(kv_blocks - 1, 0)), (0, kv_blocks), p_range,
        (0, total_q), (None, None), (None, None), (0, max_kv),
        (0, max(blocknum - 1, 0)), (0, 1), (None, None),
    )
    return {
        "testcase_name": testcase_name,
        "api_name": API_NAME,
        "tensor_view_shapes": repr(shapes),
        "tensor_dtypes": repr(dtypes),
        "tensor_formats": repr(TENSOR_FORMATS),
        "attributes": repr(attrs),
        "input_data_ranges": repr(ranges),
        "precision_tolerances": repr(PRECISION_TOLERANCES),
        "absolute_precision": repr(1e-8),
    }

def load_python_cases():
    namespace = runpy.run_path(str(PYTHON_CASES))
    return [
        dict(case)
        for case in namespace["TestCases"]
        if case.get("enable") is True
    ]


def sequence_lengths(case):
    batch = int(case["B"])
    cu_q = [int(item) for item in case["cu_seqlens_q"]]
    cu_kv = [int(item) for item in case["cu_seqlens_kv"]]
    seq_q = [int(item) for item in case["seqused_q"]]
    kv_lengths = [int(item) for item in case["seqused_kv"]]
    if len(cu_q) != batch + 1 or not cu_q or cu_q[0] != 0:
        raise ValueError(f"{case['name']}: invalid cu_seqlens_q")
    if cu_kv or seq_q:
        raise ValueError(
            f"{case['name']}: cu_seqlens_kv and seqused_q must be empty"
        )
    if len(kv_lengths) != batch:
        raise ValueError(f"{case['name']}: seqused_kv must contain B values")
    q_lengths = [end - start for start, end in zip(cu_q, cu_q[1:])]
    if any(length < 0 for length in q_lengths + kv_lengths):
        raise ValueError(f"{case['name']}: sequence lengths must be non-negative")
    return q_lengths, kv_lengths


def normalize_data_range(case, field):
    value = case[field]
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{case['name']}: {field} must contain exactly [min, max]"
            )
        low, high = float(value[0]), float(value[1])
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(
                f"{case['name']}: {field} bounds must be finite, got {value}"
            )
        if low > high:
            raise ValueError(
                f"{case['name']}: {field} min must not exceed max, got {value}"
            )
        return [low, high]

    radius = float(value)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(
            f"{case['name']}: scalar {field} must be finite and non-negative, "
            f"got {value}"
        )
    return radius


def validate_case(case, q_lengths, kv_lengths):
    name = case["name"]
    if int(case["quant_mode"]) != 2:
        raise ValueError(f"{name}: only quant_mode=2 belongs in the MXFP8 CSV")
    if case["layout_q"] != "TND":
        raise ValueError(f"{name}: MXFP8 currently requires layout_q='TND'")
    if case["layout_kv"] != "PA_BNSD":
        raise ValueError(f"{name}: MXFP8 currently requires layout_kv='PA_BNSD'")
    if int(case["D"]) % 64 != 0:
        raise ValueError(f"{name}: D must be divisible by 64 for paired MXFP8 scales")
    if int(case["N1"]) % int(case["N2"]) != 0:
        raise ValueError(f"{name}: N1 must be divisible by N2")
    page_size = int(case["block_size"])
    if page_size <= 0 or page_size % int(case["sparse_kv_block_size"]) != 0:
        raise ValueError(
            f"{name}: block_size must be a positive multiple of sparse_kv_block_size"
    )
    if sum(q_lengths) <= 0 or sum(kv_lengths) <= 0:
        raise ValueError(f"{name}: total Q/KV sequence length must be positive")
    blocknum = case.get("blocknum")
    if blocknum is not None and int(blocknum) <= 0:
        warnings.warn(
            f"{name}: blocknum is expected to be >= 0; got {blocknum} (<= 0), "
            "so it will be derived from seqused_kv",
            UserWarning,
            stacklevel=2,
        )


def case_to_row(case):
    q_lengths, kv_lengths = sequence_lengths(case)
    validate_case(case, q_lengths, kv_lengths)

    batch = int(case["B"])
    n1 = int(case["N1"])
    n2 = int(case["N2"])
    head_dim = int(case["D"])
    page_size = int(case["block_size"])
    sparse_q = int(case["sparse_q_block_size"])
    sparse_kv = int(case["sparse_kv_block_size"])
    logical_page_count = sum(math.ceil(length / page_size) for length in kv_lengths)
    blocknum = case.get("blocknum")
    physical_page_count = (
        int(blocknum)
        if blocknum is not None and int(blocknum) > 0
        else logical_page_count
    )
    p_scale_value = case.get("p_scale_value")
    softmax_scale = case.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    return build_case_row(
        testcase_name=case["name"],
        batch_size=batch,
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=head_dim,
        q_lengths=q_lengths,
        kv_lengths=kv_lengths,
        blocksize=page_size,
        blocknum=physical_page_count,
        max_block_per_batch=int(case["max_block_per_batch"]),
        sparse_block_size_q=sparse_q,
        sparse_block_size_kv=sparse_kv,
        mask_mode=int(case["mask_mode"]),
        p_scale_value=p_scale_value,
        softmax_scale=softmax_scale,
        layout_q=case["layout_q"],
        layout_kv=case["layout_kv"],
        layout_out=case["layout_out"],
        layout_sparse_indices=case["layout_sparse_indices"],
        return_softmax_lse=bool(case["return_softmax_lse"]),
        sparse_mode=case["sparse_mode"],
        data_range_q=normalize_data_range(case, "data_range_q"),
        data_range_k=normalize_data_range(case, "data_range_k"),
        data_range_v=normalize_data_range(case, "data_range_v"),
    )


def render_csv():
    rows = []
    names = set()
    for case in load_python_cases():
        row = case_to_row(case)
        name = row["testcase_name"]
        if name in names:
            raise ValueError(f"duplicate enabled testcase_name: {name}")
        names.add(name)
        rows.append(row)

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue(), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in CSV differs from regenerated content",
    )
    args = parser.parse_args()

    content, count = render_csv()
    if args.check:
        current = args.output.read_text(encoding="utf-8")
        if current != content:
            raise SystemExit(f"{args.output} is stale; regenerate it")
        print(f"validated {count} MXFP8 TTK cases")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="")
    print(f"generated {count} MXFP8 TTK cases at {args.output}")


if __name__ == "__main__":
    main()
