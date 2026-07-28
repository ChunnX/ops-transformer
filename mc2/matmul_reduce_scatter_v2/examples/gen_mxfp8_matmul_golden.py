#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Generate random MXFP8 matmul inputs and the dequantized golden result."""

import argparse
import math
from pathlib import Path

import torch


MX_GROUP_SIZE = 32
FP8_DTYPES = {
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
}
OUTPUT_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def get_dtype(name):
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(f"Current PyTorch does not support torch.{name}")
    return dtype


def build_mxfp8_matmul_golden(
    m,
    k,
    n,
    x1_dtype_name="float8_e4m3fn",
    x2_dtype_name="float8_e4m3fn",
    output_dtype_name="float16",
    seed=0,
):
    """Build x1/x2/scales and calculate the MXFP8 matmul result.

    Scale layouts match MatmulReduceScatterV2:
      x1Scale: [M, ceil(K / 64), 2]
      x2Scale: [ceil(K / 64), N, 2]
    """
    if m <= 0 or k <= 0 or n <= 0:
        raise ValueError(f"m, k and n must be positive, got {(m, k, n)}")

    torch.manual_seed(seed)
    x1_dtype = get_dtype(FP8_DTYPES[x1_dtype_name])
    x2_dtype = get_dtype(FP8_DTYPES[x2_dtype_name])
    scale_dtype = get_dtype("float8_e8m0fnu")

    # Integer values in this range are exactly representable by both FP8 types.
    x1 = torch.randint(-4, 5, (m, k), dtype=torch.int32).to(x1_dtype)
    x2 = torch.randint(-4, 5, (k, n), dtype=torch.int32).to(x2_dtype)

    group_count = math.ceil(k / MX_GROUP_SIZE)
    padded_group_count = math.ceil(group_count / 2) * 2

    # E8M0 scale is a power of two. Generate random exponents in [-3, 3].
    x1_scale_exp = torch.randint(-3, 4, (m, padded_group_count))
    x2_scale_exp = torch.randint(-3, 4, (padded_group_count, n))
    if group_count != padded_group_count:
        x1_scale_exp[:, -1] = 0
        x2_scale_exp[-1, :] = 0

    x1_scale_unpacked = torch.exp2(x1_scale_exp.to(torch.float32)).to(scale_dtype)
    x2_scale_unpacked = torch.exp2(x2_scale_exp.to(torch.float32)).to(scale_dtype)

    scale_pack_count = padded_group_count // 2
    x1_scale = x1_scale_unpacked.reshape(m, scale_pack_count, 2).contiguous()
    x2_scale = (
        x2_scale_unpacked.reshape(scale_pack_count, 2, n)
        .permute(0, 2, 1)
        .contiguous()
    )

    # Expand every K/32 scale and calculate the equivalent dequantized matmul.
    x1_scale_expanded = x1_scale_unpacked.to(torch.float32).repeat_interleave(
        MX_GROUP_SIZE, dim=1
    )[:, :k]
    x2_scale_expanded = x2_scale_unpacked.to(torch.float32).repeat_interleave(
        MX_GROUP_SIZE, dim=0
    )[:k, :]
    x1_dequant = x1.to(torch.float32) * x1_scale_expanded
    x2_dequant = x2.to(torch.float32) * x2_scale_expanded
    expected_fp32 = x1_dequant @ x2_dequant

    expected = expected_fp32.to(OUTPUT_DTYPES[output_dtype_name])
    return {
        "x1": x1,
        "x2": x2,
        "x1Scale": x1_scale,
        "x2Scale": x2_scale,
        "expected": expected,
        "expectedFp32": expected_fp32,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--x1-dtype", choices=FP8_DTYPES, default="float8_e4m3fn")
    parser.add_argument("--x2-dtype", choices=FP8_DTYPES, default="float8_e4m3fn")
    parser.add_argument("--output-dtype", choices=OUTPUT_DTYPES, default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("mxfp8_matmul_data.pt"))
    args = parser.parse_args()

    data = build_mxfp8_matmul_golden(
        args.m,
        args.k,
        args.n,
        args.x1_dtype,
        args.x2_dtype,
        args.output_dtype,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)

    for name, tensor in data.items():
        print(f"{name:14s}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
