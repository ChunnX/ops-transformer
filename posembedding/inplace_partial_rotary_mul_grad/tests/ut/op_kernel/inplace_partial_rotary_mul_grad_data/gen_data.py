#!/usr/bin/env python3
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
Generate test data for InplacePartialRotaryMulGrad kernel UT.

Usage: python3 gen_data.py <B> <S> <N> <D> <dy_dtype> <cos_dtype> [cos_shape_desc]

  dy_dtype:   float16 | bfloat16 | float32
  cos_dtype:  float16 | bfloat16 | float32
  cos_shape_desc: NO_BROADCAST(default) | BROADCAST_BSN(111D) |
                  BSND(1S1D) | SBND | BNSD_11SD | BNSD_B1SD

Outputs:
  dy.bin   — input dy tensor [B,S,N,D] or [B,N,S,D] depending on layout
  cos.bin  — cos tensor
  sin.bin  — sin tensor

Interleave mode: dx[2k]=cos[2k]*dy[2k]+sin[2k+1]*dy[2k+1]
                 dx[2k+1]=cos[2k+1]*dy[2k+1]-sin[2k]*dy[2k]
"""

import sys
import numpy as np


def interleave_grad_ref(dy, cos, sin, dx, layout, slice_start, slice_end, b, s, n, d):
    """Reference implementation of interleave rotary grad with partial slice."""
    cosD = slice_end - slice_start

    if layout in ("NO_BROADCAST", "BROADCAST_BSN"):
        # Both use [B,N,S,D] layout after possible broadcast
        for bi in range(b):
            if layout == "BROADCAST_BSN":
                ci = 0  # cos broadcast on all dims
            else:
                ci = bi
            for ni in range(n):
                for si in range(s):
                    # After reshape: flat index
                    base = (bi * n * s + ni * s + si) * d
                    cos_base = (
                        ci * s * cosD + si * cosD
                        if layout == "NO_BROADCAST"
                        else si * cosD
                    )
                    for di in range(d):
                        dx[base + di] = dy[base + di]
                    for k in range(cosD // 2):
                        idx0 = base + slice_start + 2 * k
                        idx1 = base + slice_start + 2 * k + 1
                        c0 = cos_base + 2 * k
                        c1 = cos_base + 2 * k + 1
                        dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1]
                        dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0]

    elif layout == "BSND":
        for bi in range(b):
            for si in range(s):
                cos_base = si * cosD
                for ni in range(n):
                    base = ((bi * s + si) * n + ni) * d
                    for di in range(d):
                        dx[base + di] = dy[base + di]
                    for k in range(cosD // 2):
                        idx0 = base + slice_start + 2 * k
                        idx1 = base + slice_start + 2 * k + 1
                        c0 = cos_base + 2 * k
                        c1 = cos_base + 2 * k + 1
                        dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1]
                        dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0]

    elif layout == "SBND":
        for si in range(s):
            for bi in range(b):
                cos_base = si * cosD
                for ni in range(n):
                    base = ((si * b + bi) * n + ni) * d
                    for di in range(d):
                        dx[base + di] = dy[base + di]
                    for k in range(cosD // 2):
                        idx0 = base + slice_start + 2 * k
                        idx1 = base + slice_start + 2 * k + 1
                        c0 = cos_base + 2 * k
                        c1 = cos_base + 2 * k + 1
                        dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1]
                        dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0]

    elif layout == "BNSD":
        for bi in range(b):
            cos_base = bi * s * cosD
            for ni in range(n):
                for si in range(s):
                    base = (bi * n * s + ni * s + si) * d
                    for di in range(d):
                        dx[base + di] = dy[base + di]
                    for k in range(cosD // 2):
                        idx0 = base + slice_start + 2 * k
                        idx1 = base + slice_start + 2 * k + 1
                        c0 = cos_base + si * cosD + 2 * k
                        c1 = cos_base + si * cosD + 2 * k + 1
                        dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1]
                        dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0]

    elif layout == "BNSD_BA":
        # BNSD with cosb_==1 — cos broadcast on B
        for bi in range(b):
            for ni in range(n):
                for si in range(s):
                    cos_base = si * cosD
                    base = (bi * n * s + ni * s + si) * d
                    for di in range(d):
                        dx[base + di] = dy[base + di]
                    for k in range(cosD // 2):
                        idx0 = base + slice_start + 2 * k
                        idx1 = base + slice_start + 2 * k + 1
                        c0 = cos_base + 2 * k
                        c1 = cos_base + 2 * k + 1
                        dx[idx0] = cos[c0] * dy[idx0] + sin[c1] * dy[idx1]
                        dx[idx1] = cos[c1] * dy[idx1] - sin[c0] * dy[idx0]


def generate_data(
    b,
    s,
    n,
    d,
    dy_dtype,
    cos_dtype,
    layout="NO_BROADCAST",
    slice_start=0,
    slice_end=None,
):
    """Generate random test data for InplacePartialRotaryMulGrad."""
    dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float32,
        "float32": np.float32,
    }
    if dy_dtype not in dtype_map or cos_dtype not in dtype_map:
        print("Error: unsupported dtype", file=sys.stderr)
        sys.exit(1)

    np_dy = dtype_map[dy_dtype]
    np_cos = dtype_map[cos_dtype]

    if slice_end is None:
        slice_end = d

    cosD = slice_end - slice_start
    assert cosD % 2 == 0, f"cosD={cosD} must be multiple of 2 for interleave mode"

    # Build shapes based on layout
    if layout in ("NO_BROADCAST", "BROADCAST_BSN"):
        dy_shape = [b, n, s, d]
        if layout == "NO_BROADCAST":
            cos_shape = [b, n, s, cosD]
        else:
            cos_shape = [1, 1, 1, cosD]
    elif layout == "BSND":
        dy_shape = [b, s, n, d]
        cos_shape = [1, s, 1, cosD]
    elif layout == "SBND":
        dy_shape = [s, b, n, d]
        cos_shape = [s, 1, 1, cosD]
    elif layout in ("BNSD", "BNSD_BA"):
        dy_shape = [b, n, s, d]
        if layout == "BNSD_BA":
            cos_shape = [1, 1, s, cosD]
        else:
            cos_shape = [b, 1, s, cosD]
    else:
        dy_shape = [b, s, n, d]
        cos_shape = [b, s, n, d]

    dy = np.random.randn(*dy_shape).astype(np_dy)
    cos = np.random.randn(*cos_shape).astype(np_cos)
    sin = np.random.randn(*cos_shape).astype(np_cos)

    # Compute reference dx
    dy_flat = dy.flatten().astype(np.float64)
    dx_ref = dy_flat.copy()
    cos_flat = cos.flatten().astype(np.float64)
    sin_flat = sin.flatten().astype(np.float64)

    interleave_grad_ref(
        dy_flat, cos_flat, sin_flat, dx_ref, layout, slice_start, slice_end, b, s, n, d
    )

    # Write binaries
    dy.astype(np_dy).tofile("dy.bin")
    cos.astype(np_cos).tofile("cos.bin")
    sin.astype(np_cos).tofile("sin.bin")
    dx_ref.astype(np.float32).tofile("dx_ref.bin")

    # Write params file
    with open("params.txt", "w") as f:
        f.write(f"B={b} S={s} N={n} D={d}\n")
        f.write(f"dy_dtype={dy_dtype} cos_dtype={cos_dtype}\n")
        f.write(f"layout={layout}\n")
        f.write(f"slice=[{slice_start},{slice_end}]\n")
        f.write(f"dy_shape={'x'.join(str(x) for x in dy_shape)}\n")
        f.write(f"cos_shape={'x'.join(str(x) for x in cos_shape)}\n")

    print(
        f"Generated: dy{dy.shape} cos{cos.shape} sin{sin.shape} "
        f"dy_dtype={dy_dtype} cos_dtype={cos_dtype} layout={layout} "
        f"slice=[{slice_start},{slice_end}]"
    )


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(
            "Usage: python3 gen_data.py <B> <S> <N> <D> <dy_dtype> <cos_dtype> "
            "[layout] [slice_start] [slice_end]",
            file=sys.stderr,
        )
        print(
            "  layout: NO_BROADCAST | BROADCAST_BSN | BSND | SBND | BNSD | BNSD_BA",
            file=sys.stderr,
        )
        sys.exit(1)

    b, s, n, d = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    dy_dtype = sys.argv[5]
    cos_dtype = sys.argv[6]
    layout = sys.argv[7] if len(sys.argv) > 7 else "NO_BROADCAST"
    slice_start = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    slice_end = int(sys.argv[9]) if len(sys.argv) > 9 else d
    generate_data(b, s, n, d, dy_dtype, cos_dtype, layout, slice_start, slice_end)
