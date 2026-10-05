# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate the native per-token INT8 / MXFP8 test matrix.

Only effective kernel options form matrix axes. Compatibility aliases and
fallback warnings are covered by test_mxfp8.py; performance shapes live in
benchmarks/cases.py. All generated inputs use the same reproducible seed.
"""

from itertools import product

DTYPES = ("float16", "bfloat16")
LAYOUTS = ("HND", "NHD")
SHAPE_FIELDS = (
    "batch_size",
    "num_qo_heads",
    "num_kv_heads",
    "qo_len",
    "kv_len",
    "head_dim",
)
SMOKE_SHAPES = (
    (1, 8, 8, 128, 128, 64),
    (1, 16, 16, 256, 256, 128),
    (2, 8, 8, 128, 128, 64),
    (1, 8, 2, 128, 128, 64),
    (1, 8, 8, 64, 128, 64),
)


def _case(shape=SMOKE_SHAPES[0], **options):
    case = {
        "suite": "functional",
        "shape": dict(zip(SHAPE_FIELDS, shape)),
        "dtype": "float16",
        "tensor_layout": "HND",
        "is_causal": False,
        "qk_quant_gran": "per_token",
        "sm_scale": None,
        "pv_accum_dtype": "fp32+fp32",
        "smooth_k": True,
        "return_lse": False,
        "distribution": "normal",
        "seed": 42,
        "expect": "pass",
    }
    case.update(options)
    return case


def _configurations():
    for shape, dtype, layout in product(SMOKE_SHAPES, DTYPES, LAYOUTS):
        yield _case(shape, dtype=dtype, tensor_layout=layout)

    for dim, dtype, layout in product(
        (1, 8, 16, 31, 32, 33, 48, 63, 64, 65), DTYPES, LAYOUTS
    ):
        yield _case((1, 8, 8, 64, 64, dim), dtype=dtype, tensor_layout=layout)

    for seq, (hq, hk), dtype in product(
        (64, 128, 256), ((8, 8), (16, 8), (32, 8), (32, 4)), DTYPES
    ):
        yield _case((1, hq, hk, seq, seq, 128), dtype=dtype, is_causal=True)

    for layout, causal, smooth_k, lse in product(
        LAYOUTS, (False, True), (False, True), (False, True)
    ):
        yield _case(
            tensor_layout=layout,
            is_causal=causal,
            smooth_k=smooth_k,
            return_lse=lse,
        )

    lengths = (
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (128, 256),
        (256, 512),
    )
    for heads, (sq, sk), dim in product((8, 16), lengths, range(16, 129, 16)):
        yield _case((1, heads, heads, sq, sk, dim))

    # Each accuracy configuration checks both the quantized CPU reference and SDPA.
    for shape, dtype, layout, distribution in product(
        SMOKE_SHAPES[:2], DTYPES, LAYOUTS, ("normal", "uniform")
    ):
        yield _case(
            shape,
            suite="accuracy",
            dtype=dtype,
            tensor_layout=layout,
            distribution=distribution,
        )

    for shape, options in (
        ((1, 8, 8, 128, 128, 256), {"error_match": "head_dim"}),
        ((1, 8, 8, 64, 128, 64), {"is_causal": True, "error_match": "causal requires"}),
        ((1, 7, 4, 64, 64, 64), {"error_match": "Hq must be divisible"}),
        (
            (1, 8, 8, 64, 64, 64),
            {"qk_quant_gran": "per_block", "error_match": "qk_quant_gran"},
        ),
        (
            (1, 8, 8, 64, 64, 64),
            {"tensor_layout": "BNSD", "error_match": "tensor_layout"},
        ),
    ):
        yield _case(shape, suite="functional_neg", expect="error", **options)


def generate_cases():
    """Yield unique configurations with readable, stable pytest IDs."""
    seen = set()
    for case in _configurations():
        shape = tuple(case["shape"][name] for name in SHAPE_FIELDS)
        fields = tuple((name, value) for name, value in case.items() if name != "shape")
        key = (shape, fields)
        if key in seen:
            continue
        seen.add(key)
        b, hq, hk, sq, sk, dim = shape
        case["id"] = (
            f"{case['suite']}-b{b}-h{hq}kv{hk}-q{sq}k{sk}-d{dim}-"
            f"{case['dtype']}-{case['tensor_layout']}-c{int(case['is_causal'])}-"
            f"smooth{int(case['smooth_k'])}-lse{int(case['return_lse'])}-"
            f"{case['distribution']}-{case['qk_quant_gran']}"
        )
        yield case
