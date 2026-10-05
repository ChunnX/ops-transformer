# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Performance cases: required P-01 to P-07 and supplemental shapes."""

from typing import NamedTuple


class PerformanceCase(NamedTuple):
    # Keep name/seq/batch/hk in the same order as test_wide_sequence.py.
    name: str
    seq: int
    batch: int
    hk: int
    heads: int
    dim: int
    dtype: str
    causal: bool


CASES = [
    PerformanceCase("P-01", 4096, 1, 32, 32, 128, "fp16", False),
    PerformanceCase("P-02", 8192, 1, 32, 32, 128, "fp16", False),
    PerformanceCase("P-03", 8192, 1, 32, 32, 128, "fp16", True),
    PerformanceCase("P-04", 8192, 2, 32, 32, 128, "fp16", False),
    PerformanceCase("P-05", 4096, 1, 8, 32, 128, "fp16", False),
    PerformanceCase("P-06", 4096, 1, 32, 32, 128, "bf16", False),
    PerformanceCase("P-07", 16384, 1, 32, 32, 128, "fp16", False),
]

ALL_CASES = CASES + [
    PerformanceCase("P-08", 32768, 1, 32, 32, 128, "fp16", False),
    PerformanceCase("P-09", 65536, 1, 32, 32, 128, "fp16", False),
    PerformanceCase("P-10", 131072, 1, 32, 32, 128, "fp16", False),
    PerformanceCase("X-01", 2048, 1, 32, 32, 128, "fp16", False),
    PerformanceCase("X-02", 4096, 1, 16, 16, 128, "fp16", False),
    PerformanceCase("X-03", 4096, 1, 4, 32, 128, "fp16", False),
    PerformanceCase("X-04", 4096, 1, 32, 32, 64, "fp16", False),
    PerformanceCase("X-05", 2048, 4, 32, 32, 128, "fp16", False),
    PerformanceCase("X-06", 8192, 1, 32, 32, 128, "bf16", False),
    PerformanceCase("X-07", 8192, 1, 8, 32, 128, "fp16", True),
]
