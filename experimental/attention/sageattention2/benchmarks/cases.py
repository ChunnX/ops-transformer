# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Performance cases: required P-01 to P-07 and supplemental shapes."""

CASES = [
    ("P-01", 4096, 32, 1, "fp16", False),
    ("P-02", 8192, 32, 1, "fp16", False),
    ("P-03", 8192, 32, 1, "fp16", True),
    ("P-04", 8192, 32, 2, "fp16", False),
    ("P-05", 4096, 8, 1, "fp16", False),
    ("P-06", 4096, 32, 1, "bf16", False),
    ("P-07", 16384, 32, 1, "fp16", False),
]

ALL_CASES = CASES + [
    ("P-08", 32768, 32, 1, "fp16", False),
    ("P-09", 65536, 32, 1, "fp16", False),
    ("P-10", 131072, 32, 1, "fp16", False),
    ("X-01", 2048, 32, 1, "fp16", False),
    ("X-02", 4096, 16, 1, "fp16", False),
    ("X-03", 4096, 4, 1, "fp16", False),
    ("X-04", 4096, 32, 1, "fp16", False),
    ("X-05", 2048, 32, 4, "fp16", False),
    ("X-06", 8192, 32, 1, "bf16", False),
    ("X-07", 8192, 8, 1, "fp16", True),
]
