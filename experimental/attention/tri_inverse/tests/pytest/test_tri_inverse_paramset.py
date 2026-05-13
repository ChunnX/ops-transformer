# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CHUNK_SIZE = 64

ALIGNED_SHAPES = [
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    10240,
]

UNALIGNED_SHAPES = [
    65,
    257,
    1025,
    4097,
    10241,
]

PROFILE_SHAPES = ALIGNED_SHAPES + UNALIGNED_SHAPES

ENABLED_PARAMS = [
    {"shape": PROFILE_SHAPES, "seed": [0, 21]},
]


def shape_to_tile_num(shape):
    return (shape + CHUNK_SIZE - 1) // CHUNK_SIZE
