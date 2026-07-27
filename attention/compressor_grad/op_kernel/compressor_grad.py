# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DenseLightningIndexLseV2 kernel.

Computes softmax LSE for dense lightning indexer.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField as tilingkey_field


class CompressorGradTilingKey:
    Coff     = tilingkey_field(bits=2, values=[1, 2])
    Layout   = tilingkey_field(bits=1, values=[0, 1])
    DataType = tilingkey_field(bits=2, values=[0, 1])

    def is_valid(self, key):
        coff, layout, dtype = key
        return coff in (1, 2) and layout in (0, 1) and dtype in (0, 1)


@dataclass
class CompressorGradTiling:
    batchSize:        int = 1
    tokenSize:        int = 0
    seqSize:          int = 0
    cmpRatio:         int = 4
    hiddenSize:       int = 512
    headDim:          int = 128
    featureDim:       int = 0
    scKvRowCount:     int = 0
    outStride:        int = 0
    groupSize:        int = 0
    dBaseSize:        int = 0
    nCorePerGroup:    int = 0
    mPerRoundPerGroup:int = 0
    mBaseSize:        int = 0
    mPerRoundAll:     int = 0
    dbRows:           int = 0
    ape_ws_elems:     int = 0
    dx_per_group:     int = 0
    dx_stride:        int = 0
    dx_ws_elems:      int = 0
    d_x_rows:         int = 0
    outputRows:       int = 0
    rgChunkNum:       int = 0
    rgPerCore:        int = 0
    rgRemain:         int = 0


# # ================================================================
# #  TilingKey
# # ================================================================
# @dataclass
# class CompressorGradTiling:
#     b: int
#     s1: int
#     s2: int
#     n1: int
#     d: int
#     cmp_ratio: int

# class CompressorGradTilingKey:
#     is_long_s2 = TilingKeyField(bits=1, values=[0, 1])
# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@pl.jit(auto_mutex=True,
        tiling_key=CompressorGradTilingKey,
        datatype={
            "x": "input_dtype",
        })
def compressor_grad(
    x: pl.Ptr[pl.DT_UINT8],
    wkv: pl.Ptr[pl.DT_UINT8],
    wgate: pl.Ptr[pl.DT_UINT8],
    d_cmp_kv: pl.Ptr[pl.DT_UINT8],
    softmax_score: pl.Ptr[pl.DT_UINT8],
    kv: pl.Ptr[pl.DT_UINT8],
    cu_seq_lens: pl.Ptr[pl.DT_UINT8],
    seq_used: pl.Ptr[pl.DT_UINT8],
    start_pos: pl.Ptr[pl.DT_UINT8],
    d_x: pl.Ptr[pl.DT_UINT8],
    d_wkv: pl.Ptr[pl.DT_UINT8],
    d_wgate: pl.Ptr[pl.DT_UINT8],
    d_ape: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: CompressorGradTiling,
):
    pass
