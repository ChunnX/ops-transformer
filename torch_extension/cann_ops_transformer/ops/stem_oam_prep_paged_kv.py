# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

DIM_QK = 128


class StemOamPrepPagedKvOpBuilder(OpBuilder):
    def __init__(self):
        super(StemOamPrepPagedKvOpBuilder, self).__init__("stem_oam_prep_paged_kv")

    def sources(self):
        return ['ops/csrc/stem_oam_prep_paged_kv.cpp']

    def schema(self) -> str:
        return [
            "stem_oam_prep_paged_kv(Tensor kCache, Tensor vCache, Tensor kvIndices, "
            "Tensor kvSeqLens, Tensor kScaleCache, Tensor vScale, "
            "float lambdaMag, int cacheLayout, int kvBlockSize, "
            "int stemBlockSize, int stemStride) -> (Tensor, Tensor)"
        ]

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def stem_oam_prep_paged_kv_meta(
            kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale,
            lambdaMag, cacheLayout, kvBlockSize, stemBlockSize, stemStride):
            batch = kvSeqLens.shape[0]
            numHeads = kCache.shape[2] if cacheLayout == 0 else kCache.shape[1]
            # estimate max_Kb from kvIndices as an upper bound
            maxKvBlocks = kvIndices.shape[1]
            maxKvLen = maxKvBlocks * kvBlockSize
            maxKb = (maxKvLen + stemBlockSize - 1) // stemBlockSize
            if maxKb < 1:
                maxKb = 1
            kflatDim = stemStride * DIM_QK
            kFlat = torch.empty([batch, numHeads, maxKb, kflatDim],
                                dtype=torch.bfloat16, device="meta")
            vBias = torch.empty([batch, numHeads, maxKb],
                                dtype=torch.float32, device="meta")
            return (kFlat, vBias)


stem_oam_prep_paged_kv_builder = StemOamPrepPagedKvOpBuilder()


@impl(AS_LIBRARY, "stem_oam_prep_paged_kv", "PrivateUse1")
def stem_oam_prep_paged_kv(
    kCache: torch.Tensor, vCache: torch.Tensor, kvIndices: torch.Tensor,
    kvSeqLens: torch.Tensor, kScaleCache: torch.Tensor, vScale: torch.Tensor,
    lambdaMag: float, cacheLayout: int, kvBlockSize: int,
    stemBlockSize: int, stemStride: int):
    op_module = stem_oam_prep_paged_kv_builder.load()
    return op_module.stem_oam_prep_paged_kv(
        kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale,
        lambdaMag, cacheLayout, kvBlockSize, stemBlockSize, stemStride)
