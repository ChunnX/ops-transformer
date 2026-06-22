# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MlaPreprocessOpBuilder(OpBuilder):
    def __init__(self):
        super(MlaPreprocessOpBuilder, self).__init__("npu_mla_preprocess")

    def sources(self):
        return ['ops/csrc/mla_preprocess.cpp']

    def schema(self) -> str:
        return (
            "npu_mla_preprocess("
            "Tensor input, Tensor gamma0, Tensor beta0, "
            "Tensor quantScale0, Tensor quantOffset0, "
            "Tensor wdqkv, Tensor deScale0, Tensor bias0, "
            "Tensor gamma1, Tensor beta1, "
            "Tensor quantScale1, Tensor quantOffset1, "
            "Tensor wuq, Tensor deScale1, Tensor bias1, "
            "Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, "
            "Tensor kvCache, Tensor slotMapping, "
            "Tensor? kvCacheRope=None, Tensor? ctkvScale=None, Tensor? qNopeScale=None, "
            "int wdqDim=1536, int qRopeDim=64, int kRopeDim=64, "
            "float epsilon=1e-5, "
            "int qRotaryCoeff=2, int kRotaryCoeff=2, "
            "bool transposeWdq=True, bool transposeWuq=True, bool transposeWuk=True, "
            "int cacheMode=1, int quantMode=0, "
            "bool doRmsNorm=True, int wdkvSplitCount=1"
            ") -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mla_preprocess_meta(
            input, gamma0, beta0,
            quantScale0, quantOffset0,
            wdqkv, deScale0, bias0,
            gamma1, beta1,
            quantScale1, quantOffset1,
            wuq, deScale1, bias1,
            gamma2, cos, sin, wuk,
            kvCache, slotMapping,
            kvCacheRope=None, ctkvScale=None, qNopeScale=None,
            wdqDim=1536, qRopeDim=64, kRopeDim=64,
            epsilon=1e-5,
            qRotaryCoeff=2, kRotaryCoeff=2,
            transposeWdq=True, transposeWuq=True, transposeWuk=True,
            cacheMode=1, quantMode=0,
            doRmsNorm=True, wdkvSplitCount=1
        ):
            tokenNum = input.size(0)
            headNum = wuk.size(0)

            if cacheMode == 0:
                qOutSize = (tokenNum, headNum, 576)
            else:
                qOutSize = (tokenNum, headNum, 512)

            qOutDtype = torch.int8 if cacheMode == 2 else input.dtype
            qOut = torch.empty(qOutSize, dtype=qOutDtype, device='meta')

            # kvCache is in-place
            kvCacheOut = torch.empty(kvCache.size(), dtype=kvCache.dtype, device='meta')

            if cacheMode == 0:
                qRopeOut = torch.empty((0,), dtype=input.dtype, device='meta')
                krCacheOut = torch.empty((0,), dtype=input.dtype, device='meta')
            else:
                qRopeOut = torch.empty((tokenNum, headNum, 64), dtype=input.dtype, device='meta')
                if cacheMode == 1:
                    blockNum = kvCache.size(0)
                    blockSize = kvCache.size(1)
                    krCacheOut = torch.empty((blockNum, blockSize, 1, 64), dtype=input.dtype, device='meta')
                else:
                    blockNum = kvCache.size(0)
                    blockSize = kvCache.size(2)
                    krCacheOut = torch.empty((blockNum, 4, blockSize, 16), dtype=input.dtype, device='meta')

            return (qOut, kvCacheOut, qRopeOut, krCacheOut)


mla_preprocess_op_builder = MlaPreprocessOpBuilder()
op_module = mla_preprocess_op_builder.load()


@impl(AS_LIBRARY, mla_preprocess_op_builder.name, "PrivateUse1")
def npu_mla_preprocess(
    input, gamma0, beta0,
    quantScale0, quantOffset0,
    wdqkv, deScale0, bias0,
    gamma1, beta1,
    quantScale1, quantOffset1,
    wuq, deScale1, bias1,
    gamma2, cos, sin, wuk,
    kvCache, slotMapping,
    kvCacheRope=None, ctkvScale=None, qNopeScale=None,
    wdqDim=1536, qRopeDim=64, kRopeDim=64,
    epsilon=1e-5,
    qRotaryCoeff=2, kRotaryCoeff=2,
    transposeWdq=True, transposeWuq=True, transposeWuk=True,
    cacheMode=1, quantMode=0,
    doRmsNorm=True, wdkvSplitCount=1
):
    return op_module.npu_mla_preprocess(
        input, gamma0, beta0,
        quantScale0, quantOffset0,
        wdqkv, deScale0, bias0,
        gamma1, beta1,
        quantScale1, quantOffset1,
        wuq, deScale1, bias1,
        gamma2, cos, sin, wuk,
        kvCache, slotMapping,
        kvCacheRope, ctkvScale, qNopeScale,
        wdqDim, qRopeDim, kRopeDim,
        epsilon,
        qRotaryCoeff, kRotaryCoeff,
        transposeWdq, transposeWuq, transposeWuk,
        cacheMode, quantMode, doRmsNorm, wdkvSplitCount
    )
