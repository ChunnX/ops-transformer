/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H

#include "lightning_indexer_v2_base_arch35.h"

namespace LIV2Common {

__aicore__ inline uint32_t GetActualSeqLen(uint32_t batchIdx, bool hasCuSeqlens, bool hasSeqused,
    AscendC::GlobalTensor<uint32_t> &cuSeqlensGm, AscendC::GlobalTensor<uint32_t> &sequsedGm,
    uint32_t defaultSeqLen)
{
    if (hasSeqused) {
        return sequsedGm.GetValue(batchIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(batchIdx + 1) - cuSeqlensGm.GetValue(batchIdx);
    } else {
        return defaultSeqLen;
    }
}

template <typename RatioT>
__aicore__ inline uint32_t GetMaskedS2BaseBlockNum(uint32_t s1gIdx, uint32_t actS1Size,
    uint32_t actS2SizeOrig, uint32_t s1BaseSize, RatioT cmpRatio, uint32_t s2BaseSize,
    uint32_t *validLength = nullptr)
{
    if (actS2SizeOrig / cmpRatio == 0) {
        return 0;
    }
    uint32_t s1Offset = s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase +
                         static_cast<int32_t>(s1BaseSize)) / static_cast<int32_t>(cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / cmpRatio);
    validS2Len = Max(validS2Len, 1);
    if (validLength != nullptr) {
        *validLength = validS2Len;
    }
    return (validS2Len + s2BaseSize - 1) / s2BaseSize;
}

// The derived kernel owns operator-specific state; the base adds no data members.
template <typename Derived>
class LightningIndexerV2KernelBase {
protected:
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx)
    {
        auto &kernel = static_cast<Derived &>(*this);
        kernel.tempLoopInfo.bN2Idx = bN2Idx;
        kernel.tempLoopInfo.bIdx = bN2Idx / kernel.constInfo.kHeadNum;
        kernel.tempLoopInfo.n2Idx = bN2Idx % kernel.constInfo.kHeadNum;
    }

    __aicore__ inline void InitBuffers()
    {
        auto &kernel = static_cast<Derived &>(*this);
        if ASCEND_IS_AIV {
            kernel.vectorService.InitBuffers(kernel.pipe);
        } else {
            kernel.matmulService.InitBuffers(kernel.pipe);
        }
    }
};

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H
