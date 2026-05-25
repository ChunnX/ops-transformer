/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_lse_copy.h
 * \brief Flash Attention专用LSE拷贝函数，使用tilingData直接获取参数
 */

#ifndef FLASH_ATTN_LSE_COPY_H_
#define FLASH_ATTN_LSE_COPY_H_

#include "kernel_operator.h"

namespace FlashAttnLseCopy {

constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = 8;

template <typename T, typename TILING_DATA_T>
__aicore__ inline void DataCopyLseBSND(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                       uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                       const TILING_DATA_T *tilingData, uint64_t s1LeftPaddingSize = 0)
{
    uint32_t startS1Idx = mOffset / tilingData->flashAttnBaseParams.gSize;
    uint32_t startGIdx = mOffset % tilingData->flashAttnBaseParams.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / tilingData->flashAttnBaseParams.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % tilingData->flashAttnBaseParams.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * tilingData->flashAttnBaseParams.s1Size + s1Idx + s1LeftPaddingSize;
        if (s1Idx != endS1Idx) {
            curDealRowCount = tilingData->flashAttnBaseParams.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (tilingData->flashAttnBaseParams.s1Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename TILING_DATA_T>
__aicore__ inline void DataCopyLseBNSD(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                       uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                       const TILING_DATA_T *tilingData,
                                       uint64_t qActSeqLens, uint64_t s1LeftPaddingSize = 0)
{
    uint64_t gOffset = mOffset / qActSeqLens * tilingData->flashAttnBaseParams.s1Size;
    uint64_t seqOffset = mOffset % qActSeqLens;
    uint64_t outOffset = bN2Offset + gOffset + seqOffset + s1LeftPaddingSize;
    uint64_t ubOffset = 0;

    if ((qActSeqLens - seqOffset) >= dealCount) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        return;
    }

    uint64_t headActSeq = qActSeqLens - seqOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = headActSeq;
    dataCopyParams.blockLen = sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    outOffset += tilingData->flashAttnBaseParams.s1Size - qActSeqLens + headActSeq;
    ubOffset += headActSeq * FP32_BLOCK_ELEMENT_NUM;

    uint64_t pendingCount = dealCount - headActSeq;
    while (pendingCount > qActSeqLens) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = qActSeqLens;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        outOffset += tilingData->flashAttnBaseParams.s1Size;
        ubOffset += qActSeqLens * FP32_BLOCK_ELEMENT_NUM;
        pendingCount -= qActSeqLens;
    }

    if (pendingCount > 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = pendingCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    }
}

template <typename T, typename TILING_DATA_T>
__aicore__ inline void DataCopyLseTNDtoNT(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                          uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                          const TILING_DATA_T *tilingData)
{
    uint32_t startS1Idx = mOffset / tilingData->flashAttnBaseParams.gSize;
    uint32_t startGIdx = mOffset % tilingData->flashAttnBaseParams.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / tilingData->flashAttnBaseParams.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % tilingData->flashAttnBaseParams.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * tilingData->flashAttnBaseParams.t1Size + s1Idx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = tilingData->flashAttnBaseParams.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (tilingData->flashAttnBaseParams.t1Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, typename TILING_DATA_T>
__aicore__ inline void DataCopyLseNTD(GlobalTensor<float> softmaxLseGm, LocalTensor<T> lseSrc,
                                      uint64_t bN2Offset, uint32_t mOffset, uint32_t dealCount,
                                      const TILING_DATA_T *tilingData, uint32_t s1Size)
{
    uint32_t startS1Idx = mOffset % s1Size;
    uint32_t startGIdx = mOffset / s1Size;
    uint32_t endS1Idx = (mOffset + dealCount - 1) % s1Size;
    uint32_t endGIdx = (mOffset + dealCount - 1) / s1Size;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t gIdx = startGIdx; gIdx <= endGIdx; gIdx++) {
        outOffset = bN2Offset + startS1Idx * tilingData->flashAttnBaseParams.n2Size * tilingData->flashAttnBaseParams.gSize + gIdx;
        if (gIdx != endGIdx) {
            curDealRowCount = s1Size - startS1Idx;
        } else {
            curDealRowCount = endS1Idx + 1 - startS1Idx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (tilingData->flashAttnBaseParams.gSize * tilingData->flashAttnBaseParams.n2Size - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startS1Idx = 0;
        ubOffset += curDealRowCount * FP32_BLOCK_ELEMENT_NUM;
    }
}

}

#endif // FLASH_ATTN_LSE_COPY_H_