/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A2AV_PERMUTE_ENGINE_H
#define A2AV_PERMUTE_ENGINE_H

#include "../a2av_gmm_utils.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../common/a2av_common_tiling.h"

using namespace AscendC;

namespace MC2KernelTemplate {

template <bool IsExpertFirst>
__aicore__ inline void ComputePrefixSum(
    uint64_t *sumCnt, uint32_t totalPos,
    const uint64_t *offsetCounts, uint32_t e,
    uint32_t startExpertIdx, uint32_t expertNum, uint32_t rankDim)
{
    for (uint32_t pos = 0; pos < totalPos; pos++) {
        uint32_t rankIndex, localExpertIdx;
        if constexpr (!IsExpertFirst) {
            rankIndex = pos / expertNum;
            localExpertIdx = pos % expertNum;
        } else {
            localExpertIdx = pos / rankDim;
            rankIndex = pos % rankDim;
        }
        uint32_t globalExpertIdx = startExpertIdx + localExpertIdx;
        sumCnt[pos] = static_cast<uint64_t>(offsetCounts[rankIndex * e + globalExpertIdx]);
        if (pos != 0U) {
            sumCnt[pos] += sumCnt[pos - 1U];
        }
    }
}

template <typename ElemType>
__aicore__ inline void TileDataCopyLoop(
    LocalTensor<ElemType> &ubTensor,
    GlobalTensor<ElemType> &srcBuffer,
    GlobalTensor<ElemType> &dstBuffer,
    uint64_t srcBaseOffset, uint64_t dstBaseOffset, uint64_t totalCnt,
    uint64_t bufferLen, TPipe *pipe, int32_t &eventID)
{
    uint64_t tileNum = CeilDiv(totalCnt, bufferLen);
    for (uint64_t tile = 0UL; tile < tileNum; tile++) {
        uint64_t realLength = bufferLen;
        if (tile == tileNum - 1UL) {
            realLength = totalCnt - tile * bufferLen;
        }
        uint64_t elemOffset = CeilDiv(tile * bufferLen, sizeof(ElemType));
        DataCopyParams dataCopyParams{1U, static_cast<uint16_t>(realLength), 0U, 0U};
        DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};

        DataCopyPad(ubTensor, srcBuffer[srcBaseOffset + elemOffset], dataCopyParams, dataCopyPadParams);

        eventID = static_cast<int32_t>(pipe->FetchEventID<HardEvent::MTE2_MTE3>());
        SetFlag<HardEvent::MTE2_MTE3>(eventID);
        WaitFlag<HardEvent::MTE2_MTE3>(eventID);

        DataCopyPad(dstBuffer[dstBaseOffset + elemOffset], ubTensor, dataCopyParams);

        eventID = static_cast<int32_t>(pipe->FetchEventID<HardEvent::MTE3_MTE2>());
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    }
}

template <typename ElemType, bool SrcIsExpertFirst, bool DstIsExpertFirst>
__aicore__ inline void PermuteImpl(
    GlobalTensor<ElemType> &srcBuffer,
    GlobalTensor<ElemType> &dstBuffer,
    const uint64_t *offsetCounts, uint32_t e, uint32_t rankDim,
    uint32_t startExpertIdx, uint32_t expertNum,
    uint64_t axis, uint64_t &permuteBaseOffset,
    uint64_t bufferLen, int32_t &eventID)
{
    TPipe *pipe = GetTPipePtr();
    pipe->Reset();
    TBuf<QuePosition::VECIN> tBuf;
    pipe->InitBuffer(tBuf, bufferLen);
    LocalTensor<ElemType> ubTensor = tBuf.Get<ElemType>();

    uint32_t totalPos = rankDim * expertNum;
    uint64_t sumCntSrc[MAX_EXPERT_SIZE] = {0UL};
    uint64_t sumCntDst[MAX_EXPERT_SIZE] = {0UL};

    ComputePrefixSum<SrcIsExpertFirst>(sumCntSrc, totalPos, offsetCounts, e, startExpertIdx, expertNum, rankDim);
    ComputePrefixSum<DstIsExpertFirst>(sumCntDst, totalPos, offsetCounts, e, startExpertIdx, expertNum, rankDim);

    uint32_t endExpertIdx = startExpertIdx + expertNum;
    for (uint32_t rankIndex = 0U; rankIndex < rankDim; rankIndex++) {
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
            uint32_t localExpertIdx = expertIdx - startExpertIdx;

            uint32_t srcPos, dstPos;
            if constexpr (!SrcIsExpertFirst) {
                srcPos = rankIndex * expertNum + localExpertIdx;
            } else {
                srcPos = localExpertIdx * rankDim + rankIndex;
            }
            if constexpr (!DstIsExpertFirst) {
                dstPos = rankIndex * expertNum + localExpertIdx;
            } else {
                dstPos = localExpertIdx * rankDim + rankIndex;
            }

            uint64_t srcOffset = permuteBaseOffset;
            if (srcPos != 0U) {
                srcOffset += sumCntSrc[srcPos - 1U] * axis;
            }
            uint64_t dstOffset = permuteBaseOffset;
            if (dstPos != 0U) {
                dstOffset += sumCntDst[dstPos - 1U] * axis;
            }

            uint64_t totalCnt = static_cast<uint64_t>(offsetCounts[rankIndex * e + expertIdx]) * axis * sizeof(ElemType);

            TileDataCopyLoop<ElemType>(
                ubTensor, srcBuffer, dstBuffer,
                srcOffset, dstOffset, totalCnt,
                bufferLen, pipe, eventID);
        }
    }

    uint64_t currentDataCount = 0UL;
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; expertIdx++) {
        for (uint32_t rankIndex = 0U; rankIndex < rankDim; rankIndex++) {
            currentDataCount += offsetCounts[rankIndex * e + expertIdx];
        }
    }
    permuteBaseOffset += currentDataCount * axis;
}

} // namespace MC2KernelTemplate

#endif // A2AV_PERMUTE_ENGINE_H