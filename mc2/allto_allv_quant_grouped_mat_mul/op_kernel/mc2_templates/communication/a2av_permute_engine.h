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
__aicore__ inline void TileDataCopyLoopDoubleBuf(
    LocalTensor<ElemType> &ubBufA,
    LocalTensor<ElemType> &ubBufB,
    GlobalTensor<ElemType> &srcBuffer,
    GlobalTensor<ElemType> &dstBuffer,
    uint64_t srcBaseOffset, uint64_t dstBaseOffset, uint64_t totalCnt,
    uint64_t bufferLen, TPipe *pipe)
{
    uint64_t tileNum = CeilDiv(totalCnt, bufferLen);
    if (tileNum == 0UL) {
        return;
    }

    DataCopyParams dataCopyParams{1U, static_cast<uint16_t>(0U), 0U, 0U};
    DataCopyPadParams dataCopyPadParams{false, 0U, 0U, 0U};

    int32_t prevMte3EventID = -1;

    for (uint64_t tile = 0UL; tile < tileNum; tile++) {
        uint64_t realLength = bufferLen;
        if (tile == tileNum - 1UL) {
            realLength = totalCnt - tile * bufferLen;
        }
        uint64_t elemOffset = CeilDiv(tile * bufferLen, sizeof(ElemType));
        dataCopyParams.blockLen = static_cast<uint16_t>(realLength);

        LocalTensor<ElemType> &curBuf = (tile % 2UL == 0UL) ? ubBufA : ubBufB;

        DataCopyPad(curBuf, srcBuffer[srcBaseOffset + elemOffset], dataCopyParams, dataCopyPadParams);

        int32_t mte2EventID = static_cast<int32_t>(pipe->FetchEventID<HardEvent::MTE2_MTE3>());
        SetFlag<HardEvent::MTE2_MTE3>(mte2EventID);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2EventID);

        if (prevMte3EventID >= 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(prevMte3EventID);
            pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(prevMte3EventID);
            prevMte3EventID = -1;
        }

        DataCopyPad(dstBuffer[dstBaseOffset + elemOffset], curBuf, dataCopyParams);

        prevMte3EventID = static_cast<int32_t>(pipe->FetchEventID<HardEvent::MTE3_MTE2>());
        SetFlag<HardEvent::MTE3_MTE2>(prevMte3EventID);
    }

    if (prevMte3EventID >= 0) {
        WaitFlag<HardEvent::MTE3_MTE2>(prevMte3EventID);
        pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(prevMte3EventID);
    }
}

struct PermutePairInfo {
    uint32_t rankIndex;
    uint32_t expertIdx;
    uint32_t srcPos;
    uint32_t dstPos;
};

template <bool SrcIsExpertFirst, bool DstIsExpertFirst>
__aicore__ inline PermutePairInfo GetPermutePairInfo(
    uint32_t pairIdx, uint32_t rankDim, uint32_t startExpertIdx, uint32_t expertNum)
{
    PermutePairInfo pairInfo;
    if constexpr (!SrcIsExpertFirst) {
        pairInfo.rankIndex = pairIdx / expertNum;
        pairInfo.expertIdx = startExpertIdx + (pairIdx % expertNum);
    } else {
        pairInfo.expertIdx = startExpertIdx + (pairIdx / rankDim);
        pairInfo.rankIndex = pairIdx % rankDim;
    }

    uint32_t localExpertIdx = pairInfo.expertIdx - startExpertIdx;

    if constexpr (!SrcIsExpertFirst) {
        pairInfo.srcPos = pairInfo.rankIndex * expertNum + localExpertIdx;
    } else {
        pairInfo.srcPos = localExpertIdx * rankDim + pairInfo.rankIndex;
    }
    if constexpr (!DstIsExpertFirst) {
        pairInfo.dstPos = pairInfo.rankIndex * expertNum + localExpertIdx;
    } else {
        pairInfo.dstPos = localExpertIdx * rankDim + pairInfo.rankIndex;
    }

    return pairInfo;
}

__aicore__ inline uint64_t GetPermuteBatchTokenCount(
    const uint64_t *offsetCounts, uint32_t e, uint32_t rankDim, uint32_t startExpertIdx, uint32_t expertNum)
{
    uint32_t endExpertIdx = startExpertIdx + expertNum;
    uint64_t currentDataCount = 0UL;
    for (uint32_t expertIdxIter = startExpertIdx; expertIdxIter < endExpertIdx; expertIdxIter++) {
        for (uint32_t rankIndexIter = 0U; rankIndexIter < rankDim; rankIndexIter++) {
            currentDataCount += offsetCounts[rankIndexIter * e + expertIdxIter];
        }
    }
    return currentDataCount;
}

template <typename ElemType, bool SrcIsExpertFirst, bool DstIsExpertFirst>
__aicore__ inline void PermuteImplParallel(
    GlobalTensor<ElemType> &srcBuffer,
    GlobalTensor<ElemType> &dstBuffer,
    const uint64_t *offsetCounts, uint32_t e, uint32_t rankDim,
    uint32_t startExpertIdx, uint32_t expertNum,
    uint64_t axis, uint64_t &permuteBaseOffset,
    TBuf<QuePosition::VECIN> &permuteTBuf, TBuf<QuePosition::VECIN> &permuteTBuf2,
    uint64_t bufferLen, int32_t &eventID,
    uint32_t aivCoreNum)
{
    LocalTensor<ElemType> ubBufA = permuteTBuf.Get<ElemType>();
    LocalTensor<ElemType> ubBufB = permuteTBuf2.Get<ElemType>();
    TPipe *pipe = GetTPipePtr();

    uint32_t totalPos = rankDim * expertNum;
    uint64_t sumCntSrc[MAX_EXPERT_SIZE] = {0UL};
    uint64_t sumCntDst[MAX_EXPERT_SIZE] = {0UL};

    ComputePrefixSum<SrcIsExpertFirst>(sumCntSrc, totalPos, offsetCounts, e, startExpertIdx, expertNum, rankDim);
    ComputePrefixSum<DstIsExpertFirst>(sumCntDst, totalPos, offsetCounts, e, startExpertIdx, expertNum, rankDim);

    uint32_t myBlockIdx = GetBlockIdx();
    uint32_t totalPairs = rankDim * expertNum;
    uint32_t pairsPerCore = CeilDiv(totalPairs, aivCoreNum);
    uint32_t myStartPair = myBlockIdx * pairsPerCore;
    uint32_t myEndPair = (myStartPair + pairsPerCore > totalPairs) ? totalPairs : (myStartPair + pairsPerCore);

    for (uint32_t pairIdx = myStartPair; pairIdx < myEndPair; pairIdx++) {
        const auto pairInfo = GetPermutePairInfo<SrcIsExpertFirst, DstIsExpertFirst>(
            pairIdx, rankDim, startExpertIdx, expertNum);

        uint64_t srcOffset = permuteBaseOffset;
        if (pairInfo.srcPos != 0U) {
            srcOffset += sumCntSrc[pairInfo.srcPos - 1U] * axis;
        }
        uint64_t dstOffset = permuteBaseOffset;
        if (pairInfo.dstPos != 0U) {
            dstOffset += sumCntDst[pairInfo.dstPos - 1U] * axis;
        }

        uint64_t totalCnt = static_cast<uint64_t>(offsetCounts[pairInfo.rankIndex * e + pairInfo.expertIdx]) *
            axis * sizeof(ElemType);

        if (totalCnt == 0UL) {
            continue;
        }

        TileDataCopyLoopDoubleBuf<ElemType>(
            ubBufA, ubBufB, srcBuffer, dstBuffer,
            srcOffset, dstOffset, totalCnt,
            bufferLen, pipe);
    }

    uint64_t currentDataCount = GetPermuteBatchTokenCount(offsetCounts, e, rankDim, startExpertIdx, expertNum);
    permuteBaseOffset += currentDataCount * axis;
}

} // namespace MC2KernelTemplate

#endif // A2AV_PERMUTE_ENGINE_H
