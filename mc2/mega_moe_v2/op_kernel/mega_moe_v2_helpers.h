/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* V2 helper utilities extracted from v1 mega_moe.h: routing/cumsum + dispatch chunk pull + Unpermute */

#ifndef MEGA_MOE_V2_HELPERS_H
#define MEGA_MOE_V2_HELPERS_H

#include "kernel_operator.h"
#include "mega_moe_v2_base.h"
#include "mega_moe_v2_tiling.h"
#include "mega_moe_v2_workspace_info.h"

using namespace AscendC;

namespace MegaMoeV2Helpers {

// ---------- per-token GM->GM ping-pong copy (v1 mega_moe.h:419-456) ----------
template <typename QuantOutType, typename QuantScaleOutType>
__aicore__ inline void CopyGMToGMPerTokenV2(
    TBuf<>& copyTempBuf,
    GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale,
    GlobalTensor<QuantOutType> src,
    int32_t rows, int32_t hiddenSize, int32_t& pingpongId, int32_t dispatchRows)
{
    constexpr int32_t BufferNum = 2;
    LocalTensor<QuantOutType> tmpBuffer1 = copyTempBuf.template Get<QuantOutType>();
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[96 * 1024];
    uint32_t scaleLen = Ops::Base::CeilDiv(static_cast<int32_t>(hiddenSize),
        static_cast<int32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = hiddenSize + scaleLen;
    auto processCount = Ops::Base::CeilDiv(rows, dispatchRows);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        LocalTensor<QuantScaleOutType> bufScale = buf[hiddenSize].template ReinterpretCast<QuantScaleOutType>();

        auto inputOffset = processIndex * dispatchRows * copyInNum;
        int32_t rowNum = dispatchRows;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * dispatchRows;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        int64_t dataLen = rowNum * copyInNum;
        DataCopy(buf, src[inputOffset], dataLen);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        auto outputOffset = processIndex * dispatchRows * hiddenSize;
        DataCopyPad(dst[outputOffset], buf,
            {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(hiddenSize),
             static_cast<uint16_t>(scaleLen / ALIGN_32), 0, 0});
        DataCopyPad(dstScale[processIndex * dispatchRows * scaleLen], bufScale,
            {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(scaleLen),
             static_cast<uint16_t>(hiddenSize / ALIGN_32), 0, 0});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
}

// ---------- DispatchPullChunk (v1:459-504 reshaped to per-chunk arrival counter) ----------
// Pulls one logical chunk worth of tokens for given expert, then bumps dispatchArrival[chunkId].
template <typename QuantOutType, typename QuantScaleOutType>
__aicore__ inline void DispatchPullChunk(
    TBuf<>& copyTempBuf,
    GM_ADDR aRingGmAddr, GM_ADDR aScaleRingGmAddr,
    GlobalTensor<int32_t> tokenPerExpertWin,
    GlobalTensor<int32_t> cumsumInWorkSpace,
    GlobalTensor<int32_t> dispatchArrivalGm,    // uint32 atomic counters
    uint32_t expertIdx, uint32_t rankId,
    uint32_t worldSize, uint32_t expertPerRank,
    uint32_t blockIdx, uint32_t blockNum,
    uint32_t commVecPerRank,                    // dispatchVecPerRank
    int32_t  dispatchRows,
    int32_t& prevSum, int64_t widthA,
    uint32_t chunkId, uint32_t maxOutputSize,
    int32_t  perRankStridesInTokenPerExpert)
{
    int64_t widthAScale = Ops::Base::CeilDiv(widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (uint32_t idx = blockIdx; idx < worldSize * commVecPerRank; idx += blockNum) {
        uint32_t dstRankIdx = idx / commVecPerRank;
        uint32_t aivIdxInGroup = idx % commVecPerRank;
        uint32_t rowStartInDst = (dstRankIdx == 0 ? 0
            : cumsumInWorkSpace((dstRankIdx - 1) * expertPerRank + expertIdx));
        if (rowStartInDst < maxOutputSize) {
            int32_t rowsCopyCnt = tokenPerExpertWin(
                dstRankIdx * perRankStridesInTokenPerExpert + rankId * expertPerRank + expertIdx);
            if (rowStartInDst + rowsCopyCnt > maxOutputSize) {
                rowsCopyCnt = maxOutputSize - rowStartInDst;
            }
            int32_t rowSrcOffset = prevSum;
            prevSum += rowsCopyCnt;
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(commVecPerRank));
            int32_t rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
                GlobalTensor<QuantOutType> gmRemoteA, aGlobalTensor;
                GlobalTensor<QuantScaleOutType> aScaleGlobalTensor;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                int64_t gmOffsetAScale = rowDstOffsetPerCore * widthAScale;
                int64_t gmOffsetPeer = rowSrcOffsetPerCore * (widthA + widthAScale);

                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(GetRankWinAddrWithOffset(
                    dstRankIdx, PEERMEM_DATA_OFFSET +
                        perRankStridesInTokenPerExpert * worldSize * sizeof(uint32_t))));
                aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(aRingGmAddr));
                aScaleGlobalTensor.SetGlobalBuffer(
                    reinterpret_cast<__gm__ QuantScaleOutType*>(aScaleRingGmAddr));

                CopyGMToGMPerTokenV2<QuantOutType, QuantScaleOutType>(
                    copyTempBuf,
                    aGlobalTensor[gmOffsetA], aScaleGlobalTensor[gmOffsetAScale],
                    gmRemoteA[gmOffsetPeer], rowsPerCore, widthA, pingpongId, dispatchRows);
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        // bump per-chunk arrival counter
        AtomicAdd((__gm__ int32_t*)(dispatchArrivalGm.GetPhyAddr()) + chunkId, int32_t(1));
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// ---------- CrossRankSyncCntPerExpert (v1:348-416 verbatim, namespaced to V2) ----------
__aicore__ inline void CrossRankSyncCntPerExpert(
    TBuf<>& crossRankSyncBuf, TBuf<>& prevSumBuf, TBuf<>& cumsumBuf,
    GlobalTensor<int32_t> tokenPerExpertWin, GlobalTensor<int32_t> preSumBeforeRankWorkSpace,
    GlobalTensor<int32_t> cumsumInWorkSpace,
    uint32_t rankId, uint32_t worldSize, uint32_t expertPerRank,
    int32_t perRankStridesInTokenPerExpert,
    GM_ADDR tokenPerExpertCurRankAddr, GM_ADDR peermemBase,
    uint32_t aivCoreIdx, uint32_t blockNum, uint32_t blockIdx,
    uint32_t dispatchVecPerRank, int32_t& prevSum)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    uint32_t numPerCore = Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize * expertPerRank),
            static_cast<uint32_t>(ALIGN_128));
    LocalTensor<int32_t> tmpBuffer = crossRankSyncBuf.Get<int32_t>();
    LocalTensor<int32_t> prevSumBufLocal = prevSumBuf.Get<int32_t>();

    uint64_t offsetInDstWinAddr = perRankStridesInTokenPerExpert * sizeof(int32_t) * rankId + PEERMEM_DATA_OFFSET;
    for (int32_t dstRankIdx = aivCoreIdx; dstRankIdx < (int32_t)worldSize; dstRankIdx += blockNum) {
        if (dstRankIdx == (int32_t)rankId) {
            continue;
        }
        GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRankAddr));
        GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr);
        dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(tmpBuffer, srcAddress, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(dstAddress, tmpBuffer, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    int32_t intPer512 = ALIGN_512 / sizeof(int);
    for (int32_t dstRankIdx = aivCoreIdx; dstRankIdx < (int32_t)worldSize; dstRankIdx += blockNum) {
        if (dstRankIdx != (int32_t)rankId) {
            for (int32_t checkIdx = 0; checkIdx < (int32_t)numPerCore; checkIdx += intPer512) {
                __gm__ int32_t* syncCheck = reinterpret_cast<__gm__ int32_t*>(peermemBase) +
                    PEERMEM_DATA_OFFSET / sizeof(int32_t) +
                    dstRankIdx * perRankStridesInTokenPerExpert + checkIdx;
                // gmSignalWait inlined: spin until value becomes non-zero
                while (ReadGmByPassDCache(syncCheck) == 0) {}
            }
            DataCopy(tmpBuffer, tokenPerExpertWin[dstRankIdx * perRankStridesInTokenPerExpert], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(tokenPerExpertWin[dstRankIdx * perRankStridesInTokenPerExpert], tmpBuffer, numPerCore);
        } else {
            DataCopy(tmpBuffer, tokenPerExpertWin[dstRankIdx * perRankStridesInTokenPerExpert], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        PipeBarrier<PIPE_ALL>();
        int32_t tmpSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (int32_t)((rankId + 1) * expertPerRank); i++) {
            if (i >= (int32_t)(rankId * expertPerRank)) {
                prevSumBufLocal(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(preSumBeforeRankWorkSpace[dstRankIdx * expertPerRank], prevSumBufLocal,
                    DataCopyParams{1, static_cast<uint16_t>(expertPerRank * sizeof(int32_t)), 0, 0});
    }
    SyncAll<true>();
    if (aivCoreIdx == 0) {
        // build cumsum (mirrors v1 GetCumsumForMMAIV)
        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        LocalTensor<int32_t> tmpBuffer1 = cumsumBuf.Get<int32_t>();
        DataCopyPad(tmpBuffer1, tokenPerExpertWin[rankId * expertPerRank],
            {static_cast<uint16_t>(worldSize), static_cast<uint16_t>(expertPerRank * sizeof(int32_t)),
             static_cast<uint16_t>((Ops::Base::CeilAlign(worldSize * expertPerRank, (uint32_t)ALIGN_128)
                 - expertPerRank) * sizeof(int32_t)), 0},
            {});
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        for (uint32_t i = 1; i < worldSize; ++i) {
            Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
                tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank);
            PipeBarrier<PIPE_V>();
        }
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(cumsumInWorkSpace, tmpBuffer1,
            {static_cast<uint16_t>(worldSize), static_cast<uint16_t>(expertPerRank * sizeof(int32_t)), 0, 0});
    }
    if (blockIdx < worldSize * dispatchVecPerRank) {
        prevSum = preSumBeforeRankWorkSpace(blockIdx / dispatchVecPerRank * expertPerRank);
    }
}

// ---------- Unpermute (v1:606-657 verbatim, kept generic) ----------
template <typename TopkWeightsType>
__aicore__ inline void Unpermute(
    TBuf<>& dataResBuf, TBuf<>& dataResFp32Buf,
    LocalTensor<float>& topKWeightsTensor,
    GM_ADDR expandedRowIdxGatherAddr, GM_ADDR ptrDAddr, GM_ADDR yOutAddr,
    uint32_t m, uint32_t topK,
    uint32_t aivCoreIdx, uint32_t blockAivNum)
{
    int32_t coreIdx = aivCoreIdx;
    int32_t coreNum = blockAivNum;
    int32_t lenPerCore = Ops::Base::CeilDiv(m, static_cast<uint32_t>(coreNum));
    int32_t coreOffset = coreIdx * lenPerCore;
    int32_t coreLen = lenPerCore;
    if (coreOffset + coreLen >= (int32_t)m) coreLen = m - coreOffset;
    if (coreOffset >= (int32_t)m) coreLen = 0;

    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdxGatherAddr);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)ptrDAddr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)yOutAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        LocalTensor<bfloat16_t> dataResBf16 = dataResBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResBf16[topK];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResBf16[topK * 2];
        LocalTensor<float> dataResFp32 = dataResFp32Buf.Get<float>();
        LocalTensor<float> dataIn0Fp32 = dataResFp32[topK];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[topK * 2];
        for (int32_t expId = 0; expId < (int32_t)topK; ++expId) {
            int32_t rowIdx = expandedRowIdx(tokenIdx * topK + expId);
            float expScale = topKWeightsTensor.GetValue(tokenIdx * topK + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[rowIdx * topK], topK);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, topK);
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(dataResFp32, dataInFp32, expScale, topK);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, topK);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32, dataResFp32, dataInFp32, topK);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, topK);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(output[tokenIdx * topK], dataResBf16, topK);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

} // namespace MegaMoeV2Helpers

#endif // MEGA_MOE_V2_HELPERS_H
