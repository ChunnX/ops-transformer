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
 * \file flash_decode_s2_split.h
 * \brief S2-split Flash Decode data-plane: staging layout, Vec1 LSE staging,
 *        Vec2 partial-O staging, and FD chunk reduction.
 *
 * GM staging layout (three contiguous regions):
 *   [partial O slots][max slots][sum slots]
 *
 * Per-slot sizes:
 *   partial O : stagingM * dAlign * sizeof(float)
 *   max       : stagingM * broadcastElems * sizeof(float)   (each row is broadcastElems identical floats)
 *   sum       : stagingM * broadcastElems * sizeof(float)
 *
 * Numerics:
 *   Vec2 must write normalized p_i = sum(exp(score-m_i)*V) / s_i (after final division).
 *   FD Reduce computes M = max(m_i), G = sum(exp(m_i-M)*s_i), w_i = exp(m_i-M)*s_i/G, O = sum(w_i*p_i).
 *
 * Synchronization:
 *   - All participating cores must enter SyncAll() after FA completes before FD reads staging.
 *   - SyncAll() is owned by the operator kernel, not this file.
 *   - Event IDs (V_MTE3, MTE3_V, V_MTE2, MTE2_V) are passed in by the caller.
 *
 * workspaceNum limit: currently maxSplits = 2. Caller must ensure 1 <= workspaceNum <= maxSplits.
 */
#ifndef FLASH_DECODE_S2_SPLIT_H
#define FLASH_DECODE_S2_SPLIT_H

#include <stdint.h>

namespace AttentionCommon {

// The three regions are contiguous: partial O, max, then sum.
// slotCount = maxSplits * physicalCoreSlots (already includes maxSplits).
// broadcastElems: each max/sum row is stored as broadcastElems identical floats (currently 8).
// chunkRows: FD reduction chunk width (currently 16).
struct S2SplitFdStagingLayout {
    uint32_t stagingM;
    uint32_t dAlign;
    uint32_t slotCount;
    uint32_t broadcastElems;
    uint32_t chunkRows;

    __aicore__ inline uint32_t StagingAttenOutBytes() const
    {
        return stagingM * dAlign * sizeof(float);
    }

    __aicore__ inline uint32_t StagingMaxSumBytes() const
    {
        return stagingM * broadcastElems * sizeof(float);
    }

    __aicore__ inline __gm__ uint8_t *AttenOutRegion(__gm__ uint8_t *base) const
    {
        return base;
    }

    __aicore__ inline __gm__ uint8_t *MaxRegion(__gm__ uint8_t *base) const
    {
        return AttenOutRegion(base) + slotCount * StagingAttenOutBytes();
    }

    __aicore__ inline __gm__ uint8_t *SumRegion(__gm__ uint8_t *base) const
    {
        return MaxRegion(base) + slotCount * StagingMaxSumBytes();
    }
};

// Stage Vec1 max/sum to GM staging.
// tmpUb must hold at least 2 * stagingM * broadcastElems floats (max + sum broadcast blocks).
__aicore__ inline void StageVec1Lse(const S2SplitFdStagingLayout &layout,
    __gm__ uint8_t *stagingBase, uint32_t workspaceIdx, uint32_t stagingMOffset,
    uint32_t validRows, LocalTensor<float> &maxUb, LocalTensor<float> &sumUb,
    LocalTensor<float> &tmpUb, uint8_t vToMte3Id, uint8_t mte3ToVId)
{
    __gm__ uint8_t *maxRegion = layout.MaxRegion(stagingBase);
    __gm__ uint8_t *sumRegion = layout.SumRegion(stagingBase);
    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float *)maxRegion);
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float *)sumRegion);

    LocalTensor<float> tmpMaxBlockUb = tmpUb;
    LocalTensor<float> tmpSumBlockUb = tmpUb[1024 / sizeof(float)];
    uint32_t mSizeAlign8 = (validRows + layout.broadcastElems - 1) /
        layout.broadcastElems * layout.broadcastElems;
    uint32_t brcbRepeat = mSizeAlign8 / layout.broadcastElems;
    Brcb(tmpMaxBlockUb, maxUb, brcbRepeat, {1, layout.broadcastElems});
    Brcb(tmpSumBlockUb, sumUb, brcbRepeat, {1, layout.broadcastElems});
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);

    uint32_t MAX_SUM_BYTES = layout.StagingMaxSumBytes();
    uint32_t floatOffset = workspaceIdx * (MAX_SUM_BYTES / sizeof(float)) +
        stagingMOffset * layout.broadcastElems;
    DataCopyExtParams lseOutParams{static_cast<uint16_t>(validRows),
        static_cast<uint32_t>(layout.broadcastElems * sizeof(float)), 0, 0, 0};
    DataCopyPad(sumGm[floatOffset], tmpSumBlockUb, lseOutParams);
    DataCopyPad(maxGm[floatOffset], tmpMaxBlockUb, lseOutParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
}

// Stage Vec2 normalized partial O to GM staging.
// vec2ResUb must contain FP32 partial O after final division.
template <typename T>
__aicore__ inline void StageVec2PartialO(const S2SplitFdStagingLayout &layout,
    __gm__ uint8_t *stagingBase, uint32_t workspaceIdx, uint32_t stagingMOffset,
    uint32_t validRows, uint32_t dValid, LocalTensor<T> &vec2ResUb,
    uint8_t vToMte3Id, uint8_t mte3ToVId)
{
    uint32_t ATTEN_OUT_BYTES = layout.StagingAttenOutBytes();
    GlobalTensor<float> stagingOut;
    stagingOut.SetGlobalBuffer((__gm__ float *)(layout.AttenOutRegion(stagingBase) +
        workspaceIdx * ATTEN_OUT_BYTES));
    SetFlag<HardEvent::V_MTE3>(vToMte3Id);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id);
    DataCopyExtParams outParams;
    outParams.blockLen = dValid * sizeof(float);
    outParams.srcStride = static_cast<uint16_t>((layout.dAlign - dValid) >> 3);
    outParams.dstStride = 0;
    outParams.blockCount = validRows;
    DataCopyPad(stagingOut[stagingMOffset * dValid], vec2ResUb, outParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId);
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId);
}

// FD chunk reduction: read all splits from staging, compute weights, reduce partial O.
// D_ALIGN is a compile-time constant (e.g. 512) used by ReduceFinalRes_const_VF.
// workspaceNum must not exceed maxSplits (currently 2).
template <typename T, uint32_t D_ALIGN>
__aicore__ inline void Reduce(const S2SplitFdStagingLayout &layout,
    __gm__ uint8_t *stagingBase, uint32_t workspaceIdx, uint32_t workspaceNum,
    uint32_t fdMOffset, uint32_t mNum, uint32_t dValid,
    LocalTensor<T> &accumulatedO, LocalTensor<float> &lseExpUb,
    LocalTensor<float> &blockMaxUb, LocalTensor<float> &blockSumUb,
    LocalTensor<T> &partialOFp32,
    uint8_t vToMte2Id0, uint8_t vToMte2Id1, uint8_t mte2ToVId)
{
    uint32_t ATTEN_OUT_BYTES = layout.StagingAttenOutBytes();
    uint32_t MAX_SUM_BYTES = layout.StagingMaxSumBytes();
    __gm__ uint8_t *maxRegion = layout.MaxRegion(stagingBase);
    __gm__ uint8_t *sumRegion = layout.SumRegion(stagingBase);

    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float *)(maxRegion + workspaceIdx * MAX_SUM_BYTES));
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float *)(sumRegion + workspaceIdx * MAX_SUM_BYTES));
    uint32_t splitStride = MAX_SUM_BYTES / sizeof(float);
    GlobalTensor<float> stagingOutGm;
    stagingOutGm.SetGlobalBuffer((__gm__ float *)(layout.AttenOutRegion(stagingBase) +
        workspaceIdx * ATTEN_OUT_BYTES));
    uint32_t outSplitStride = ATTEN_OUT_BYTES / sizeof(float);

    uint32_t mChunks = (mNum + layout.chunkRows - 1) / layout.chunkRows;
    for (uint32_t chunkIdx = 0; chunkIdx < mChunks; chunkIdx++) {
        uint32_t startRow = chunkIdx * layout.chunkRows;
        uint32_t dealRowCount = layout.chunkRows;
        if (startRow + dealRowCount > mNum) {
            dealRowCount = mNum - startRow;
        }

        WaitFlag<HardEvent::V_MTE2>(vToMte2Id0);
        for (uint32_t splitIdx = 0; splitIdx < workspaceNum; splitIdx++) {
            uint32_t srcOffset = splitIdx * splitStride +
                (fdMOffset + startRow) * layout.broadcastElems;
            uint32_t dstOffset = splitIdx * dealRowCount * layout.broadcastElems;
            DataCopy(blockMaxUb[dstOffset], maxGm[srcOffset],
                dealRowCount * layout.broadcastElems);
            DataCopy(blockSumUb[dstOffset], sumGm[srcOffset],
                dealRowCount * layout.broadcastElems);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVId);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVId);

        LocalTensor<T> sinkUb;
        ComputeScaleValue_VF<T, T>(sinkUb, blockMaxUb, blockSumUb, lseExpUb,
                                   dealRowCount, workspaceNum, false, false);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> chunkAccumO = accumulatedO[startRow * D_ALIGN];
        for (uint32_t splitIdx = 0; splitIdx < workspaceNum; splitIdx++) {
            DataCopyExtParams inParams;
            inParams.blockLen = dValid * sizeof(float);
            inParams.srcStride = 0;
            constexpr uint32_t outputElemsPerBlock = BUFFER_SIZE_BYTE_32B / sizeof(float);
            inParams.dstStride = static_cast<uint16_t>((D_ALIGN - dValid) / outputElemsPerBlock);
            inParams.blockCount = dealRowCount;
            DataCopyPadExtParams<float> padParams{true, 0,
                static_cast<uint8_t>((D_ALIGN - dValid) % outputElemsPerBlock), 0};
            uint32_t outSrcOffset = splitIdx * outSplitStride + (fdMOffset + startRow) * dValid;
            WaitFlag<HardEvent::V_MTE2>(vToMte2Id1);
            DataCopyPad(partialOFp32, stagingOutGm[outSrcOffset], inParams, padParams);
            SetFlag<HardEvent::MTE2_V>(mte2ToVId);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVId);
            ReduceFinalRes_const_VF<T, D_ALIGN>(chunkAccumO, blockSumUb, partialOFp32,
                                                         dealRowCount, splitIdx);
            SetFlag<HardEvent::V_MTE2>(vToMte2Id1);
        }

        SetFlag<HardEvent::V_MTE2>(vToMte2Id0);
    }
}

} // namespace AttentionCommon

#endif  // FLASH_DECODE_S2_SPLIT_H
