/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SFA_ARCH35_SPLIT_KV_H
#define SFA_ARCH35_SPLIT_KV_H

#include "../../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"

namespace AttentionCommon {
constexpr int64_t FD_BROADCAST_ELEMS_PER_ROW = 8;
constexpr int64_t FD_REDUCE_CHUNK_ROWS = 16;

struct S2SplitFdStagingLayout {
    int64_t stagingM;
    int64_t dAlign;
    int64_t slotCount;
    int64_t broadcastElems;
    int64_t chunkRows;

    __aicore__ inline int64_t StagingAttenOutElems() const
    {
        return stagingM * dAlign;
    }
    __aicore__ inline int64_t StagingMaxSumBytes() const
    {
        return stagingM * broadcastElems * sizeof(float);
    }
    __aicore__ inline __gm__ uint8_t* AttenOutRegion(__gm__ uint8_t* base) const
    {
        return base;
    }
    __aicore__ inline __gm__ uint8_t* MaxRegion(__gm__ uint8_t* base) const
    {
        return base + slotCount * StagingAttenOutElems() * sizeof(float);
    }
    __aicore__ inline __gm__ uint8_t* SumRegion(__gm__ uint8_t* base) const
    {
        return MaxRegion(base) + slotCount * StagingMaxSumBytes();
    }
};

__aicore__ inline void StageVec1Lse(const S2SplitFdStagingLayout& layout, __gm__ uint8_t* stagingBase, int64_t slot,
                                    int64_t rowOffset, int64_t rows, LocalTensor<float>& maxUb,
                                    LocalTensor<float>& sumUb, LocalTensor<float>& tmpUb, uint8_t vToMte3,
                                    uint8_t mte3ToV)
{
    LocalTensor<float> tmpMax = tmpUb;
    LocalTensor<float> tmpSum = tmpUb[1024 / sizeof(float)];
    int64_t repeats = (rows + layout.broadcastElems - 1) / layout.broadcastElems;
    Brcb(tmpMax, maxUb, repeats, {1, static_cast<uint16_t>(layout.broadcastElems)});
    Brcb(tmpSum, sumUb, repeats, {1, static_cast<uint16_t>(layout.broadcastElems)});
    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float*)layout.MaxRegion(stagingBase));
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float*)layout.SumRegion(stagingBase));
    int64_t offset = slot * layout.stagingM * layout.broadcastElems + rowOffset * layout.broadcastElems;
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(layout.broadcastElems * sizeof(float)),
                             0, 0, 0};
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopyPad(sumGm[offset], tmpSum, params);
    DataCopyPad(maxGm[offset], tmpMax, params);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

template <typename T>
__aicore__ inline void StageVec2PartialOAndWait(const S2SplitFdStagingLayout& layout, GlobalTensor<float>& outGm,
                                                int64_t slot, int64_t rowOffset, int64_t rows, int64_t dValid,
                                                LocalTensor<T>& outUb, uint8_t vToMte3, uint8_t mte3ToV)
{
    int64_t offset = slot * layout.StagingAttenOutElems() + rowOffset * dValid;
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(dValid * sizeof(float)),
                             static_cast<uint16_t>((layout.dAlign - dValid) / 8), 0, 0};
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopyPad(outGm[offset], outUb, params);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

template <typename T, int64_t D_ALIGN>
__aicore__ inline void Reduce(const S2SplitFdStagingLayout& layout, __gm__ uint8_t* stagingBase, int64_t firstSlot,
                              int64_t splitNum, int64_t rowOffset, int64_t rows, int64_t dValid, LocalTensor<T>& accum,
                              LocalTensor<float>& lseExp, LocalTensor<float>& blockMax, LocalTensor<float>& blockSum,
                              LocalTensor<T>& partial, uint8_t vToMte2Stats, uint8_t vToMte2Out, uint8_t mte2ToV)
{
    GlobalTensor<float> maxGm;
    maxGm.SetGlobalBuffer((__gm__ float*)layout.MaxRegion(stagingBase));
    GlobalTensor<float> sumGm;
    sumGm.SetGlobalBuffer((__gm__ float*)layout.SumRegion(stagingBase));
    GlobalTensor<float> outGm;
    outGm.SetGlobalBuffer((__gm__ float*)layout.AttenOutRegion(stagingBase));
    LocalTensor<T> unusedSink;
    for (int64_t startRow = 0; startRow < rows; startRow += layout.chunkRows) {
        int64_t chunkRows = (rows - startRow < layout.chunkRows) ? rows - startRow : layout.chunkRows;
        int64_t statsElems = chunkRows * layout.broadcastElems;
        int64_t statsOffset = (firstSlot * layout.stagingM + rowOffset + startRow) * layout.broadcastElems;
        WaitFlag<HardEvent::V_MTE2>(vToMte2Stats);
        for (int64_t split = 0; split < splitNum; ++split) {
            DataCopy(blockMax[split * statsElems], maxGm[statsOffset], statsElems);
            DataCopy(blockSum[split * statsElems], sumGm[statsOffset], statsElems);
            statsOffset += layout.stagingM * layout.broadcastElems;
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        FaVectorApi::ComputeScaleValue_VF<T, T>(unusedSink, blockMax, blockSum, lseExp, chunkRows, splitNum, false,
                                                false);
        PipeBarrier<PIPE_V>();
        int64_t outOffset = firstSlot * layout.StagingAttenOutElems() + (rowOffset + startRow) * dValid;
        DataCopyExtParams params{static_cast<uint16_t>(chunkRows), static_cast<uint32_t>(dValid * sizeof(float)), 0,
                                 static_cast<uint16_t>((D_ALIGN - dValid) / 8), 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((D_ALIGN - dValid) % 8), 0};
        LocalTensor<T> chunkAccum = accum[startRow * D_ALIGN];
        for (int64_t split = 0; split < splitNum; ++split) {
            WaitFlag<HardEvent::V_MTE2>(vToMte2Out);
            DataCopyPad(partial, outGm[outOffset], params, pad);
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            FaVectorApi::ReduceFinalRes_const_VF<T, D_ALIGN>(chunkAccum, blockSum, partial, chunkRows, split);
            SetFlag<HardEvent::V_MTE2>(vToMte2Out);
            outOffset += layout.StagingAttenOutElems();
        }
        SetFlag<HardEvent::V_MTE2>(vToMte2Stats);
    }
}
} // namespace AttentionCommon
#endif // SFA_ARCH35_SPLIT_KV_H
