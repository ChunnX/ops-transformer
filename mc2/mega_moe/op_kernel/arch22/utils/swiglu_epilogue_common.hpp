/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SWIGLU_EPILOGUE_COMMON_HPP
#define MEGA_MOE_SWIGLU_EPILOGUE_COMMON_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Epilogue::Block::detail {

CATLASS_DEVICE bool SwigluTaskRange(uint32_t blockM, uint32_t epilogueCoreNum, uint32_t& loopStartIdx,
                                    uint32_t& tasksForIdx)
{
    uint32_t subblockIdx = get_block_idx() + get_subblockid() * get_block_num();
    uint32_t moveDataCoreNum = get_block_num() * 2 - epilogueCoreNum;
    if (subblockIdx < moveDataCoreNum) {
        return false;
    }
    uint32_t epilogueCoreIdx = subblockIdx - moveDataCoreNum;
    uint32_t perCoreData = blockM / epilogueCoreNum;
    uint32_t remainderData = blockM % epilogueCoreNum;
    tasksForIdx = epilogueCoreIdx < remainderData ? perCoreData + 1 : perCoreData;
    loopStartIdx = epilogueCoreIdx * perCoreData + (epilogueCoreIdx < remainderData ? epilogueCoreIdx : remainderData);
    return true;
}

template <uint32_t SCALE_INPUT_COUNT, class ElementPerTokenScale>
CATLASS_DEVICE ElementPerTokenScale
ReadSwigluInputScale(AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleInput,
                     AscendC::GlobalTensor<ElementPerTokenScale> const& gmPerTokenScale1, uint32_t loopIdx,
                     uint32_t loopStartIdx, uint32_t tasksForIdx)
{
    uint32_t scaleOffsetInBatch = (loopIdx - loopStartIdx) % SCALE_INPUT_COUNT;
    if (scaleOffsetInBatch == 0) {
        uint32_t remain = tasksForIdx - (loopIdx - loopStartIdx);
        uint32_t copyCount = remain >= SCALE_INPUT_COUNT ? SCALE_INPUT_COUNT :
                                                           (remain + BYTE_PER_BLK / sizeof(ElementPerTokenScale) - 1) /
                                                               (BYTE_PER_BLK / sizeof(ElementPerTokenScale)) *
                                                               (BYTE_PER_BLK / sizeof(ElementPerTokenScale));
        AscendC::DataCopy(ubPerTokenScaleInput, gmPerTokenScale1[loopIdx], copyCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    return ubPerTokenScaleInput.GetValue(scaleOffsetInBatch);
}

template <class LayoutPerTokenScale, class CopyUbToGmDequantScale>
CATLASS_DEVICE void FlushSwigluScaleBatch(AscendC::GlobalTensor<float> const& gmPerTokenScale,
                                          AscendC::LocalTensor<float> ubScale,
                                          CopyUbToGmDequantScale& copyUbToGmDequantScale, uint32_t gmOffset,
                                          uint32_t scaleCount, uint32_t scaleStage)
{
    if (scaleCount == 0) {
        return;
    }
    LayoutPerTokenScale scaleLayout{scaleCount};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scaleStage);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scaleStage);
    copyUbToGmDequantScale(gmPerTokenScale[gmOffset], ubScale, scaleLayout, scaleLayout);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(scaleStage);
}

CATLASS_DEVICE void WaitSwigluScaleBuffer(uint32_t scaleStage)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(scaleStage);
}

CATLASS_DEVICE void DrainSwigluScaleBuffers(uint32_t scaleBatchCount, uint32_t scaleBufferCount)
{
    uint32_t pendingCount = scaleBatchCount < scaleBufferCount ? scaleBatchCount : scaleBufferCount;
    uint32_t firstPendingBatch = scaleBatchCount - pendingCount;
    for (uint32_t i = 0; i < pendingCount; ++i) {
        WaitSwigluScaleBuffer((firstPendingBatch + i) % scaleBufferCount);
    }
}

template <uint32_t TILE_LENGTH>
CATLASS_DEVICE uint32_t SwigluTileCount(uint32_t branchLength)
{
    return (branchLength + TILE_LENGTH - 1) / TILE_LENGTH;
}

template <uint32_t TILE_LENGTH>
CATLASS_DEVICE uint32_t SwigluTileLength(uint32_t branchLength, uint32_t tileOffset)
{
    uint32_t remaining = branchLength - tileOffset;
    return remaining < TILE_LENGTH ? remaining : TILE_LENGTH;
}

template <uint32_t UB_STAGES>
CATLASS_DEVICE uint32_t SwigluNextStage(uint32_t stageId)
{
    return stageId + 1 < UB_STAGES ? stageId + 1 : 0;
}

CATLASS_DEVICE void ResetSwigluRowAbsMax(AscendC::LocalTensor<float> ubRowAbsMax)
{
    AscendC::Duplicate(ubRowAbsMax, 0.0f, BYTE_PER_BLK / sizeof(float));
    AscendC::PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE float ReadSwigluRowAbsMax(AscendC::LocalTensor<float> ubRowAbsMax)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    return ubRowAbsMax.GetValue(0);
}

} // namespace Catlass::Epilogue::Block::detail

#endif // MEGA_MOE_SWIGLU_EPILOGUE_COMMON_HPP
