/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SWIGLU_EPILOGUE_HELPERS_HPP
#define MEGA_MOE_SWIGLU_EPILOGUE_HELPERS_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Block::detail {

// SwiGLU epilogue 各特化共享的 tile 划分与事件同步辅助函数，
// 语义与原各 BlockEpilogue 内的实现保持一致。

CATLASS_DEVICE
inline uint32_t SwigluTileCount(uint32_t branchLength, uint32_t tileLength)
{
    return (branchLength + tileLength - 1) / tileLength;
}

CATLASS_DEVICE
inline uint32_t SwigluTileLength(uint32_t branchLength, uint32_t tileOffset, uint32_t tileLength)
{
    uint32_t remaining = branchLength - tileOffset;
    return remaining < tileLength ? remaining : tileLength;
}

CATLASS_DEVICE
inline uint32_t SwigluNextStage(uint32_t stageId, uint32_t ubStages)
{
    return stageId + 1 < ubStages ? stageId + 1 : 0;
}

// 将 blockM 行按 epilogue 参与核数划分任务区间；搬数核返回 false。
CATLASS_DEVICE
inline bool GetEpilogueTaskRange(uint32_t blockM, uint32_t epilogueCoreNum, uint32_t &loopStartIdx,
                                 uint32_t &tasksForIdx)
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

CATLASS_DEVICE
inline void WaitScaleBuffer(uint32_t scaleStage)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(scaleStage);
}

CATLASS_DEVICE
inline void DrainScaleBuffers(uint32_t scaleBatchCount, uint32_t scaleBufferCount)
{
    uint32_t pendingCount = scaleBatchCount < scaleBufferCount ? scaleBatchCount : scaleBufferCount;
    uint32_t firstPendingBatch = scaleBatchCount - pendingCount;
    for (uint32_t i = 0; i < pendingCount; ++i) {
        WaitScaleBuffer((firstPendingBatch + i) % scaleBufferCount);
    }
}

// 将累积的 per-token scale 批量写回 GM；调用方保证事件同步语义不变。
template <class LayoutScale, class CopyOp, class ElementScale>
CATLASS_DEVICE void FlushScaleBatch(CopyOp &copyOp, AscendC::GlobalTensor<ElementScale> const &gmPerTokenScale,
                                    AscendC::LocalTensor<ElementScale> const &ubScaleOutput, uint32_t gmOffset,
                                    uint32_t scaleCount, uint32_t scaleStage)
{
    if (scaleCount == 0) {
        return;
    }
    LayoutScale scaleLayout{scaleCount};
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scaleStage);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scaleStage);
    copyOp(gmPerTokenScale[gmOffset], ubScaleOutput, scaleLayout, scaleLayout);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(scaleStage);
}

} // namespace Catlass::Epilogue::Block::detail

#endif // MEGA_MOE_SWIGLU_EPILOGUE_HELPERS_HPP
