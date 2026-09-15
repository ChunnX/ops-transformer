/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Arch22 MTE stage operations.
#ifndef STAGE_GMMA2AV_MTE_STAGE_H
#define STAGE_GMMA2AV_MTE_STAGE_H

#include "../common/gmma2av_mte_context.h"

namespace MC2KernelTemplate {

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::StageExpertPayload(uint32_t startExpertIdx, uint32_t endExpertIdx)
{
    if ASCEND_IS_AIC {
        return;
    }
    PublishLocalChunk(startExpertIdx, endExpertIdx);
}

template <typename T>
__aicore__ inline void GmmA2avMteOp<T>::PublishLocalChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
{
    const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
    const uint32_t aicCoreNum = rankWorkerCount_;
    const uint32_t tileRows = Gmma2avMteTiling::L1_TILE_M;
    for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
        // Mirror the AIC's GetExpertTokens exactly (raw sum, no filter):
        // the tile count and the rotation must match the producer side.
        uint64_t expertRows = 0UL;
        for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
            expertRows += static_cast<uint64_t>(taskTilingInfo_->sendCnt[dstRank * expertNum + expertIdx]);
        }
        const uint32_t loopsM = static_cast<uint32_t>((expertRows + tileRows - 1UL) / tileRows);
        if (aicCoreNum != 0U && expertRows > 0UL) {
            const uint32_t startMIdx = (logicalCoreIdx_ + aicCoreNum - expertStartCoreIdx_) % aicCoreNum;
            uint32_t ownedTileIndex = 0U;
            for (uint32_t mIdx = startMIdx; mIdx < loopsM; mIdx += aicCoreNum, ++ownedTileIndex) {
                if (ownedTileIndex % Gmma2avMteTiling::M_GROUPS_PER_WAVE == 0U) {
                    WaitComputeProgress(++computeWaveSequence_);
                }
                const uint64_t tileRowStart = static_cast<uint64_t>(mIdx) * tileRows;
                const uint64_t remaining = expertRows - tileRowStart;
                const uint64_t tileRowCount = remaining > tileRows ? tileRows : remaining;
                const uint64_t sourceRow = localExpertBaseRows_ + tileRowStart;
                CopyGmToGm(sendBuffer_ + sourceRow * taskTilingInfo_->N1,
                           GetOutputData(rank_) + sourceRow * taskTilingInfo_->N1,
                           tileRowCount * taskTilingInfo_->N1);
            }
        }
        localExpertBaseRows_ += expertRows;
        expertStartCoreIdx_ = (expertStartCoreIdx_ + loopsM) % aicCoreNum;
    }
}

} // namespace MC2KernelTemplate

#endif // STAGE_GMMA2AV_MTE_STAGE_H
