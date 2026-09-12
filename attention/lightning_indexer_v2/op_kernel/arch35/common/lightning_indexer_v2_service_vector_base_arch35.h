/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_BASE_ARCH35_H

#include "lightning_indexer_v2_base_arch35.h"

namespace LIV2Common {

template <typename SCORE_T>
__aicore__ inline void InitVecWorkspaceTensor(AscendC::GlobalTensor<SCORE_T> &scoreTensor,
    AscendC::GlobalTensor<SCORE_T> &ldScoreTensor, AscendC::GlobalTensor<int32_t> &ldIndexTensor,
    AscendC::GlobalTensor<SCORE_T> scoreGm, AscendC::GlobalTensor<SCORE_T> ldScoreGm,
    AscendC::GlobalTensor<int32_t> ldIndexGm)
{
    scoreTensor = scoreGm;
    ldScoreTensor = ldScoreGm;
    ldIndexTensor = ldIndexGm;
}

__aicore__ inline void CleanInvalidIndices(AscendC::GlobalTensor<int32_t> &indiceOutGm, int32_t invalidIdx,
    int64_t offset, uint64_t size)
{
    AscendC::GlobalTensor<int32_t> output = indiceOutGm[offset];
    AscendC::InitGlobalMemory(output, size, invalidIdx);
}

template <typename ValueBits, typename ValueOutT, typename RunInfoT>
__aicore__ inline void DoTndPadding(const RunInfoT &runInfo, uint64_t topk, bool returnValue,
    ValueBits negativeInfinity, uint32_t kHeadNum, int32_t invalidIdx, uint32_t mte3VEvent,
    AscendC::GlobalTensor<int32_t> &indiceOutGm, AscendC::GlobalTensor<ValueOutT> &valueOutGm)
{
    uint32_t paddingLen = runInfo.curCuSeqlensQ - runInfo.curSequsedQ;
    uint64_t paddingOffset = runInfo.indiceOutOffset + runInfo.curSequsedQ * kHeadNum * topk;
    uint64_t dealSize = paddingLen * kHeadNum * topk;
    CleanInvalidIndices(indiceOutGm, invalidIdx, paddingOffset, dealSize);
    if (returnValue) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
        AscendC::GlobalTensor<ValueBits> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ ValueBits *)valueOutGm.GetPhyAddr());
        AscendC::GlobalTensor<ValueBits> valueOut = valueOutGmTmp[paddingOffset];
        AscendC::InitGlobalMemory(valueOut, dealSize, negativeInfinity);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent);
    }
}

template <typename LdInfoT>
__aicore__ inline uint64_t GetLdGmOffset(const LdInfoT &ldInfo, uint32_t s1BaseSize, uint32_t topkCountAlign16,
    uint32_t row, uint32_t block, uint32_t workspaceCount)
{
    return ldInfo.workspaceIdx * s1BaseSize * topkCountAlign16 + topkCountAlign16 * (ldInfo.mStart + row) +
           block * workspaceCount * s1BaseSize * topkCountAlign16;
}

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_SERVICE_VECTOR_BASE_ARCH35_H
