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
 * \file moe_re_routing_topk_weight_common.h
 * \brief Common topk-weight GM to GM chunk copy for arch35 R/RE templates.
 */
#ifndef MOE_RE_ROUTING_TOPK_WEIGHT_COMMON_H
#define MOE_RE_ROUTING_TOPK_WEIGHT_COMMON_H

#include "kernel_operator.h"
#include "platform.h"
#include "op_kernel/math_util.h"
#include "moe_re_routing_common.h"

namespace MoeReRoutingTopk {
using namespace AscendC;

template <typename TQueBindType>
__aicore__ inline void CopyWeightChunks(GlobalTensor<float> &src, GlobalTensor<float> &dst, TQueBindType &queue,
                                        const int64_t rows, const int64_t srcOffset, const int64_t dstOffset,
                                        const int64_t ubElements)
{
    if (rows <= 0 || ubElements <= 0) {
        return;
    }
    const int64_t loopCount = Ops::Base::CeilDiv(rows, ubElements);
    for (int64_t loopIdx = 0; loopIdx < loopCount; ++loopIdx) {
        const int64_t offset = loopIdx * ubElements;
        const int64_t currentRows = (loopIdx == loopCount - 1) ? rows - offset : ubElements;
        LocalTensor<float> local = queue.template AllocTensor<float>();
        DataCopyExtParams copyParams(1, currentRows * sizeof(float), 0, 0, 0);
        DataCopyPadExtParams<float> padParams(false, 0, 0, 0);
        DataCopyPad(local, src[srcOffset + offset], copyParams, padParams);
        queue.EnQue(local);
        local = queue.template DeQue<float>();
        DataCopyPad(dst[dstOffset + offset], local, copyParams);
        queue.FreeTensor(local);
    }
}

struct TopkWeightGroupState {
    int64_t expertStart = 0;
    int64_t expertCount = 0;
    int64_t rows = 0;
    int64_t localRows = 0;
    int64_t srcOffset = 0;
};

__aicore__ inline void ResetGroup(TopkWeightGroupState &state)
{
    state.expertStart = 0;
    state.expertCount = 0;
    state.rows = 0;
    state.localRows = 0;
    state.srcOffset = 0;
}

template <typename TTokenCountTensor, typename TQueBindType>
__aicore__ inline void CopyWeightGroup(GlobalTensor<float> &srcTopkWeight, GlobalTensor<float> &dstTopkWeight,
                                       TTokenCountTensor &srcRankToken, TQueBindType &queue,
                                       LocalTensor<int64_t> &topkOffsets, const TopkWeightGroupState &state,
                                       const int64_t rankId, const int64_t expertNum, const int64_t topkBlockElements)
{
    if (state.rows <= 0 || state.expertCount <= 0) {
        return;
    }
    LocalTensor<float> topkLocal = queue.template AllocTensor<float>();
    if (state.rows == state.localRows) {
        DataCopyExtParams copyParams(1, state.rows * sizeof(float), 0, 0, 0);
        DataCopyPadExtParams<float> padParams(false, 0, 0, 0);
        DataCopyPad(topkLocal, srcTopkWeight[state.srcOffset], copyParams, padParams);
    } else {
        int64_t srcRows = 0;
        int64_t localOffset = 0;
        for (int64_t expertOffset = 0; expertOffset < state.expertCount; ++expertOffset) {
            const int64_t currRows = srcRankToken((rankId * expertNum) + state.expertStart + expertOffset);
            if (currRows > 0) {
                DataCopyExtParams copyParams(1, currRows * sizeof(float), 0, 0, 0);
                DataCopyPadExtParams<float> padParams(false, 0, 0, 0);
                DataCopyPad(topkLocal[localOffset], srcTopkWeight[state.srcOffset + srcRows], copyParams, padParams);
            }
            srcRows += currRows;
            localOffset += Ops::Base::CeilAlign(currRows, topkBlockElements);
        }
    }
    queue.EnQue(topkLocal);
    topkLocal = queue.template DeQue<float>();
    int64_t localOffset = 0;
    for (int64_t expertOffset = 0; expertOffset < state.expertCount; ++expertOffset) {
        const int64_t currRows = srcRankToken((rankId * expertNum) + state.expertStart + expertOffset);
        if (currRows > 0) {
            DataCopyExtParams copyParams(1, currRows * sizeof(float), 0, 0, 0);
            const int64_t dstOffset = topkOffsets.GetValue(expertOffset);
            DataCopyPad(dstTopkWeight[dstOffset], topkLocal[localOffset], copyParams);
        }
        localOffset += Ops::Base::CeilAlign(currRows, topkBlockElements);
    }
    queue.FreeTensor(topkLocal);
}

template <typename TTokenCountTensor, typename TQueBindType>
__aicore__ inline void AppendWeight(GlobalTensor<float> &srcTopkWeight, GlobalTensor<float> &dstTopkWeight,
                                    TTokenCountTensor &srcRankToken, TQueBindType &queue,
                                    LocalTensor<int64_t> &topkOffsets, TopkWeightGroupState &state, const int64_t rows,
                                    const int64_t srcOffset, const int64_t dstOffset, const int64_t rankId,
                                    const int64_t expertId, const int64_t expertNum, const int64_t ubElements)
{
    const int64_t topkBlockElements = platform::GetUbBlockSize() / sizeof(float);
    const int64_t localRows = Ops::Base::CeilAlign(rows, topkBlockElements);
    if (state.expertCount == 0) {
        state.expertStart = expertId;
        state.srcOffset = srcOffset;
    }
    if (state.expertCount > 0 &&
        (expertId != state.expertStart + state.expertCount || state.expertCount == TOPK_GROUP_EXPERTS ||
         state.localRows + localRows > ubElements)) {
        CopyWeightGroup(srcTopkWeight, dstTopkWeight, srcRankToken, queue, topkOffsets, state, rankId, expertNum,
                        topkBlockElements);
        ResetGroup(state);
        state.expertStart = expertId;
        state.srcOffset = srcOffset;
    }
    if (localRows > ubElements) {
        CopyWeightChunks(srcTopkWeight, dstTopkWeight, queue, rows, srcOffset, dstOffset, ubElements);
        ResetGroup(state);
        return;
    }
    topkOffsets.SetValue(state.expertCount, dstOffset);
    state.expertCount++;
    state.rows += rows;
    state.localRows += localRows;
}

template <typename TTokenCountTensor, typename TQueBindType>
__aicore__ inline void FlushWeight(GlobalTensor<float> &srcTopkWeight, GlobalTensor<float> &dstTopkWeight,
                                   TTokenCountTensor &srcRankToken, TQueBindType &queue,
                                   LocalTensor<int64_t> &topkOffsets, TopkWeightGroupState &state, const int64_t rankId,
                                   const int64_t expertNum)
{
    const int64_t topkBlockElements = platform::GetUbBlockSize() / sizeof(float);
    CopyWeightGroup(srcTopkWeight, dstTopkWeight, srcRankToken, queue, topkOffsets, state, rankId, expertNum,
                    topkBlockElements);
    ResetGroup(state);
}
} // namespace MoeReRoutingTopk

#endif // MOE_RE_ROUTING_TOPK_WEIGHT_COMMON_H
