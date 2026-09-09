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
 * \file moe_re_routing_topk_weight_kernel_common.h
 * \brief Common topk-weight GM to GM chunk copy for arch35 R/RE kernels.
 */
#ifndef MOE_RE_ROUTING_TOPK_WEIGHT_KERNEL_COMMON_H
#define MOE_RE_ROUTING_TOPK_WEIGHT_KERNEL_COMMON_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "moe_re_routing_topk_weight_common.h"

namespace MoeReRoutingTopk {
using namespace AscendC;

template <typename TQueBindType>
__aicore__ inline void CopyWeightSmall(GlobalTensor<float> &src, GlobalTensor<float> &dst, TQueBindType &queue,
                                       const int64_t rows, const int64_t srcOffset, const int64_t dstOffset)
{
    if (rows <= 0) {
        return;
    }
    const int64_t loopCount = Ops::Base::CeilDiv(rows, INDEX_UB_SIZE);
    for (int64_t loopIdx = 0; loopIdx < loopCount; ++loopIdx) {
        const int64_t offset = loopIdx * INDEX_UB_SIZE;
        const int64_t currentRows = (loopIdx == loopCount - 1) ? rows - offset : INDEX_UB_SIZE;
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

template <typename TTokenCountTensor, typename TQueBindType, typename TOffsetBufType>
__aicore__ inline void CopyWeightGroup(GlobalTensor<float> &srcTopkWeight, GlobalTensor<float> &dstTopkWeight,
                                       TTokenCountTensor &srcRankToken, TQueBindType &queue, TOffsetBufType &offsetBuf,
                                       const bool useDirectH1Offsets, const int64_t rows, const int64_t localRows,
                                       const int64_t srcOffset, const int64_t rankId, const int64_t expertStart,
                                       const int64_t expertCount, const int64_t rankNum, const int64_t expertNum,
                                       const int64_t topkBlockElements)
{
    if (rows <= 0) {
        return;
    }
    LocalTensor<int64_t> topkOffsets;
    if (!useDirectH1Offsets) {
        topkOffsets = offsetBuf.template Get<int64_t>();
    }
    LocalTensor<float> topkLocal = queue.template AllocTensor<float>();
    if (rows == localRows) {
        DataCopyExtParams copyParams(1, rows * sizeof(float), 0, 0, 0);
        DataCopyPadExtParams<float> padParams(false, 0, 0, 0);
        DataCopyPad(topkLocal, srcTopkWeight[srcOffset], copyParams, padParams);
    } else {
        int64_t srcRows = 0;
        int64_t localOffset = 0;
        for (int64_t expertOffset = 0; expertOffset < expertCount; ++expertOffset) {
            const int64_t currRows = srcRankToken((rankId * expertNum) + expertStart + expertOffset);
            if (currRows > 0) {
                DataCopyExtParams copyParams(1, currRows * sizeof(float), 0, 0, 0);
                DataCopyPadExtParams<float> padParams(false, 0, 0, 0);
                DataCopyPad(topkLocal[localOffset], srcTopkWeight[srcOffset + srcRows], copyParams, padParams);
            }
            srcRows += currRows;
            localOffset += Ops::Base::CeilAlign(currRows, topkBlockElements);
        }
    }
    queue.EnQue(topkLocal);
    topkLocal = queue.template DeQue<float>();
    int64_t localOffset = 0;
    int64_t expertDstBase = 0;
    if (useDirectH1Offsets) {
        for (int64_t expertBefore = 0; expertBefore < expertStart; ++expertBefore) {
            for (int64_t rankBefore = 0; rankBefore < rankNum; ++rankBefore) {
                expertDstBase += srcRankToken(rankBefore * expertNum + expertBefore);
            }
        }
    }
    for (int64_t expertOffset = 0; expertOffset < expertCount; ++expertOffset) {
        const int64_t currRows = srcRankToken((rankId * expertNum) + expertStart + expertOffset);
        if (currRows > 0) {
            DataCopyExtParams copyParams(1, currRows * sizeof(float), 0, 0, 0);
            int64_t dstOffset = 0;
            if (useDirectH1Offsets) {
                dstOffset = expertDstBase;
                for (int64_t rankBefore = 0; rankBefore < rankId; ++rankBefore) {
                    dstOffset += srcRankToken(rankBefore * expertNum + expertStart + expertOffset);
                }
            } else {
                dstOffset = topkOffsets.GetValue(expertOffset);
            }
            DataCopyPad(dstTopkWeight[dstOffset], topkLocal[localOffset], copyParams);
        }
        localOffset += Ops::Base::CeilAlign(currRows, topkBlockElements);
        if (useDirectH1Offsets) {
            for (int64_t rankBefore = 0; rankBefore < rankNum; ++rankBefore) {
                expertDstBase += srcRankToken(rankBefore * expertNum + expertStart + expertOffset);
            }
        }
    }
    queue.FreeTensor(topkLocal);
}
} // namespace MoeReRoutingTopk

#endif // MOE_RE_ROUTING_TOPK_WEIGHT_KERNEL_COMMON_H
