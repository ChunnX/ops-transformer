/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_row_idx_gather.h
 * \brief
 */
#ifndef MOE_V3_ROW_IDX_GATHER_H
#define MOE_V3_ROW_IDX_GATHER_H

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class RowIdxGather {
public:
    __aicore__ inline RowIdxGather(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR workspace, const MoeInitRoutingV3TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t loop, int64_t elements);
    __aicore__ inline void CopyOut(int64_t loop, int64_t elements);

private:
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> sortedExpertIndicesGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> assistGm;

    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIndicesInQueue_;
    TQue<QuePosition::VECOUT, 1> copyOutQueue_;

    const MoeV3ExpertTokensCountTilingData *expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t actualExpertNum_ = 0;

    int64_t loops_ = 0;
    int64_t perLoopElements_ = 0;
    int64_t lastLoopElements_ = 0;

    int64_t perCoreElements;
};

__aicore__ inline void RowIdxGather::Init(GM_ADDR expandedRowIdx, GM_ADDR workspace,
                                          const MoeInitRoutingV3TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    actualExpertNum_ = tilingData->actualExpertNum;

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, actualExpertNum_);
    if (blockIdx_ < needCoreNum_ - 1) {
        curCoreElements_ = perCoreElements_;
    } else if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
    }

    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                            Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        actualExpertNum_);
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreElements = Ceil(expertTotalCount_, needCoreNum_);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreElements);
    int64_t lastCoreElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreElements;
    int64_t perCoreLoops = Ceil(perCoreElements, expertTokensCountTilingData_->perCorePerLoopElements);
    int64_t perCorePerLoopElements = Ceil(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    int64_t lastCoreLoops = Ceil(lastCoreElements, expertTokensCountTilingData_->perCorePerLoopElements);
    int64_t lastCorePerLoopElements = Ceil(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopELements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    loops_ = perCoreLoops;
    if (blockIdx_ == needCoreNum_ - 1) {
        loops_ = lastCoreLoops;
        perLoopElements_ = lastCorePerLoopElements;
        lastLoopElements_ = lastCoreLastLoopELements;
    } else {
        loops_ = perCoreLoops;
        perLoopElements_ = perCorePerLoopElements;
        lastLoopElements_ = perCoreLastLoopElements;
    }

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, actualExpertNum_);
    sortedExpertIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                               Align(tilingData->n * tilingData->k, sizeof(int32_t)) +
                                               blockIdx_ * perCoreElements,
                                           actualExpertNum_);

    pipe_->InitBuffer(sortedExpertIndicesInQueue_, 1, AlignBytes(perLoopElements_, sizeof(int32_t)));
    pipe_->InitBuffer(copyOutQueue_, 1, AlignBytes(1, sizeof(int32_t)));
}

__aicore__ inline void RowIdxGather::Process()
{
    if (blockIdx_ < needCoreNum_) {
        for (int64_t loop = 0; loop < loops_; loop++) {
            int64_t elements = perLoopElements_;
            if (loop == loops_ - 1) {
                elements = lastLoopElements_;
            }
            CopyIn(loop, elements);
            CopyOut(loop, elements);
        }
    }
}

__aicore__ inline void RowIdxGather::CopyIn(int64_t loop, int64_t elements)
{
    LocalTensor<int32_t> sortedExpertIndicesInLocal = sortedExpertIndicesInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(elements * sizeof(int32_t)), 0, 0,
                                     0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(sortedExpertIndicesInLocal, sortedExpertIndicesGm_[loop * perLoopElements_], dataCopyParams,
                dataCopyPadParams);
    sortedExpertIndicesInQueue_.EnQue(sortedExpertIndicesInLocal);
}

__aicore__ inline void RowIdxGather::CopyOut(int64_t loop, int64_t elements)
{
    LocalTensor<int32_t> sortedExpertIndicesInLocal = sortedExpertIndicesInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> outLocal = copyOutQueue_.AllocTensor<int32_t>();
    int64_t valueBase = blockIdx_ * perCoreElements + loop * perLoopElements_;
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t i = 0; i < elements; i++) {
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        int64_t outIndices = sortedExpertIndicesInLocal.GetValue(i);
        int64_t value = valueBase + i;
        outLocal.SetValue(0, value);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[outIndices], outLocal, copyParams);
    }
    sortedExpertIndicesInQueue_.FreeTensor(sortedExpertIndicesInLocal);
    copyOutQueue_.FreeTensor(outLocal);
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_ROW_IDX_GATHER_H