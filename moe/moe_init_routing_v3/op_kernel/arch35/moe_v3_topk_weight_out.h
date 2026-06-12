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
 * \file moe_v3_topk_weight_out.h
 * \brief TopkWeight rearrangement for non-FullLoad path (arch35/regbase)
 */
#ifndef MOE_V3_TOPK_WEIGHT_OUT_H
#define MOE_V3_TOPK_WEIGHT_OUT_H

#include "moe_v3_common.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class MoeV3TopkWeightOut {
public:
    __aicore__ inline MoeV3TopkWeightOut(){};
    __aicore__ inline void Init(GM_ADDR topkWeight, GM_ADDR expandedRowIdx, GM_ADDR expandedTopkWeight,
                                GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> topkWeightCopyInQueue_;

    GlobalTensor<float> topkWeightGm_;
    GlobalTensor<float> expandedTopkWeightGm_;
    GlobalTensor<int32_t> expandDstToSrcRowGm_;
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;

    int64_t blockIdx_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t activeNum_;
    int64_t expertTotalCount_;
    int64_t rowIdxType_;
    int64_t dropPadMode_;
    int64_t expertNum_;
    int64_t expertCapacity_;
    int64_t expertStart_;
    int64_t expertEnd_;
    int64_t outputRows_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t coreElements_;
    int64_t coreNum_;
};

__aicore__ inline void MoeV3TopkWeightOut::Init(GM_ADDR topkWeight, GM_ADDR expandedRowIdx,
                                                  GM_ADDR expandedTopkWeight, GM_ADDR workspace,
                                                  const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    coreNum_ = tilingData->coreNum;
    n_ = tilingData->n;
    k_ = tilingData->k;
    totalLength_ = n_ * k_;
    activeNum_ = tilingData->activeNum;
    rowIdxType_ = tilingData->rowIdxType;
    dropPadMode_ = tilingData->dropPadMode;
    expertNum_ = tilingData->expertNum;
    expertCapacity_ = tilingData->expertCapacity;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;

    outputRows_ = (dropPadMode_ == DROP_PAD_MODE) ? expertNum_ * expertCapacity_ : activeNum_;

    GlobalTensor<int32_t> expertTotalCountGm;
    expertTotalCountGm.SetGlobalBuffer((__gm__ int32_t *)workspace +
        Align(totalLength_, sizeof(int32_t)) * 2 + Align(tilingData->actualExpertNum, sizeof(int32_t)), 1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(expertTotalCountGm);
    expertTotalCount_ = expertTotalCountGm.GetValue(0);

    topkWeightGm_.SetGlobalBuffer((__gm__ float *)topkWeight, totalLength_);
    expandedTopkWeightGm_.SetGlobalBuffer((__gm__ float *)expandedTopkWeight, outputRows_);

    if (dropPadMode_ != DROP_PAD_MODE) {
        expandDstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
            Align(totalLength_, sizeof(int32_t)), Align(totalLength_, sizeof(int32_t)));
    }

    if (rowIdxType_ == SCATTER && dropPadMode_ != DROP_PAD_MODE) {
        sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace, Align(totalLength_, sizeof(int32_t)));
    }

    if (dropPadMode_ == DROP_PAD_MODE) {
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, totalLength_);
    }

    int64_t splitBase;
    if (dropPadMode_ == DROP_PAD_MODE) {
        splitBase = totalLength_;
    } else if (rowIdxType_ == SCATTER) {
        splitBase = activeNum_;
    } else {
        splitBase = expertTotalCount_;
    }
    perCoreElements_ = Ceil(splitBase, coreNum_);
    needCoreNum_ = Ceil(splitBase, perCoreElements_);
    if (blockIdx_ == needCoreNum_ - 1) {
        coreElements_ = splitBase - (needCoreNum_ - 1) * perCoreElements_;
    } else {
        coreElements_ = perCoreElements_;
    }

    pipe_->InitBuffer(topkWeightCopyInQueue_, 1, AlignBytes(1, sizeof(float)));
}

__aicore__ inline void MoeV3TopkWeightOut::Process()
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    int64_t startIdx = blockIdx_ * perCoreElements_;

    if (dropPadMode_ == DROP_PAD_MODE) {
        int64_t perCoreZeroRows = Ceil(outputRows_, coreNum_);
        int64_t startRow = blockIdx_ * perCoreZeroRows;
        int64_t endRow = Min(startRow + perCoreZeroRows, outputRows_);
        if (startRow < endRow) {
            LocalTensor<float> zeroLocal = topkWeightCopyInQueue_.AllocTensor<float>();
            Duplicate<float>(zeroLocal, 0.0f, static_cast<int32_t>(1));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            for (int64_t row = startRow; row < endRow; row++) {
                DataCopyPad(expandedTopkWeightGm_[row], zeroLocal, copyParams);
            }
            topkWeightCopyInQueue_.FreeTensor(zeroLocal);
        }
#ifndef __CCE_KT_TEST__
        AscendC::SyncAll();
#endif

        if (blockIdx_ < needCoreNum_) {
            int64_t endIdx = startIdx + coreElements_;
            for (int64_t globalSortIdx = startIdx; globalSortIdx < endIdx; globalSortIdx++) {
                int64_t dstIdx = expandedRowIdxGm_.GetValue(globalSortIdx);
                if (dstIdx != -1) {
                    LocalTensor<float> topkWeightLocal = topkWeightCopyInQueue_.AllocTensor<float>();
                    DataCopyPad(topkWeightLocal, topkWeightGm_[globalSortIdx], copyParams, padParams);
                    topkWeightCopyInQueue_.EnQue(topkWeightLocal);
                    topkWeightLocal = topkWeightCopyInQueue_.DeQue<float>();
                    DataCopyPad(expandedTopkWeightGm_[dstIdx], topkWeightLocal, copyParams);
                    topkWeightCopyInQueue_.FreeTensor(topkWeightLocal);
                }
            }
        }
        return;
    }

    if (blockIdx_ >= needCoreNum_) return;

    if (rowIdxType_ == SCATTER) {
        int64_t endIdx = Min(startIdx + coreElements_, activeNum_);
        for (int64_t i = startIdx; i < endIdx; i++) {
            int32_t curExpertId = sortedExpertIdxGm_.GetValue(i);
            if (curExpertId < expertStart_ || curExpertId >= expertEnd_) {
                break;
            }
            int64_t originalPos = expandDstToSrcRowGm_.GetValue(i);
            LocalTensor<float> topkWeightLocal = topkWeightCopyInQueue_.AllocTensor<float>();
            DataCopyPad(topkWeightLocal, topkWeightGm_[originalPos], copyParams, padParams);
            topkWeightCopyInQueue_.EnQue(topkWeightLocal);
            topkWeightLocal = topkWeightCopyInQueue_.DeQue<float>();
            DataCopyPad(expandedTopkWeightGm_[i], topkWeightLocal, copyParams);
            topkWeightCopyInQueue_.FreeTensor(topkWeightLocal);
        }
    } else {
        int64_t endIdx = startIdx + coreElements_;
        for (int64_t curIndex = startIdx; curIndex < endIdx; curIndex++) {
            int64_t originalPos = expandDstToSrcRowGm_.GetValue(curIndex);
            LocalTensor<float> topkWeightLocal = topkWeightCopyInQueue_.AllocTensor<float>();
            DataCopyPad(topkWeightLocal, topkWeightGm_[originalPos], copyParams, padParams);
            topkWeightCopyInQueue_.EnQue(topkWeightLocal);
            topkWeightLocal = topkWeightCopyInQueue_.DeQue<float>();
            DataCopyPad(expandedTopkWeightGm_[curIndex], topkWeightLocal, copyParams);
            topkWeightCopyInQueue_.FreeTensor(topkWeightLocal);
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_TOPK_WEIGHT_OUT_H