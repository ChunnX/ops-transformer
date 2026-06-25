/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_expert_token_out_simt.h
 * \brief
 */
#ifndef MOE_V2_EXPERT_TOKEN_OUT_SIMT_H
#define MOE_V2_EXPERT_TOKEN_OUT_SIMT_H

#include "moe_v2_common.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

class MoeV2ExpertTokenOutSimt {
public:
    __aicore__ inline MoeV2ExpertTokenOutSimt(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                GM_ADDR expandedRowIdx, GM_ADDR workspace, const TilingData *tilingData, TPipe *tPipe);
    template <bool CALCHIST = true>
    __aicore__ inline void Process();

private:
    __aicore__ inline void SyncAll();
    __aicore__ inline void InitLocal();
    template <bool CALCHIST>
    __aicore__ inline void ProcessDropPad();

private:
    TPipe *pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expertIdCunsumTBuf_;

    GlobalTensor<int32_t> expandedRowIdxGmGT_;
    __gm__ int32_t *expandedRowIdxGm_;
    __gm__ int32_t *expandedExpertIdxGm_;
    __gm__ int32_t *expertTokensBeforeCapacityGm_;
    __gm__ int32_t *expertTokensCountOrCumsumGm_;
    GlobalTensor<int32_t> expertTokensCountOrCumsumGmOut;
    __gm__ int32_t *expertFirstIndexGm_;
    __gm__ int32_t *expandedExpertIdxIndexGm_;
    __gm__ int32_t *expandedRowIdxIndexGm_;

    LocalTensor<int32_t> lastExpertIdCunsum_;
    __ubuf__ int32_t *lastExpertIdCunsumAddr_;

    const MoeV2GatherOutComputeTilingData *srcToDstTilingData;

    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t expertNum_;
    int64_t expertCapacity_;
    int64_t dropPadMode_ = 0;
    int64_t expertTokensCountOrCumsumFlag_ = 0;
    int64_t expertTokensBeforeCapacityFlag_ = 0;
    int32_t startIndex_;
    int64_t needCoreNum_ = 0;
    int32_t threadNum_;
    bool expertCount_ = false;
    bool expertCumsum_ = false;
};

__aicore__ inline void MoeV2ExpertTokenOutSimt::InitLocal()
{
    // expandedRowIdx initialized to -1, which is used in the src_to_dst_with_capacity step.
    // use this step SyncAll to synchronize every core data
    if (this->dropPadMode_ == 0) {
        return;
    }
    InitOutput(expandedRowIdxGmGT_[this->startIndex_], this->coreRows_, static_cast<int32_t>(-1));
}

__aicore__ inline void MoeV2ExpertTokenOutSimt::SyncAll()
{
    if (this->coreNum_ == 1) {
        return;
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename TilingData>
__aicore__ inline void MoeV2ExpertTokenOutSimt::Init(GM_ADDR expertTokensCountOrCumsum,
                                                     GM_ADDR expertTokensBeforeCapacity, GM_ADDR expandedRowIdx,
                                                     GM_ADDR workspace, const TilingData *tilingData, TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx_ = GetBlockIdx();

    this->coreNum_ = tilingData->coreNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);
    this->expertNum_ = tilingData->expertNum;
    this->expertCapacity_ = tilingData->expertCapacity;
    this->dropPadMode_ = tilingData->dropPadMode;
    this->expertTokensCountOrCumsumFlag_ = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag_ = tilingData->expertTokensBeforeCapacityFlag;
    this->needCoreNum_ = this->srcToDstTilingData->needCoreNum;

    this->startIndex_ = this->blockIdx_ * this->srcToDstTilingData->perCoreRows;

    if (this->blockIdx_ == this->needCoreNum_ - 1) {
        this->coreRows_ = this->srcToDstTilingData->lastCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData->lastCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData->lastCoreLastLoopRows;
    } else {
        this->coreRows_ = this->srcToDstTilingData->perCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData->perCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData->perCoreLastLoopRows;
    }
    this->threadNum_ = this->coreRows_ > THREAD_NUM ? THREAD_NUM : this->coreRows_;

    expandedRowIdxGmGT_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, Align(this->totalLength_, sizeof(int32_t)));
    expandedRowIdxGm_ = (__gm__ int32_t *)expandedRowIdx;

    if (this->dropPadMode_ == DROPLESS_MODE && this->expertTokensCountOrCumsumFlag_ > EXERPT_TOKENS_NONE) {
        if (this->expertTokensCountOrCumsumFlag_ == EXERPT_TOKENS_COUNT) {
            expertTokensBeforeCapacityGm_ = (__gm__ int32_t *)expertTokensCountOrCumsum;
            this->expertCount_ = true;
        } else {
            expertTokensCountOrCumsumGm_ = (__gm__ int32_t *)expertTokensCountOrCumsum;
            expertTokensCountOrCumsumGmOut.SetGlobalBuffer((__gm__ int32_t *)expertTokensCountOrCumsum,
                                                           this->expertNum_);
            this->expertCumsum_ = true;
        }
    }

    if (this->dropPadMode_ == DROP_PAD_MODE && this->expertTokensBeforeCapacityFlag_ == EXERPT_TOKENS_BEFORE_CAPACITY) {
        expertTokensBeforeCapacityGm_ = (__gm__ int32_t *)expertTokensBeforeCapacity;
        this->expertCount_ = true;
    }

    expertFirstIndexGm_ = (__gm__ int32_t *)workspace + Align(this->totalLength_, sizeof(int32_t)) * 2;
    expandedExpertIdxGm_ = (__gm__ int32_t *)workspace;
    expandedExpertIdxIndexGm_ = (__gm__ int32_t *)workspace + Align(this->totalLength_, sizeof(int32_t));
    expandedRowIdxIndexGm_ = (__gm__ int32_t *)workspace + Align(this->totalLength_, sizeof(int32_t)) * 2 +
                             this->expertNum_;
    pipe->InitBuffer(expertIdCunsumTBuf_, BLOCK_BYTES);
    lastExpertIdCunsum_ = expertIdCunsumTBuf_.Get<int32_t>();
    lastExpertIdCunsumAddr_ = (__ubuf__ int32_t *)lastExpertIdCunsum_.GetPhyAddr();
}


__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ExpertFirstIndexComputeSimt(
    int64_t coreRows, int64_t startIndex, int64_t totalLength, int64_t expertNum,
    __gm__ int32_t *expertFirstIndexGm, __gm__ int32_t *expandedExpertIdxGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t curIndex = index + startIndex;
        int32_t curExpertId = expandedExpertIdxGm[curIndex];
        if (curIndex == 0 && curExpertId >= 0 && curExpertId < static_cast<int32_t>(expertNum)) {
            expertFirstIndexGm[curExpertId] = 0;
        }
        if (curIndex == totalLength - 1) {
            continue;
        }
        int64_t nextIndex = curIndex + 1;
        int32_t nextExpertId = expandedExpertIdxGm[nextIndex];
        if (curExpertId != nextExpertId && nextExpertId >= 0 && nextExpertId < static_cast<int32_t>(expertNum)) {
            expertFirstIndexGm[nextExpertId] = nextIndex;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void TokensComputeSimt(
    int64_t coreRows, int64_t startIndex, int64_t totalLength, int64_t expertNum,
    __gm__ int32_t *expertFirstIndexGm, __gm__ int32_t *expandedExpertIdxGm,
    __gm__ int32_t *expertTokensBeforeCapacityGm)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t curIndex = index + startIndex;
        int64_t nextIndex = curIndex + 1;
        int32_t curExpertId = expandedExpertIdxGm[curIndex];
        bool curExpertIdValid = curExpertId >= 0 && curExpertId < static_cast<int32_t>(expertNum);
        if (curIndex == totalLength - 1) { // 最后一个数
            if (curExpertIdValid) {
                expertTokensBeforeCapacityGm[curExpertId] = nextIndex - expertFirstIndexGm[curExpertId];
            }
            continue;
        }
        int32_t nextExpertId = expandedExpertIdxGm[nextIndex];
        if (curExpertId != nextExpertId && curExpertIdValid) {
            expertTokensBeforeCapacityGm[curExpertId] = nextIndex - expertFirstIndexGm[curExpertId];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void CumsumComputeSimt(
    int64_t coreRows, int64_t startIndex, int64_t totalLength, int64_t expertNum,
    __gm__ int32_t *expandedExpertIdxGm, __gm__ int32_t *expertTokensCountOrCumsumGm,
    __ubuf__ int32_t *lastExpertIdCunsumAddr)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t curIndex = startIndex + index;
        int32_t curExpertId = expandedExpertIdxGm[curIndex];
        bool curExpertIdValid = curExpertId >= 0 && curExpertId < static_cast<int32_t>(expertNum);
        if (curIndex == totalLength - 1) { // 最后一个数
            if (threadIdx.y == 0) {
                lastExpertIdCunsumAddr[0] = curExpertIdValid ? curExpertId : static_cast<int32_t>(expertNum);
                lastExpertIdCunsumAddr[1] = curIndex + 1;
            }
            continue;
        }
        int32_t nextExpertId = expandedExpertIdxGm[curIndex + 1];
        if (curExpertId != nextExpertId && curExpertIdValid) {
            int32_t nextExpertIdLimit = nextExpertId < curExpertId ? curExpertId : nextExpertId;
            nextExpertIdLimit = nextExpertIdLimit > static_cast<int32_t>(expertNum) ?
                                static_cast<int32_t>(expertNum) : nextExpertIdLimit;
            int32_t count = nextExpertIdLimit - curExpertId;
            int32_t num = threadIdx.y;
            for (; num < count; num += blockDim.y) {
                expertTokensCountOrCumsumGm[curExpertId + num] = curIndex + 1;
            }
        }
    }
}

template <bool CALCHIST>
__aicore__ inline void MoeV2ExpertTokenOutSimt::ProcessDropPad()
{
    if (this->blockIdx_ != 0) {
        return;
    }

    for (int64_t expertId = 0; expertId < this->expertNum_; expertId++) {
        expertFirstIndexGm_[expertId] = 0;
        if constexpr (CALCHIST) {
            if (this->expertCount_) {
                expertTokensBeforeCapacityGm_[expertId] = 0;
            }
        }
    }

    for (int64_t index = 0; index < this->totalLength_; index++) {
        int32_t expertId = expandedExpertIdxGm_[index];
        if (expertId < 0 || expertId >= static_cast<int32_t>(this->expertNum_) || this->expertCapacity_ <= 0) {
            expandedRowIdxIndexGm_[index] = 0;
            continue;
        }

        int32_t expertTokenCount = expertFirstIndexGm_[expertId];
        int32_t clippedTokenOffset = expertTokenCount < static_cast<int32_t>(this->expertCapacity_) ?
                                     expertTokenCount : static_cast<int32_t>(this->expertCapacity_ - 1);
        int32_t expandedRowIdx = expertId * static_cast<int32_t>(this->expertCapacity_) + clippedTokenOffset;
        expandedRowIdxIndexGm_[index] = expandedRowIdx;

        if (expertTokenCount < static_cast<int32_t>(this->expertCapacity_)) {
            int32_t srcIndex = expandedExpertIdxIndexGm_[index];
            if (srcIndex >= 0 && srcIndex < static_cast<int32_t>(this->totalLength_)) {
                expandedRowIdxGm_[srcIndex] = expandedRowIdx;
            }
        }

        expertTokenCount++;
        expertFirstIndexGm_[expertId] = expertTokenCount;
        if constexpr (CALCHIST) {
            if (this->expertCount_) {
                expertTokensBeforeCapacityGm_[expertId] = expertTokenCount;
            }
        }
    }
}

template <bool CALCHIST>
__aicore__ inline void MoeV2ExpertTokenOutSimt::Process()
{
    if (this->blockIdx_ < this->needCoreNum_) {
        InitLocal();
    }

    this->SyncAll();

    if (this->dropPadMode_ == DROP_PAD_MODE) {
        ProcessDropPad<CALCHIST>();
        this->SyncAll();
        return;
    }

    if (this->blockIdx_ < this->needCoreNum_) {
        asc_vf_call<ExpertFirstIndexComputeSimt>(dim3{static_cast<uint32_t>(this->threadNum_), 1, 1},
                                                   this->coreRows_, this->startIndex_, this->totalLength_,
                                                   this->expertNum_, expertFirstIndexGm_, expandedExpertIdxGm_);
    }

    this->SyncAll();

    if constexpr (!CALCHIST) {
        return;
    }

    if (this->expertCount_ && this->blockIdx_ < this->needCoreNum_) {
        asc_vf_call<TokensComputeSimt>(dim3{static_cast<uint32_t>(this->threadNum_), 1, 1}, this->coreRows_,
                                         this->startIndex_, this->totalLength_, this->expertNum_, expertFirstIndexGm_,
                                         expandedExpertIdxGm_, expertTokensBeforeCapacityGm_);
    }

    if (this->expertCumsum_ && this->blockIdx_ < this->needCoreNum_) {
        int32_t threadDimY =
            THREAD_NUM / this->threadNum_ > MAX_THREAD_DIM_Y_NUM ? MAX_THREAD_DIM_Y_NUM : THREAD_NUM / this->threadNum_;
        asc_vf_call<CumsumComputeSimt>(
            dim3{static_cast<uint32_t>(this->threadNum_), static_cast<uint32_t>(threadDimY), 1}, this->coreRows_,
            this->startIndex_, this->totalLength_, this->expertNum_, expandedExpertIdxGm_, expertTokensCountOrCumsumGm_,
            lastExpertIdCunsumAddr_);
    }

    if (this->expertCumsum_ && this->blockIdx_ == this->needCoreNum_ - 1) {
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        int32_t expertId = lastExpertIdCunsumAddr_[0];
        int32_t cunsumNum = lastExpertIdCunsumAddr_[1];
        int32_t count = expertNum_ - expertId;
        if (expertId >= 0 && expertId < expertNum_ && count > 0) {
            InitOutput(expertTokensCountOrCumsumGmOut[expertId], static_cast<uint32_t>(count), cunsumNum);
        }
    }
}
} // namespace MoeInitRoutingV2
#endif // MOE_V2_EXPERT_TOKEN_OUT_SIMT_H
