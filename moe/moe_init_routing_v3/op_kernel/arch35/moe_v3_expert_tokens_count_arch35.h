/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_expert_tokens_count_arch35.h
 * \brief
 */
#ifndef MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE_ARCH35
#define MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE_ARCH35

#include "moe_v3_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

constexpr int64_t CUMSUM_MODE = 0LL;
constexpr int64_t COUNT_MODE = 1LL;
constexpr int64_t KEY_VALUE_MODE = 2LL;
constexpr int64_t KEY_VALUE_MODE_DIM_NUM = 2LL;

class ExpertTokensCount {
public:
    __aicore__ inline ExpertTokensCount(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut();

    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();
    __aicore__ inline void InitBasicParams(const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe);

private:
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expertIdxValueGm_;
    TPipe* pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

    const MoeV3Arch35ExpertTokensCountTilingData* expertTokensCountTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t expertTokensNumType_ = 0;
    int64_t expertCountElements_ = 0;
    int64_t coreNum_ = 0;
    int64_t dropPadMode_ = 0;
    int64_t needExpertTotalCount_ = 0;
    int64_t useGatherCopy_ = 0;
    int32_t finalExpertId_ = -1;
    int32_t expertTokenValue_ = 0;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t* sortedExpertIdGmAddr,
    __ubuf__ int32_t* expertFirstIndexLocalAddr)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elementNum; i += static_cast<int32_t>(blockDim.x)) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId < expertStart) {
            continue;
        }
        if (currExpertId >= expertEnd) {
            break;
        }
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertCountOutSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd, __gm__ int32_t* sortedExpertIdGmAddr,
    __ubuf__ int32_t* expertFirstIndexLocalAddr, __ubuf__ int32_t* expertCountOutLocalAddr)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elementNum; i += static_cast<int32_t>(blockDim.x)) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId < expertStart) {
            continue;
        }
        if (currExpertId >= expertEnd) {
            break;
        }
        if (i == elementNum - 1 || currExpertId != sortedExpertIdGmAddr[i + 1]) {
            expertCountOutLocalAddr[currExpertId - expertStart] =
                i + 1 - expertFirstIndexLocalAddr[currExpertId - expertStart];
        }
    }
}

// One warp scans contiguous expert ranges. All lanes participate in the
// shuffles, including lanes whose range is empty, so partial warps need no
// special case. Keep prefix sums in int64 to preserve the output semantics.
constexpr int32_t EXPERT_SCAN_THREADS = 32;

template <bool keyValue>
__simt_vf__ __aicore__ LAUNCH_BOUND(EXPERT_SCAN_THREADS) inline void ComputeExpertPrefixSimt(
    int32_t expertCount, int64_t expertStart, int64_t outputElements, __ubuf__ int32_t* counts,
    __ubuf__ int64_t* output, __ubuf__ int64_t* summary)
{
    int32_t lane = static_cast<int32_t>(threadIdx.x);
    int32_t chunk = (expertCount + EXPERT_SCAN_THREADS - 1) / EXPERT_SCAN_THREADS;
    int32_t begin = lane * chunk;
    int32_t end = begin + chunk < expertCount ? begin + chunk : expertCount;
    int64_t localSum = 0;
    int32_t localNonzero = 0;
    for (int32_t i = begin; i < end; ++i) {
        int32_t count = counts[i];
        localSum += static_cast<int64_t>(count);
        if constexpr (keyValue) {
            localNonzero += count != 0;
        }
    }
    int64_t inclusiveSum = localSum;
    int32_t inclusiveNonzero = localNonzero;
    for (uint32_t offset = 1; offset < EXPERT_SCAN_THREADS; offset <<= 1) {
        int64_t previousSum = asc_shfl_up(inclusiveSum, offset);
        if (lane >= offset) {
            inclusiveSum += previousSum;
        }
        if constexpr (keyValue) {
            int32_t previousNonzero = asc_shfl_up(inclusiveNonzero, offset);
            if (lane >= offset) {
                inclusiveNonzero += previousNonzero;
            }
        }
    }
    int64_t prefix = inclusiveSum - localSum;
    int32_t position = inclusiveNonzero - localNonzero;
    for (int32_t i = begin; i < end; ++i) {
        int64_t count = static_cast<int64_t>(counts[i]);
        if constexpr (keyValue) {
            if (count != 0) {
                output[position * KEY_VALUE_MODE_DIM_NUM] = expertStart + i;
                output[position * KEY_VALUE_MODE_DIM_NUM + 1] = count;
                ++position;
            }
        } else {
            prefix += count;
            output[i] = prefix;
        }
    }
    if (lane == EXPERT_SCAN_THREADS - 1) {
        summary[0] = inclusiveSum;
        if constexpr (keyValue) {
            summary[1] = inclusiveNonzero;
            // Emit the existing zero terminator only when output capacity
            // allows it; a fully populated expert table has no spare pair.
            int64_t terminator = static_cast<int64_t>(inclusiveNonzero) * KEY_VALUE_MODE_DIM_NUM;
            if (terminator < outputElements) {
                output[terminator] = 0;
                output[terminator + 1] = 0;
            }
        }
    }
}

__aicore__ inline void ExpertTokensCount::InitBasicParams(const MoeInitRoutingV3Arch35TilingData* tilingData,
                                                          TPipe* tPipe)
{
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;
    expertNum_ = tilingData->expertNum;
    expertTokensNumType_ = tilingData->expertTokensNumType;
    coreNum_ = tilingData->coreNum;
    dropPadMode_ = tilingData->dropPadMode;
    useGatherCopy_ = tilingData->useGatherCopy;
    needExpertTotalCount_ = static_cast<int64_t>(dropPadMode_ != DROP_PAD_MODE);

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
    } else {
        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
    }
    if (expertTokensNumType_ == KEY_VALUE_MODE) {
        expertCountElements_ = ((actualExpertNum_ + 1) < expertNum_) ? (actualExpertNum_ + 1) * KEY_VALUE_MODE_DIM_NUM :
                                                                       expertNum_ * KEY_VALUE_MODE_DIM_NUM;
    } else {
        expertCountElements_ = actualExpertNum_;
    }
}

__aicore__ inline void ExpertTokensCount::Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                               const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe)
{
    InitBasicParams(tilingData, tPipe);

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + blockIdx_ * perCoreElements_, curCoreElements_);
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int64_t*)expertTokensCount, expertCountElements_);
    expertCountTempGm_.SetGlobalBuffer(
        (__gm__ int32_t*)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2, actualExpertNum_);
    if (needExpertTotalCount_) {
        expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                                Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                                Align(actualExpertNum_, sizeof(int32_t)),
                                            1);
    }
    if (dropPadMode_ == DROP_PAD_MODE) {
        expertIdxValueGm_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                              Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                              Align(actualExpertNum_, sizeof(int32_t)),
                                          coreNum_ * EXPERT_ID_VALUE_NUM);
    }

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx + blockIdx_ * perCoreElements_);
    if ((tilingData->rowIdxType == GATHER) && (blockIdx_ < needCoreNum_)) {
        InitGlobalMemory(expandedRowIdxGm_, curCoreElements_, -1);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
    if (useGatherCopy_ && tilingData->rowIdxType && (blockIdx_ < needCoreNum_)) {
        int64_t totalLength = Align(tilingData->n * tilingData->k, sizeof(int32_t));
        GlobalTensor<int32_t> gatherRowIdxInitGm;
        gatherRowIdxInitGm.SetGlobalBuffer((__gm__ int32_t*)workspace + totalLength + blockIdx_ * perCoreElements_,
                                           curCoreElements_);
        InitGlobalMemory(gatherRowIdxInitGm, curCoreElements_, static_cast<int32_t>(-1));
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
    pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(expertCountElements_, sizeof(int64_t)));
    if (needExpertTotalCount_) {
        pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
    }
}

__aicore__ inline void ExpertTokensCount::Process()
{
    if (blockIdx_ < needCoreNum_) {
        LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
        Duplicate(expertCountOutLocal, 0, actualExpertNum_);

        __gm__ int32_t* sortedExpertIdxGmAddr = (__gm__ int32_t*)sortedExpertIdxGm_.GetPhyAddr();
        __ubuf__ int32_t* expertCountOutLocalAddr = (__ubuf__ int32_t*)expertCountOutLocal.GetPhyAddr();

        asc_vf_call<ComputeExpertFirstIndexSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_,
                                                 expertEnd_, sortedExpertIdxGmAddr, expertCountOutLocalAddr);
        asc_vf_call<ComputeExpertCountOutSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, curCoreElements_, expertStart_, expertEnd_,
                                               sortedExpertIdxGmAddr, expertCountOutLocalAddr, expertCountOutLocalAddr);

        expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);

        if (dropPadMode_ == DROP_PAD_MODE && curCoreElements_ > 0) {
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            finalExpertId_ = sortedExpertIdxGm_.GetValue(curCoreElements_ - 1);
            if (finalExpertId_ >= expertStart_ && finalExpertId_ < expertEnd_) {
                expertTokenValue_ = expertCountOutLocal.GetValue(finalExpertId_ - expertStart_);
            }
        }

        CopyOut();
    }

    SyncAll();

    if (blockIdx_ == 0) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
    if (needExpertTotalCount_) {
        SyncAll();
    }
}

__aicore__ inline void ExpertTokensCount::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                 0, 0, 0};
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
    SetAtomicNone();
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

    if (dropPadMode_ == DROP_PAD_MODE) {
        expertCountOutLocal.SetValue(0, finalExpertId_);
        expertCountOutLocal.SetValue(1, expertTokenValue_);
        DataCopyExtParams expertIdxCopyParams{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>(EXPERT_ID_VALUE_NUM * sizeof(int32_t)), 0, 0, 0};
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expertIdxValueGm_[blockIdx_ * EXPERT_ID_VALUE_NUM], expertCountOutLocal, expertIdxCopyParams);
    }
    expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyIn()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
    expertCountTempInQueue_.EnQue(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCompute()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
    if (expertTokensNumType_ == COUNT_MODE) {
        Cast(expertCountOutLocal, expertCountTempInLocal, RoundMode::CAST_NONE, actualExpertNum_);
        PipeBarrier<PIPE_V>();
    } else {
        // The histogram queue is free after CopyOut. Its aligned allocation
        // provides at least 32 bytes, enough for the two int64 summary values.
        LocalTensor<int64_t> summary = expertCountOutToTempQueue_.AllocTensor<int64_t>();
        auto* counts = (__ubuf__ int32_t*)expertCountTempInLocal.GetPhyAddr();
        auto* output = (__ubuf__ int64_t*)expertCountOutLocal.GetPhyAddr();
        auto* summaryAddr = (__ubuf__ int64_t*)summary.GetPhyAddr();
        if (expertTokensNumType_ == KEY_VALUE_MODE) {
            asc_vf_call<ComputeExpertPrefixSimt<true>>(dim3{EXPERT_SCAN_THREADS, 1, 1},
                                                       static_cast<int32_t>(actualExpertNum_), expertStart_,
                                                       expertCountElements_, counts, output, summaryAddr);
        } else {
            asc_vf_call<ComputeExpertPrefixSimt<false>>(dim3{EXPERT_SCAN_THREADS, 1, 1},
                                                        static_cast<int32_t>(actualExpertNum_), expertStart_,
                                                        expertCountElements_, counts, output, summaryAddr);
        }
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        actualExpertTotalNum_ = summary.GetValue(0);
        if (expertTokensNumType_ == KEY_VALUE_MODE) {
            int64_t expertOffset = summary.GetValue(1);
            expertCountElements_ = Min(expertCountElements_, (expertOffset + 1) * KEY_VALUE_MODE_DIM_NUM);
        }
        expertCountOutToTempQueue_.FreeTensor(summary);
    }
    event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
    expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
    if (needExpertTotalCount_) {
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
        if (expertTokensNumType_ == COUNT_MODE) {
            // Reuse the released local histogram as reduction scratch. The
            // workspace total is int32, just like each histogram entry; keep
            // the reduction on the vector pipe instead of reading every
            // expert count back through the scalar pipe.
            LocalTensor<int32_t> reduceTmp = expertCountOutToTempQueue_.AllocTensor<int32_t>();
            ReduceSum(expertTotalCountLocal, expertCountTempInLocal, reduceTmp, actualExpertNum_);
            expertCountOutToTempQueue_.FreeTensor(reduceTmp);
        } else {
            expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
        }
        expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
    }
    expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyOut()
{
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    if (needExpertTotalCount_) {
        LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
        copyParams.blockLen = sizeof(int32_t);
        DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
        expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
    }
    expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_EXPERT_TOKENS_COUNT_H_REGBASE_ARCH35
