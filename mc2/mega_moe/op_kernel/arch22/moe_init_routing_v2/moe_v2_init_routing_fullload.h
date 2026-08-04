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
 * \file moe_v2_init_routing_fullload.h
 * \brief
 */

#ifndef MC2_MOE_V2_INIT_ROUTING_FULLLOAD_H
#define MC2_MOE_V2_INIT_ROUTING_FULLLOAD_H

#include "moe_v2_mrgsort.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;
using namespace Mc2Tiling;


template <typename T>
class MoeV2FullLoad : public MoeV2SortBase {
public:
    __aicore__ inline MoeV2FullLoad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                const MoeInitRoutingV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void SortCompute();
    __aicore__ inline void CopyOutIdx();
    __aicore__ inline void CopyOutEmpty();
    __aicore__ inline void CopyOutX();
    __aicore__ inline void ComputeExpertTokenCountOrCumsum();

private:
    int64_t sortNum_;
    const MoeV2GatherOutComputeTilingData *gatherOutTilingData_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t sortNeedCoreNum_;
    int64_t coreRows_;
    int64_t perCoreRows_;
    int64_t k_;
    int64_t n_;
    int64_t cols_;
    int64_t activateRows_;
    int64_t expertNum;
    int64_t expertCapacity;

    TQue<QuePosition::VECIN, 1> xCopyInQueue_;
    TQue<QuePosition::VECOUT, 1> expandedRowIdxCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> expandedExpertIdxCopyOutQueue_;
    TQue<QuePosition::VECOUT, 1> expandDstToSrcRowQueue_;
    TQue<QuePosition::VECOUT, 1> expertTokensCopyOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> expertIdxGm_;

    GlobalTensor<T> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
    GlobalTensor<int32_t> expertTokensBeforeCapacityGm;

    int64_t expertTokensCountOrCumsumFlag = 0;
    int64_t expertTokensBeforeCapacityFlag = 0;
    int64_t dropPadMode = 0;
};

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyIn()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(this->totalLength * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    DataCopyPadCustom(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
#else
    DataCopyPad(inLocal[0], expertIdxGm_, dataCopyParams, dataCopyPadParams);
#endif
    ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->totalLength);
    sortDataCopyInQueue.EnQue(inLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::SortCompute()
{
    LocalTensor<int32_t> inLocal = sortDataCopyInQueue.DeQue<int32_t>();
    LocalTensor<int32_t> expertIdxLocal = inLocal[0];
    LocalTensor<float> expertIdxLocalFp32 = expertIdxLocal.ReinterpretCast<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    Cast(expertIdxLocalFp32, expertIdxLocal, RoundMode::CAST_NONE, this->totalLength);
#else
    Cast(expertIdxLocalFp32, expertIdxLocal, RoundMode::CAST_ROUND, this->totalLength);
#endif
    PipeBarrier<PIPE_V>();
    Muls(expertIdxLocalFp32, expertIdxLocalFp32, (float)-1, this->totalLength);
    PipeBarrier<PIPE_V>();
    int64_t duplicateNum = this->totalLength % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expertIdxLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    LocalTensor<float> concatLocal;
#else
    LocalTensor<float> concatLocal = expertIdxLocalFp32;
#endif
    LocalTensor<float> tempTensor = tempBuffer.Get<float>(GetSortLen<float>(this->sortNum_));
    Concat(concatLocal, expertIdxLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> rowIdxLocal = inLocal[this->sortNum_].template ReinterpretCast<uint32_t>();
    LocalTensor<float> sortedLocal = sortedBuffer.Get<float>(GetSortLen<float>(this->sortNum_));
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> expandedExpertIdxLocal = expandedExpertIdxCopyOutQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> expandDstToSrcRowLocal = expandDstToSrcRowQueue_.AllocTensor<uint32_t>();
    LocalTensor<float> expandDstToSrcRowLocalFp32 = expandDstToSrcRowLocal.ReinterpretCast<float>();
    Extract(expandedExpertIdxLocal, expandDstToSrcRowLocal, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    Cast(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
         this->totalLength);
#else
    Cast(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND,
         this->totalLength);
#endif
    PipeBarrier<PIPE_V>();
    Muls(expandedExpertIdxLocal, expandedExpertIdxLocal, (float)-1, this->totalLength);
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> expandedExpertIdxLocalInt32;
    expandedExpertIdxLocalInt32 = expandedExpertIdxLocal.ReinterpretCast<int32_t>();
    Cast(expandedExpertIdxLocalInt32, expandedExpertIdxLocal, RoundMode::CAST_ROUND, this->totalLength);
    PipeBarrier<PIPE_V>();
    expandedExpertIdxCopyOutQueue_.EnQue<int32_t>(expandedExpertIdxLocalInt32);

    LocalTensor<uint32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.AllocTensor<uint32_t>();
    LocalTensor<uint32_t> expandedRowIdxU32 = expandedRowIdx.ReinterpretCast<uint32_t>();
    Muls(expandDstToSrcRowLocalFp32, expandDstToSrcRowLocalFp32, (float)-1, this->totalLength);
    PipeBarrier<PIPE_V>();
    ArithProgression<int32_t>(inLocal[this->sortNum_], 0, 1, this->totalLength);
    PipeBarrier<PIPE_V>();
    if (duplicateNum > 0) {
        int duplicateIndex = this->totalLength - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> (FP32_ONE_REPEAT_NUM - ONE_REPEAT_SORT_NUM));
        uint64_t mask[2] = {mask0, 0};
        Duplicate(expandDstToSrcRowLocalFp32[duplicateIndex], MIN_FP32, mask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
        PipeBarrier<PIPE_V>();
    }
    Concat(concatLocal, expandDstToSrcRowLocalFp32, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedLocal, concatLocal, rowIdxLocal, tempTensor, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    Extract(tempTensor, expandedRowIdxU32, sortedLocal, this->sortNum_ / ONE_REPEAT_SORT_NUM);
    PipeBarrier<PIPE_V>();
    expandedRowIdxCopyOutQueue_.EnQue<uint32_t>(expandedRowIdx);
    sortDataCopyInQueue.FreeTensor(inLocal);

    expandDstToSrcRowQueue_.FreeTensor(expandDstToSrcRowLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutIdx()
{
    LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    DataCopyExtParams intriParams;
#else
    DataCopyParams intriParams;
#endif
    intriParams.blockCount = 1;
    intriParams.blockLen = this->totalLength * sizeof(int32_t);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    DataCopyCustom(expandedRowIdxGm_, expandedRowIdx, intriParams.blockCount, intriParams.blockLen);
#else
    DataCopyPad(expandedRowIdxGm_, expandedRowIdx, intriParams);
#endif
    expandedRowIdxCopyOutQueue_.EnQue(expandedRowIdx);
}

// 统计每个专家的 token 数（EXERPT_TOKENS_COUNT）或其前缀和（EXERPT_TOKENS_CUMSUM）。
//
// 【原实现的问题】原本这段由 `sortNeedCoreNum_ - 1` 号核**一个核**串行完成：先
// Duplicate 清零 expertNum 个 int32，再对全部 totalLength 个已排序专家 id 逐元素
// 做标量 GetValue，中间穿插按专家推进的标量 SetValue（CUMSUM 模式尾部还要再按
// expertNum 规模标量填充一遍）。其余核全部在调用侧的 SyncAll 上干等。
// 实测（A3 ws16 decode，BF16）：dec16 4.00us / dec32 6.54us / dec64 11.34us，
// 约 22ns/元素——瓶颈不是算法复杂度，而是 O(totalLength + expertNum) 次标量
// GetValue/SetValue 的 UB 往返。且该段位于 AIC 头部空闲窗口内（此时 HBM 空闲、
// MTE2 为 0% 忙），省下的每一微秒都是净收益，不会被带宽吸收。
//
// 【本实现】把工作按**专家区间**切给所有 sort 核并行做。两个关键前提：
//  1) fullload 形态下，每个 `blockIdx_ < sortNeedCoreNum_` 的核都在 SortCompute()
//     里冗余地排出了一份**完整**的 expandedExpertIdx，数据本来就在各核自己的 UB
//     里——分核统计不需要任何额外的数据搬运。
//  2) 切分的是**输出域**（专家），各核写出的 GM 切片互不重叠，天然无需跨核归约，
//     因此**不新增任何同步点**。调用侧（mega_moe_kernel_a3.hpp / a2_bf16.hpp）在
//     moe_init_routing_v2 返回后本就有一次 `SyncAll<true>`，足以保证后续
//     ALLGATHER 读到全部核的写出。
// 于是每核只需：① 二分定位本区间的第一个 token（O(log2 totalLength) 次标量读）；
// ② 线性扫过落在本区间内的 token；③ 写出本区间的专家计数。
// 单核 O(totalLength + expertNum) 降为每核约 O(totalLength/N + expertNum/N + log2 totalLength)。
//
// 输出逐位不变：每核对区间内每个专家都显式写值（故不再需要 Duplicate 清零），
// 各核区间的并集恰为 [0, expertNum)，写出字节数与原实现完全一致。
template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::ComputeExpertTokenCountOrCumsum()
{
    LocalTensor<int32_t> expandedExpertIdx = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();

    // 每核负责的专家个数按 8（= 32B / sizeof(int32_t)）向上对齐，
    // 保证各核写出的 GM 切片起始地址 32B 对齐且互不重叠。
    int64_t perCoreExperts = Align(Ceil(this->expertNum, this->sortNeedCoreNum_), sizeof(int32_t));
    int64_t expertLo = this->blockIdx_ * perCoreExperts;
    if (expertLo >= this->expertNum) {
        // 本核没有分到专家（expertNum 小于核数时会出现），不写出任何数据。
        expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdx);
        return;
    }
    int64_t expertHi = Min(expertLo + perCoreExperts, this->expertNum);

    LocalTensor<int32_t> expertTokensCount = expertTokensCopyOutQueue_.AllocTensor<int32_t>();

    // 二分找到第一个满足 expandedExpertIdx[i] >= expertLo 的位置。
    // expandedExpertIdx 是 Sort2 的输出，沿 token 维已按专家 id 升序排列，
    // 因此 lower_bound 成立；tokenIdx 同时就是 #{i : idx[i] < expertLo}，
    // 即「本区间之前所有专家的 token 总数」，正是 CUMSUM 模式所需的起始累加值。
    int64_t left = 0;
    int64_t right = this->totalLength;
    while (left < right) {
        int64_t mid = (left + right) >> 1;
        if (static_cast<int64_t>(expandedExpertIdx.GetValue(mid)) < expertLo) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    int64_t tokenIdx = left;
    int64_t acc = tokenIdx;

    for (int64_t e = expertLo; e < expertHi; e++) {
        int64_t tokenCount = 0;
        // 已排序，故等于 e 的 token 必然连续；扫到第一个 > e 的元素即停。
        while (tokenIdx < this->totalLength &&
               static_cast<int64_t>(expandedExpertIdx.GetValue(tokenIdx)) <= e) {
            tokenIdx++;
            tokenCount++;
        }
        acc += tokenCount;
        // COUNT：本专家独立 token 数；CUMSUM：截至本专家（含）的 token 总数。
        // 专家无 token 时 tokenCount 为 0、acc 保持不变，与原实现（Duplicate 清零后
        // 不写 / 尾部用总数填充）逐位一致。
        expertTokensCount.SetValue(e - expertLo, static_cast<int32_t>(
            this->expertTokensCountOrCumsumFlag == EXERPT_TOKENS_CUMSUM ? acc : tokenCount));
    }

    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>((expertHi - expertLo) * sizeof(int32_t)), 0, 0, 0};
    if (this->expertTokensCountOrCumsumFlag > 0) {
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        DataCopyCustom(expertTokensCountOrCumsumGm[expertLo], expertTokensCount, copyParams.blockCount,
                       copyParams.blockLen);
#else
        DataCopyPad(expertTokensCountOrCumsumGm[expertLo], expertTokensCount, copyParams);
#endif
    }
    expertTokensCopyOutQueue_.FreeTensor(expertTokensCount);
    expandedExpertIdxCopyOutQueue_.FreeTensor(expandedExpertIdx);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutX()
{
    LocalTensor<T> xLocal = xCopyInQueue_.AllocTensor<T>();
    LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = this->cols_ * sizeof(T);
    int64_t inFactor = Align(this->cols_, sizeof(T));
    int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
    int64_t startXRow = curRowsStart / this->k_;
    int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;

    DataCopyExtParams dataXCopyParams{static_cast<uint16_t>(endXRow - startXRow + 1),
                                      static_cast<uint32_t>(this->cols_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataXCopyPadParams{false, 0, 0, 0};
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    DataCopyPadCustom(xLocal, xGm_[startXRow * this->cols_], dataXCopyParams, dataXCopyPadParams);
#else
    DataCopyPad(xLocal, xGm_[startXRow * this->cols_], dataXCopyParams, dataXCopyPadParams);
#endif
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t k = 0;
    for (int64_t i = startXRow; i <= endXRow; i++) {
        for (; k < this->perCoreRows_ && curRowsStart / this->k_ == i; curRowsStart++, k++) {
            int32_t outIndex = expandedRowIdx.GetValue(curRowsStart);
            if (outIndex < this->activateRows_) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
                DataCopyCustom(expandedXGm_[outIndex * this->cols_], xLocal[(i - startXRow) * inFactor],
                               intriParams.blockCount, intriParams.blockLen);
#else
                DataCopyPad(expandedXGm_[outIndex * this->cols_], xLocal[(i - startXRow) * inFactor], intriParams);
#endif
            }
        }
    }
    expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
    xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::CopyOutEmpty()
{
    LocalTensor<int32_t> outLocal = expandedExpertIdxCopyOutQueue_.DeQue<int32_t>();
    expandedExpertIdxCopyOutQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::Init(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                              GM_ADDR expertTokensCountOrCumsum, GM_ADDR workspace,
                                              const MoeInitRoutingV2TilingData *tilingData, TPipe *tPipe)
{
    this->gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    // this->blockIdx_ = GetBlockIdx();
    this->blockIdx_ = get_block_idx() + get_subblockid() * get_block_num();
    this->n_ = tilingData->n;
    this->k_ = tilingData->k;
    this->cols_ = tilingData->cols;
    this->sortNeedCoreNum_ = this->gatherOutTilingData_->needCoreNum;
    this->needCoreNum_ = this->gatherOutTilingData_->needCoreNum;
    if (needCoreNum_ == 0) {
        this->sortNeedCoreNum_ = tilingData->vbsComputeParamsOp.needCoreNum;
    }
    this->perCoreRows_ = this->gatherOutTilingData_->perCoreRows;
    this->activateRows_ = this->gatherOutTilingData_->activateRows;
    if (this->blockIdx_ == this->gatherOutTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->gatherOutTilingData_->lastCoreRows;
    } else {
        this->coreRows_ = this->gatherOutTilingData_->perCoreRows;
    }
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
    this->dropPadMode = tilingData->dropPadMode;
    this->expertNum = tilingData->expertNum;

    this->tileLength = Align(tilingData->vbsComputeParamsOp.lastCorePerLoopElements, sizeof(int32_t));
    this->sortNum_ = Ceil(this->tileLength, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    this->totalLength = tilingData->n * tilingData->k;
    this->pipe = tPipe;

    xGm_.SetGlobalBuffer((__gm__ T *)x);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx, this->tileLength);

    expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, this->tileLength);
    if (this->expertTokensCountOrCumsumFlag > 0) {
        // dropless
        expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t *)expertTokensCountOrCumsum,
                                                    Align(this->expertNum, sizeof(int32_t)));
    }

    int64_t kvFactor = 2;
    int64_t buffSize = this->sortNum_ * sizeof(int32_t);

    int64_t curRowsStart = this->blockIdx_ * this->perCoreRows_;
    int64_t startXRow = curRowsStart / this->k_;
    int64_t endXRow = (curRowsStart + this->coreRows_ - 1) / this->k_;
    pipe->InitBuffer(xCopyInQueue_, bufferNum, AlignBytes(this->cols_, sizeof(T)) * (endXRow - startXRow + 1));
    pipe->InitBuffer(expandedRowIdxCopyOutQueue_, bufferNum, buffSize);
    pipe->InitBuffer(expandedExpertIdxCopyOutQueue_, bufferNum, buffSize);
    pipe->InitBuffer(expertTokensCopyOutQueue_, bufferNum, AlignBytes(this->expertNum, sizeof(int32_t)));
    pipe->InitBuffer(expandDstToSrcRowQueue_, bufferNum, buffSize);
    pipe->InitBuffer(sortDataCopyInQueue, bufferNum, buffSize * kvFactor);

// sort310p
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    pipe->InitBuffer(tempBuffer, buffSize * REGIONP_ROPOSAL_KV_RATIO * kvFactor);
    pipe->InitBuffer(sortedBuffer, buffSize * REGIONP_ROPOSAL_KV_RATIO * kvFactor);
#else
    pipe->InitBuffer(tempBuffer, buffSize * kvFactor);
    pipe->InitBuffer(sortedBuffer, buffSize * kvFactor);
#endif
}

template <typename T>
__aicore__ inline void MoeV2FullLoad<T>::Process()
{
    if (this->blockIdx_ < this->sortNeedCoreNum_) {
        CopyIn();
        SortCompute();
        if (this->blockIdx_ == 0) {
            CopyOutIdx();
        }
        // ComputeExpertTokenCountOrCumsum 现在按专家区间分摊到**所有** sort 核并行执行
        // （原本只有最后一个核做，其余核在调用侧的 SyncAll 上干等）。各核写出的 GM
        // 切片互不重叠，不需要新增同步点。注意仍要保证每个走到这里的核都恰好
        // DeQue 一次 expandedExpertIdxCopyOutQueue_，否则队列状态会失配。
        if (this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
            ComputeExpertTokenCountOrCumsum();
        } else {
            CopyOutEmpty();
        }
    }
    if (this->needCoreNum_ != 0) {
        if (this->blockIdx_ < this->needCoreNum_) {
            CopyOutX();
        }

    } else {
        if (this->blockIdx_ < this->sortNeedCoreNum_) {
            LocalTensor<int32_t> expandedRowIdx = expandedRowIdxCopyOutQueue_.DeQue<int32_t>();
            expandedRowIdxCopyOutQueue_.FreeTensor(expandedRowIdx);
        }
    }
}
} // namespace MoeInitRoutingV2
#endif // MC2_MOE_V2_INIT_ROUTING_FULLLOAD_H
