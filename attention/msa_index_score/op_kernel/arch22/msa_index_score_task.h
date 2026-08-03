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
 * \file msa_index_score_task.h
 * \brief varlen + 因果裁剪的任务枚举器。
 *
 * AIC 与 AIV 各自持有一个实例并走完全相同的任务序列，是 cross-core flag
 * set/wait 计数配对的前提，因此本文件中的推导必须保持为纯函数、两侧共用。
 */

#ifndef MSA_INDEX_SCORE_TASK_H
#define MSA_INDEX_SCORE_TASK_H

#include "kernel_operator.h"
#include "../msa_index_score_common.h"

namespace MsaIndexScoreNs {

__aicore__ inline uint32_t MsaCeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t MsaMinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

/// 一个任务 = 一个请求内的一个 M-tile。
/// M 维 = 请求内 (token, head) 扁平行索引，head 为低位。
struct MsaTask {
    uint32_t batchIdx;
    uint32_t cuQStart;      // 该请求首个 token 的全局扁平下标
    uint32_t mStart;        // tile 在请求内的起始行
    uint32_t mActual;       // tile 实际行数
    uint32_t globalRowBase; // cuQStart * numQHeads + mStart
    int32_t startLoc;
    int32_t kvLen;
    uint32_t fullEndBlk;    // [0, fullEndBlk) 全可见，无需 mask
    uint32_t visibleEndBlk; // [fullEndBlk, visibleEndBlk) 需 mask；>= visibleEndBlk 全不可见
    uint32_t numSTiles;     // 本任务需要产出的 S-tile 数
};

class MsaTaskScheduler {
public:
    __aicore__ inline MsaTaskScheduler() {}

    __aicore__ inline void Init(uint32_t batch, uint32_t numQHeads, uint32_t maxBlocksPerBatch,
                                uint32_t scoreBlockStride, bool fillInvalidTail,
                                const AscendC::GlobalTensor<int32_t> &cuSeqQ,
                                const AscendC::GlobalTensor<int32_t> &kvLen,
                                const AscendC::GlobalTensor<int32_t> &startLoc)
    {
        batch_ = batch;
        numQHeads_ = numQHeads;
        maxBlocksPerBatch_ = maxBlocksPerBatch;
        scoreBlockStride_ = scoreBlockStride;
        fillInvalidTail_ = fillInvalidTail;
        gCuSeqQ_ = cuSeqQ;
        gKvLen_ = kvLen;
        gStartLoc_ = startLoc;

        totalTasks_ = 0;
        for (uint32_t b = 0; b < batch_; ++b) {
            const uint32_t qLen = static_cast<uint32_t>(gCuSeqQ_.GetValue(b + 1) - gCuSeqQ_.GetValue(b));
            totalTasks_ += MsaCeilDiv(qLen * numQHeads_, MSA_ROW_TILE_M);
        }
        Reset();
    }

    __aicore__ inline uint32_t TotalTasks() const
    {
        return totalTasks_;
    }

    __aicore__ inline void Reset()
    {
        curBatch_ = 0;
        taskBase_ = 0;
        LoadBatch(0);
    }

    /// taskIdx 必须单调不减（核内 taskIdx += coreNum 的遍历顺序天然满足）。
    __aicore__ inline void Decode(uint32_t taskIdx, MsaTask &task)
    {
        while (taskIdx >= taskBase_ + curTasks_) {
            taskBase_ += curTasks_;
            ++curBatch_;
            LoadBatch(curBatch_);
        }

        task.batchIdx = curBatch_;
        task.cuQStart = cuQStart_;
        task.startLoc = startLoc_;
        task.kvLen = kvLen_;
        task.mStart = (taskIdx - taskBase_) * MSA_ROW_TILE_M;
        task.mActual = MsaMinU32(MSA_ROW_TILE_M, curRows_ - task.mStart);
        task.globalRowBase = cuQStart_ * numQHeads_ + task.mStart;

        // 因果 / 长度边界：与 Triton _prefill_index_score_kernel L217-L237 等价。
        const uint32_t tLo = task.mStart / numQHeads_;
        const uint32_t tHi = (task.mStart + task.mActual - 1) / numQHeads_;
        const int32_t qPosLo = startLoc_ + static_cast<int32_t>(tLo);
        const int32_t qPosHi = startLoc_ + static_cast<int32_t>(tHi);

        int32_t visibleKeyEnd = (qPosHi + 1) < kvLen_ ? (qPosHi + 1) : kvLen_;
        if (visibleKeyEnd < 0) {
            visibleKeyEnd = 0;
        }
        uint32_t visibleEndBlk = MsaCeilDiv(static_cast<uint32_t>(visibleKeyEnd), MSA_BLOCK_SIZE);
        visibleEndBlk = MsaMinU32(visibleEndBlk, maxBlocksPerBatch_);

        const int32_t causalFullSigned = (qPosLo + 1) < 0 ? 0 : (qPosLo + 1);
        const uint32_t causalFull = static_cast<uint32_t>(causalFullSigned) / MSA_BLOCK_SIZE;
        const uint32_t seqFull = (kvLen_ < 0 ? 0U : static_cast<uint32_t>(kvLen_)) / MSA_BLOCK_SIZE;
        uint32_t fullEndBlk = MsaMinU32(causalFull, seqFull);

        task.visibleEndBlk = visibleEndBlk;
        task.fullEndBlk = MsaMinU32(fullEndBlk, visibleEndBlk);

        const uint32_t outEndBlk = fillInvalidTail_ ?
            scoreBlockStride_ :
            MsaCeilDiv(visibleEndBlk, MSA_BLOCKS_PER_STILE) * MSA_BLOCKS_PER_STILE;
        task.numSTiles = MsaCeilDiv(outEndBlk, MSA_BLOCKS_PER_STILE);
    }

private:
    __aicore__ inline void LoadBatch(uint32_t b)
    {
        if (b >= batch_) {
            curTasks_ = 0;
            curRows_ = 0;
            cuQStart_ = 0;
            kvLen_ = 0;
            startLoc_ = 0;
            return;
        }
        cuQStart_ = static_cast<uint32_t>(gCuSeqQ_.GetValue(b));
        const uint32_t qLen = static_cast<uint32_t>(gCuSeqQ_.GetValue(b + 1)) - cuQStart_;
        curRows_ = qLen * numQHeads_;
        curTasks_ = MsaCeilDiv(curRows_, MSA_ROW_TILE_M);
        kvLen_ = gKvLen_.GetValue(b);
        startLoc_ = gStartLoc_.GetValue(b);
    }

    AscendC::GlobalTensor<int32_t> gCuSeqQ_;
    AscendC::GlobalTensor<int32_t> gKvLen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;

    uint32_t batch_ = 0;
    uint32_t numQHeads_ = 1;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t scoreBlockStride_ = 0;
    bool fillInvalidTail_ = true;

    uint32_t totalTasks_ = 0;
    uint32_t curBatch_ = 0;
    uint32_t taskBase_ = 0;
    uint32_t curTasks_ = 0;
    uint32_t curRows_ = 0;
    uint32_t cuQStart_ = 0;
    int32_t kvLen_ = 0;
    int32_t startLoc_ = 0;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_TASK_H
