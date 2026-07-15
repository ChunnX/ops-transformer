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
 * \file moe_scatter_out_batch_row.h
 * \brief Scatter consecutive output rows directly from the sorted dst-to-src index.
 */
#ifndef MOE_SCATTER_OUT_BATCH_ROW_H
#define MOE_SCATTER_OUT_BATCH_ROW_H

#include "moe_common.h"
#include "kernel_operator.h"

constexpr int64_t BUFFER_NUM_BATCH = 2;

namespace MoeTokenPermute {
using namespace AscendC;

template <typename T>
class MoeScatterOutBatchRow {
public:
    __aicore__ inline MoeScatterOutBatchRow(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR expandedX, GM_ADDR workspace,
                                const MoeTokenPermuteTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInDstToSrcIndices(int64_t progress, int64_t loopRows);
    __aicore__ inline void ScatterOutOneBlock(int64_t outPosStartIdx, int64_t outPosRowsNum, int64_t indexOffset,
                                              LocalTensor<int32_t> &dstToSrcLocal);
    __aicore__ inline void ScatterOutCompute(int64_t progress, int64_t loopRows);

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM_BATCH> dstToSrcIndexQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM_BATCH> outputRowsQueue;

    GlobalTensor<T> inputXGm;
    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> dstToSrcIndexGm;

    const PermuteGatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t totalLength;
    int64_t activateRows;
    int64_t ubRowsPerLoop;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
};

template <typename T>
__aicore__ inline void MoeScatterOutBatchRow<T>::CopyInDstToSrcIndices(int64_t progress, int64_t loopRows)
{
    LocalTensor<int32_t> indicesLocal = dstToSrcIndexQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(loopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, dstToSrcIndexGm[progress * this->perLoopRows], dataCopyParams, dataCopyPadParams);
    dstToSrcIndexQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeScatterOutBatchRow<T>::ScatterOutOneBlock(int64_t outPosStartIdx, int64_t outPosRowsNum,
                                                                    int64_t indexOffset,
                                                                    LocalTensor<int32_t> &dstToSrcLocal)
{
    LocalTensor<T> outRowsLocal = outputRowsQueue.AllocTensor<T>();
    int64_t validRows = Min(outPosRowsNum, this->activateRows - outPosStartIdx);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};
    for (int64_t outRow = 0; outRow < validRows; outRow++) {
        int32_t expandedTokenIdx = dstToSrcLocal.GetValue(indexOffset + outRow);
        if (expandedTokenIdx < 0 || expandedTokenIdx >= this->totalLength) {
            continue;
        }
        int64_t srcRow = expandedTokenIdx / this->k;
        if (srcRow < 0 || srcRow >= this->n) {
            continue;
        }
        DataCopyPad(outRowsLocal[outRow * this->cols], inputXGm[srcRow * this->cols], copyInParams, copyInPadParams);
    }

    SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
    if (validRows > 0) {
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(validRows), static_cast<uint32_t>(this->cols * sizeof(T)),
                                        0, 0, 0};
        DataCopyPad(expandedXGm[outPosStartIdx * this->cols], outRowsLocal, copyOutParams);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    outputRowsQueue.FreeTensor(outRowsLocal);
}

template <typename T>
__aicore__ inline void MoeScatterOutBatchRow<T>::ScatterOutCompute(int64_t progress, int64_t loopRows)
{
    LocalTensor<int32_t> dstToSrcLocal = dstToSrcIndexQueue.DeQue<int32_t>();
    int64_t outPosStartIdx = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    if (outPosStartIdx >= this->activateRows) {
        dstToSrcIndexQueue.FreeTensor(dstToSrcLocal);
        return;
    }

    int64_t ubLoops = Ceil(loopRows, this->ubRowsPerLoop);
    for (int64_t ubLoop = 0; ubLoop < ubLoops; ubLoop++) {
        int64_t currentRows = Min(this->ubRowsPerLoop, loopRows - ubLoop * this->ubRowsPerLoop);
        ScatterOutOneBlock(outPosStartIdx + ubLoop * this->ubRowsPerLoop, currentRows, ubLoop * this->ubRowsPerLoop,
                           dstToSrcLocal);
    }
    dstToSrcIndexQueue.FreeTensor(dstToSrcLocal);
}

template <typename T>
__aicore__ inline void MoeScatterOutBatchRow<T>::Init(GM_ADDR inputX, GM_ADDR expandedX, GM_ADDR workspace,
                                                      const MoeTokenPermuteTilingData *tilingData, TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx = GetBlockIdx();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);
    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->topK;
    this->totalLength = tilingData->n * tilingData->topK;

    if (this->blockIdx == this->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
        this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->lastCoreLoops;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->perCoreLoops;
    }
    this->ubRowsPerLoop = this->gatherOutTilingData->perLoopCols;

    inputXGm.SetGlobalBuffer((__gm__ T *)inputX, this->n * this->cols);
    expandedXGm.SetGlobalBuffer((__gm__ T *)expandedX, this->activateRows * this->cols);
    dstToSrcIndexGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) +
                                        this->blockIdx * this->gatherOutTilingData->perCoreRows,
                                    Align(this->coreRows, sizeof(int32_t)));

    pipe->InitBuffer(outputRowsQueue, BUFFER_NUM_BATCH, AlignBytes(this->ubRowsPerLoop * this->cols, sizeof(T)));
    pipe->InitBuffer(dstToSrcIndexQueue, BUFFER_NUM_BATCH, AlignBytes(this->perLoopRows, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeScatterOutBatchRow<T>::Process()
{
    if (this->blockIdx >= this->needCoreNum || this->rowLoops == 0) {
        return;
    }

    CopyInDstToSrcIndices(0, this->perLoopRows);
    for (int64_t loop = 0; loop < this->rowLoops; loop++) {
        int64_t currentRows = (loop == this->rowLoops - 1) ? this->lastLoopRows : this->perLoopRows;
        ScatterOutCompute(loop, currentRows);
        if (loop < this->rowLoops - 1) {
            int64_t nextRows = (loop + 1 == this->rowLoops - 1) ? this->lastLoopRows : this->perLoopRows;
            CopyInDstToSrcIndices(loop + 1, nextRows);
        }
    }
}

} // namespace MoeTokenPermute
#endif // MOE_SCATTER_OUT_BATCH_ROW_H
