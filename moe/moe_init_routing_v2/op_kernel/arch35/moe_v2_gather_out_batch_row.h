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
 * \file moe_v2_gather_out_batch_row.h
 * \brief 按行批量Gather输出模板，用于token数多、hiddenSize小的场景优化
 *        按UB能放的完整token行数进行批量搬运，减少搬入次数，提升性能
 */
#ifndef MOE_V2_GATHER_OUT_BATCH_ROW_H
#define MOE_V2_GATHER_OUT_BATCH_ROW_H

#include "moe_v2_common.h"
#include "kernel_operator.h"

constexpr int64_t BUFFER_NUM_GATHER = 2;

namespace MoeInitRoutingV2 {
using namespace AscendC;

template <typename T>
class MoeV2GatherOutBatchRow {
public:
    __aicore__ inline MoeV2GatherOutBatchRow(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR workspace,
                                const MoeInitRoutingV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t progress);
    __aicore__ inline void CopyInZeroIndices(int64_t progress);
    __aicore__ inline void CopyInSrcRows(int64_t srcBlockStart, int64_t srcBlockRows);
    __aicore__ inline void GatherOut(int64_t progress);
    __aicore__ inline void CopyOutZero(int64_t progress);

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM_GATHER> inputRowsQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM_GATHER> expandedRowIdxQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM_GATHER> expandedRowIdxIndexQueue;

    GlobalTensor<T> inputXGm;
    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int32_t> expandedRowIdxIndexGm;

    const MoeV2GatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t n;
    int64_t k;
    int64_t expertNum;
    int64_t totalLength;
    int64_t activateRows;
    int64_t dropPadMode;

    int64_t ubRowsPerLoop;

    int64_t currentUbRows;
    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
};

template <typename T>
__aicore__ inline void MoeV2GatherOutBatchRow<T>::CopyInIndices(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm[progress * this->perLoopRows], dataCopyParams, dataCopyPadParams);
    expandedRowIdxQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutBatchRow<T>::CopyInZeroIndices(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((this->currentLoopRows + 1) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxIndexGm[progress * this->perLoopRows], dataCopyParams, dataCopyPadParams);
    expandedRowIdxIndexQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutBatchRow<T>::CopyInSrcRows(int64_t srcBlockStart, int64_t srcBlockRows)
{
    LocalTensor<T> srcRowsLocal = inputRowsQueue.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{static_cast<uint32_t>(srcBlockRows), static_cast<uint32_t>(this->cols * sizeof(T)),
                                     0, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(srcRowsLocal, inputXGm[srcBlockStart * this->cols], dataCopyParams, dataCopyPadParams);
    inputRowsQueue.EnQue<T>(srcRowsLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutBatchRow<T>::GatherOut(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxQueue.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t srcRowStart = initialRow / this->k;
    int64_t srcRowEnd = (initialRow + this->currentLoopRows - 1) / this->k;
    int64_t totalSrcRows = srcRowEnd - srcRowStart + 1;
    int64_t ubRowsLoops = Ceil(totalSrcRows, this->ubRowsPerLoop);

    for (int64_t ubBlockIdx = 0; ubBlockIdx < ubRowsLoops; ubBlockIdx++) {
        this->currentUbRows = this->ubRowsPerLoop;
        int64_t remainingSrcRows = totalSrcRows - ubBlockIdx * this->ubRowsPerLoop;
        if (remainingSrcRows < this->ubRowsPerLoop) {
            this->currentUbRows = remainingSrcRows;
        }

        int64_t srcBlockStart = srcRowStart + ubBlockIdx * this->ubRowsPerLoop;
        int64_t srcBlockEnd = srcBlockStart + this->currentUbRows - 1;

        CopyInSrcRows(srcBlockStart, this->currentUbRows);
        LocalTensor<T> srcRowsLocal = inputRowsQueue.DeQue<T>();
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

        DataCopyExtParams intriParams{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};

        for (int64_t curLoopRow = 0; curLoopRow < this->currentLoopRows; curLoopRow++) {
            int64_t globalTokenIdx = initialRow + curLoopRow;
            int64_t srcRowIdx = globalTokenIdx / this->k;

            if (srcRowIdx < srcBlockStart || srcRowIdx > srcBlockEnd) {
                continue;
            }

            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
                continue;
            }

            int64_t ubRowOffset = (srcRowIdx - srcBlockStart) * this->cols;
            int64_t outOffset = outIndex * this->cols;
            DataCopyPad(expandedXGm[outOffset], srcRowsLocal[ubRowOffset], intriParams);
        }

        inputRowsQueue.FreeTensor(srcRowsLocal);
    }

    expandedRowIdxQueue.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutBatchRow<T>::CopyOutZero(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexQueue.DeQue<int32_t>();
    LocalTensor<T> zeroTensor = inputRowsQueue.AllocTensor<T>();

    if (this->blockIdx == 0 && progress == 0) {
        int32_t curIndex = 0;
        int32_t nextIndex = indicesLocal.GetValue(0);
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitGmValue(expandedXGm[curIndex * cols], zeroTensor, count * cols, static_cast<T>(0));
        }
    }

    for (int i = 0; i < currentLoopRows; i++) {
        int32_t curIndex = indicesLocal.GetValue(i) + 1;
        if (activateRows <= curIndex) {
            break;
        }
        int32_t nextIndex;
        if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1 && progress == this->rowLoops - 1 &&
            i == currentLoopRows - 1) {
            nextIndex = activateRows;
        } else {
            nextIndex = indicesLocal.GetValue(i + 1);
        }
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitGmValue(expandedXGm[curIndex * cols], zeroTensor, count * cols, static_cast<T>(0));
        }
    }

    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    inputRowsQueue.FreeTensor(zeroTensor);
    expandedRowIdxIndexQueue.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherOutBatchRow<T>::Init(GM_ADDR inputX, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                       GM_ADDR workspace, const MoeInitRoutingV2TilingData *tilingData,
                                                       TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx = GetBlockIdx();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->expertNum = tilingData->expertNum;
    this->dropPadMode = tilingData->dropPadMode;
    this->totalLength = tilingData->n * tilingData->k;

    if (this->dropPadMode == DROP_PAD_MODE) {
        this->activateRows = tilingData->expertCapacity * tilingData->expertNum;
    }

    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
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
    expandedXGm.SetGlobalBuffer((__gm__ T *)expandedX, this->totalLength * this->cols);
    expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx +
                                         this->blockIdx * this->gatherOutTilingData->perCoreRows,
                                     Align(this->coreRows, sizeof(int32_t)));

    expandedRowIdxIndexGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2 +
                                              this->expertNum + this->blockIdx * this->gatherOutTilingData->perCoreRows,
                                          0);

    pipe->InitBuffer(inputRowsQueue, BUFFER_NUM_GATHER, AlignBytes(this->ubRowsPerLoop * this->cols, sizeof(T)));
    pipe->InitBuffer(expandedRowIdxQueue, BUFFER_NUM_GATHER, AlignBytes(this->perLoopRows, sizeof(int32_t)));
    pipe->InitBuffer(expandedRowIdxIndexQueue, BUFFER_NUM_GATHER, AlignBytes(this->perLoopRows, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV2GatherOutBatchRow<T>::Process()
{
    if (this->blockIdx < this->needCoreNum) {
        currentLoopRows = perLoopRows;
        for (int64_t loop = 0; loop < this->rowLoops; loop++) {
            if (loop == this->rowLoops - 1) {
                currentLoopRows = lastLoopRows;
            }

            if (this->dropPadMode == DROP_PAD_MODE) {
                CopyInZeroIndices(loop);
                CopyOutZero(loop);
            }

            CopyInIndices(loop);
            GatherOut(loop);
        }
    }
}

} // namespace MoeInitRoutingV2
#endif // MOE_V2_GATHER_OUT_BATCH_ROW_H