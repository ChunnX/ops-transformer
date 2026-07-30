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
 * \file moe_finalize_routing_v2_unpermute_fast.h
 * \brief BF16 K=1 fast path for the pure row-gather form of MoeFinalizeRoutingV2.
 */
#ifndef MOE_FINALIZE_ROUTING_V2_UNPERMUTE_FAST
#define MOE_FINALIZE_ROUTING_V2_UNPERMUTE_FAST

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRoutingV2 {

using namespace AscendC;

template <typename T>
class MoeFinalizeRoutingV2UnpermuteFast {
public:
    __aicore__ inline MoeFinalizeRoutingV2UnpermuteFast(const MoeFinalizeRoutingV2TilingData &tilingData, TPipe &pipe)
        : tilingData_(tilingData), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR expandedRows, GM_ADDR rowIndices, GM_ADDR out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessRows(int64_t outputOffset, int64_t rowCount);
    __aicore__ inline void ProcessOneRow(int64_t localIndex, int64_t outputRow);
    __aicore__ inline void ProcessRowSlice(int32_t inputRow, int64_t outputRow, int64_t sliceIndex,
                                           int64_t sliceLength);
    __aicore__ inline void CopyRowIn(int32_t inputRow, int64_t sliceIndex, int64_t sliceLength);
    __aicore__ inline void CastRow(int64_t sliceLength);
    __aicore__ inline void CopyRowOut(int64_t outputRow, int64_t sliceIndex, int64_t sliceLength);

private:
    static constexpr int64_t BLOCK_SIZE = 32;
    static constexpr int64_t ALIGN_512 = 512;

    const MoeFinalizeRoutingV2TilingData &tilingData_;
    TPipe &pipe_;
    TQue<QuePosition::VECIN, 1> rowsQueue_;
    TQue<QuePosition::VECIN, 1> indicesQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> castBuffer0_;
    TBuf<TPosition::VECCALC> castBuffer1_;

    GlobalTensor<T> rowsGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<T> outGm_;
    LocalTensor<int32_t> indicesLocal_;
    LocalTensor<float> rowFloat_;

    DataCopyPadExtParams<T> rowPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> indexPadParams_{false, 0, 0, 0};
    DataCopyExtParams copyParams_{1, 0, 0, 0, 0};

    int64_t hiddenSize_{0};
    int64_t hiddenSliceLength_{0};
    int64_t hiddenSliceCount_{0};
    int64_t hiddenSliceRemain_{0};
    int64_t rowsPerCore_{0};
    int64_t remainderCoreCount_{0};
    int64_t rowsPerTile_{0};
    int64_t tileCount_{0};
    int64_t tileRemain_{0};
};

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::Init(GM_ADDR expandedRows, GM_ADDR rowIndices, GM_ADDR out)
{
    hiddenSize_ = tilingData_.H;
    hiddenSliceLength_ = tilingData_.normalH;
    hiddenSliceCount_ = tilingData_.hSliceNum;
    hiddenSliceRemain_ = tilingData_.unnormalH;
    rowsPerCore_ = tilingData_.normalCoreHandleNum;
    remainderCoreCount_ = tilingData_.tailCoreHandleNum;
    rowsPerTile_ = tilingData_.normalCoreHandleNumPerLoop;
    tileCount_ = tilingData_.normalCoreLoopNum;
    tileRemain_ = tilingData_.normalCoreHandleNumTailLoop;

    if (remainderCoreCount_ > 0 && GetBlockIdx() < remainderCoreCount_) {
        ++rowsPerCore_;
        ++tileRemain_;
    }

    int64_t coreRowOffset;
    if (remainderCoreCount_ > 0) {
        if (GetBlockIdx() < remainderCoreCount_) {
            coreRowOffset = rowsPerCore_ * GetBlockIdx();
        } else {
            coreRowOffset =
                (rowsPerCore_ + 1) * remainderCoreCount_ + rowsPerCore_ * (GetBlockIdx() - remainderCoreCount_);
        }
    } else {
        coreRowOffset = rowsPerCore_ * GetBlockIdx();
    }

    rowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(expandedRows), tilingData_.totalRowNum * hiddenSize_);
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rowIndices) + coreRowOffset, rowsPerCore_);
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out) + coreRowOffset * hiddenSize_, rowsPerCore_ * hiddenSize_);

    int64_t alignedHiddenSlice = (hiddenSliceLength_ + ALIGN_512 - 1) & ~(ALIGN_512 - 1);
    pipe_.InitBuffer(rowsQueue_, tilingData_.normalK, alignedHiddenSlice * sizeof(T));
    pipe_.InitBuffer(indicesQueue_, 1, rowsPerTile_ * sizeof(int32_t));
    pipe_.InitBuffer(outQueue_, 1, alignedHiddenSlice * sizeof(T));
    pipe_.InitBuffer(castBuffer0_, alignedHiddenSlice * sizeof(float) + 256);
    pipe_.InitBuffer(castBuffer1_, alignedHiddenSlice * sizeof(float));
    rowFloat_ = castBuffer0_.Get<float>();
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::Process()
{
    for (int64_t tile = 0; tile < tileCount_; ++tile) {
        ProcessRows(tile * rowsPerTile_, rowsPerTile_);
    }
    if (tileRemain_ > 0) {
        ProcessRows(tileCount_ * rowsPerTile_, tileRemain_);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::ProcessRows(int64_t outputOffset, int64_t rowCount)
{
    indicesLocal_ = indicesQueue_.AllocTensor<int32_t>();
    copyParams_.blockLen = rowCount * sizeof(int32_t);
    DataCopyPad(indicesLocal_, indicesGm_[outputOffset], copyParams_, indexPadParams_);
    indicesQueue_.EnQue(indicesLocal_);
    indicesLocal_ = indicesQueue_.DeQue<int32_t>();

#pragma bisheng auto_sync parallel
    for (int64_t row = 0; row < rowCount; ++row) {
        ProcessOneRow(row, outputOffset + row);
    }

    indicesQueue_.FreeTensor(indicesLocal_);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::ProcessOneRow(int64_t localIndex, int64_t outputRow)
{
    int32_t inputRow = indicesLocal_.GetValue(localIndex);
    for (int64_t slice = 0; slice < hiddenSliceCount_; ++slice) {
        ProcessRowSlice(inputRow, outputRow, slice, hiddenSliceLength_);
    }
    if (hiddenSliceRemain_ > 0) {
        ProcessRowSlice(inputRow, outputRow, hiddenSliceCount_, hiddenSliceRemain_);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::ProcessRowSlice(int32_t inputRow, int64_t outputRow,
                                                                             int64_t sliceIndex, int64_t sliceLength)
{
    CopyRowIn(inputRow, sliceIndex, sliceLength);
    PipeBarrier<PIPE_V>();
    CastRow(sliceLength);
    CopyRowOut(outputRow, sliceIndex, sliceLength);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::CopyRowIn(int32_t inputRow, int64_t sliceIndex,
                                                                       int64_t sliceLength)
{
    LocalTensor<T> rowLocal = rowsQueue_.AllocTensor<T>();
    int64_t inputOffset = inputRow * hiddenSize_ + sliceIndex * hiddenSliceLength_;
    if (likely((sliceLength * sizeof(T)) % BLOCK_SIZE == 0)) {
        DataCopy(rowLocal, rowsGm_[inputOffset], sliceLength);
    } else {
        copyParams_.blockLen = sliceLength * sizeof(T);
        DataCopyPad(rowLocal, rowsGm_[inputOffset], copyParams_, rowPadParams_);
    }
    rowsQueue_.EnQue(rowLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::CastRow(int64_t sliceLength)
{
    LocalTensor<T> rowLocal = rowsQueue_.DeQue<T>();
    Cast(rowFloat_, rowLocal, RoundMode::CAST_NONE, sliceLength);
    rowsQueue_.FreeTensor(rowLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2UnpermuteFast<T>::CopyRowOut(int64_t outputRow, int64_t sliceIndex,
                                                                        int64_t sliceLength)
{
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    PipeBarrier<PIPE_V>();
    Cast(outLocal, rowFloat_, RoundMode::CAST_RINT, sliceLength);
    outQueue_.EnQue(outLocal);
    outLocal = outQueue_.DeQue<T>();

    int64_t outputOffset = outputRow * hiddenSize_ + sliceIndex * hiddenSliceLength_;
    if (likely((sliceLength * sizeof(T)) % BLOCK_SIZE == 0)) {
        DataCopy(outGm_[outputOffset], outLocal, sliceLength);
    } else {
        copyParams_.blockLen = sliceLength * sizeof(T);
        DataCopyPad(outGm_[outputOffset], outLocal, copyParams_);
    }
    outQueue_.FreeTensor(outLocal);
}

} // namespace MoeFinalizeRoutingV2

#endif // MOE_FINALIZE_ROUTING_V2_UNPERMUTE_FAST
