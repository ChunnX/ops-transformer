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
* \file moe_v3_gather_out_mxfp8.h
* \brief
*/
#ifndef MOE_V3_GATHER_OUT_MXFP8_H_REGBASE
#define MOE_V3_GATHER_OUT_MXFP8_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3GatherOutMxfp8 {
public:
    __aicore__ inline MoeV3GatherOutMxfp8(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR expandedScale, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements);
    __aicore__ inline void CopyXIn(int64_t xSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyXOut(int64_t xDstOffset, int64_t curLoopCols);
    __aicore__ inline void CopyScaleIn(int64_t scaleSrcOffset, int64_t curLoopCols);
    __aicore__ inline void CopyScaleOut(int64_t scaleDstOffset, int64_t curLoopCols);

private:
    __aicore__ inline void InitBasicParams(GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                           TPipe *tPipe);
    __aicore__ inline void ScatterCopyExpandedXandMXQuant(int64_t curExpertLoopOffset, int64_t curLoopElements,
                                                          LocalTensor<int32_t> &subRowIdxLocal);
    __aicore__ inline void GatherCopyExpandedXandMXQuant(int64_t curExpertLoopOffset, int64_t curLoopElements,
                                                         LocalTensor<int32_t> &subRowIdxLocal);
    TPipe *pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> xCopyInQueue_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, GATHER_OUT_BUFFER_NUM> scaleCopyInQueue_;
    TQue<QuePosition::VECIN, GATHER_OUT_BUFFER_NUM> expandedRowIdxCopyInQueue_;

    GlobalTensor<uint8_t> xUint8tGm_;
    GlobalTensor<uint8_t> xGscaleGm_;
    GlobalTensor<int32_t> sortedExpertIdxGm_;
    GlobalTensor<uint8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<uint8_t> expandedScaleGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;

    int64_t colsLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;

    int64_t indicesLoops_;

    int64_t perCoreIndicesElements_;
    int64_t lastCoreIndicesElements_;
    int64_t perCorePerLoopIndicesElements_;
    int64_t lastCorePerLoopIndicesElements_;
    int64_t curCorePerLoopIndicesElements_;
    int64_t curCoreLastLoopIndicesElements_;
    int64_t needCoreNum_;
    int64_t curCoreIndicesElements_;

    int64_t actualExpertNum_;
    int64_t expertTotalCount_;

    int64_t rowIdxType_ = 0;
    int64_t isInputScale_ = 0;
    int64_t ep_ = 0;
};

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::InitBasicParams(GM_ADDR workspace,
                                                               const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                               TPipe *tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();

    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;

    isInputScale_ = tilingData->isInputScale;
    rowIdxType_ = tilingData->rowIdxType;
    if (tilingData->epFullload == 0 && rowIdxType_ == GATHER) {
        ep_ = 0;
    } else {
        ep_ = 1;
    }

    colsLoops_ = tilingData->gatherOutComputeParamsOp.colsLoops;
    perLoopCols_ = tilingData->gatherOutComputeParamsOp.perLoopCols;
    lastLoopCols_ = tilingData->gatherOutComputeParamsOp.lastLoopCols;

    actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                         Align(actualExpertNum_, sizeof(int32_t)), 1);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        expertTotalCountGm_);
    expertTotalCount_ = expertTotalCountGm_.GetValue(0);

    perCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.perCorePerLoopIndicesElements;
    lastCorePerLoopIndicesElements_ = tilingData->gatherOutComputeParamsOp.lastCorePerLoopIndicesElements;
    perCoreIndicesElements_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreIndicesElements_);
    lastCoreIndicesElements_ = expertTotalCount_ - (needCoreNum_ - 1) * perCoreIndicesElements_;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreIndicesElements_ = lastCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(lastCorePerLoopIndicesElements_, curCoreIndicesElements_);
    } else {
        curCoreIndicesElements_ = perCoreIndicesElements_;
        curCorePerLoopIndicesElements_ = Min(perCorePerLoopIndicesElements_, curCoreIndicesElements_);
    }
    indicesLoops_ = Ceil(curCoreIndicesElements_, curCorePerLoopIndicesElements_);
    curCoreLastLoopIndicesElements_ = curCoreIndicesElements_ - (indicesLoops_ - 1) * curCorePerLoopIndicesElements_;
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR workspace,
                                                    GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR expandedScale,
                                                    const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    InitBasicParams(workspace, tilingData, tPipe);
    xUint8tGm_.SetGlobalBuffer((__gm__ uint8_t *)x, n_ * cols_);
    uint64_t alignScaleCols = Ops::Base::CeilDiv(cols_, SCALE_BLOCK_SIZE) * SCALE_THIRD_DIM_SIZE;
    xGscaleGm_.SetGlobalBuffer((__gm__ uint8_t *)scale, n_ * alignScaleCols);
    expandedXGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedX + blockIdx_ * perCoreIndicesElements_ * cols_,
                                 curCoreIndicesElements_ * cols_);
    expandedScaleGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedScale + blockIdx_ * perCoreIndicesElements_ *
                                     alignScaleCols, curCoreIndicesElements_ * alignScaleCols);

    pipe_->InitBuffer(expandedRowIdxCopyInQueue_, GATHER_OUT_BUFFER_NUM,
                      AlignBytes(curCorePerLoopIndicesElements_, sizeof(int32_t)));
    pipe_->InitBuffer(xCopyInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_, sizeof(uint8_t)));
    pipe_->InitBuffer(scaleCopyInQueue_, GATHER_OUT_BUFFER_NUM, AlignBytes(perLoopCols_ / SCALE_FACTOR_WITH_X,
                                                                           sizeof(uint8_t)));

    sortedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * perCoreIndicesElements_,
                                       Align(curCoreIndicesElements_, sizeof(int32_t)));

    if (ep_) {
        if (rowIdxType_ == SCATTER) {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        } else {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) +
                                              blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        }
    } else {
        if (rowIdxType_ == GATHER) {
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        } else {
            // TODO: 暂不支持，当前走不到该分支
            expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(n_ * k_, sizeof(int32_t)) +
                                              blockIdx_ * perCoreIndicesElements_,
                                              Align(curCoreIndicesElements_, sizeof(int32_t)));
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyExpertIn(int64_t curExpertLoopOffset, int64_t curLoopElements)
{
    LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopElements * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(subRowIdxLocal, expandedRowIdxGm_[curExpertLoopOffset], copyParams, padParams);
    expandedRowIdxCopyInQueue_.EnQue(subRowIdxLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyXIn(int64_t xSrcOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> xLocal = xCopyInQueue_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams0{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)),
                                  0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams0{false, 0, 0, 0};
    DataCopyPad(xLocal, xUint8tGm_[xSrcOffset], copyParams0, padParams0);
    xCopyInQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyXOut(int64_t xDstOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> xLocal = xCopyInQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyParams2{1, static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams2);
    xCopyInQueue_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyScaleIn(int64_t scaleSrcOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> scaleLocal = scaleCopyInQueue_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)),
                                  0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams1{false, 0, 0, 0};
    DataCopyPad(scaleLocal, xGscaleGm_[scaleSrcOffset], copyParams1, padParams1);
    scaleCopyInQueue_.EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::CopyScaleOut(int64_t scaleDstOffset, int64_t curLoopCols)
{
    LocalTensor<uint8_t> scaleLocal = scaleCopyInQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyParams3{1, static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad(expandedScaleGm_[scaleDstOffset], scaleLocal, copyParams3);
    scaleCopyInQueue_.FreeTensor(scaleLocal);
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::Process()
{
    if (blockIdx_ >= needCoreNum_) {
        return;
    }

    int64_t curLoopElements = curCorePerLoopIndicesElements_;
    for (int64_t indicesLoop = 0; indicesLoop < indicesLoops_; indicesLoop++) {
        if (indicesLoop == indicesLoops_ - 1) {
            curLoopElements = curCoreLastLoopIndicesElements_;
        }
        int64_t curExpertLoopOffset = indicesLoop * curCorePerLoopIndicesElements_;
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        CopyExpertIn(curExpertLoopOffset, curLoopElements);

        LocalTensor<int32_t> subRowIdxLocal = expandedRowIdxCopyInQueue_.DeQue<int32_t>();
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        if (ep_) {
            ScatterCopyExpandedXandMXQuant(curExpertLoopOffset, curLoopElements, subRowIdxLocal);
        } else {
            GatherCopyExpandedXandMXQuant(curExpertLoopOffset, curLoopElements, subRowIdxLocal);
        }

        expandedRowIdxCopyInQueue_.FreeTensor(subRowIdxLocal);
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::ScatterCopyExpandedXandMXQuant(int64_t curExpertLoopOffset,
                                                                               int64_t curLoopElements,
                                                                               LocalTensor<int32_t> &subRowIdxLocal)
{
    // Scatter 模式（EP 满载）：逐索引独立搬运 CopyIn → CopyOut
    // 输入地址分散，输出地址连续
    for (int64_t indicesIndex = 0; indicesIndex < curLoopElements; indicesIndex++) {
        int64_t rowIdx = subRowIdxLocal.GetValue(indicesIndex);
        int64_t scaleHCols = Ops::Base::CeilDiv<int64_t>(cols_, SCALE_BLOCK_SIZE) * SCALE_THIRD_DIM_SIZE;
        int64_t xSrcOffset = rowIdx / k_ * cols_;
        int64_t scaleSrcOffset = rowIdx / k_ * scaleHCols;
        int64_t scaleDstOffset = (curExpertLoopOffset + indicesIndex) * scaleHCols;
        int64_t xDstOffset = (curExpertLoopOffset + indicesIndex) * cols_;
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);

        int64_t curLoopCols = perLoopCols_;
        int64_t perLoopScaleHCols = perLoopCols_ / SCALE_FACTOR_WITH_X;
        int64_t curLoopScaleHCols = perLoopScaleHCols;
        for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
            if (colsLoop == colsLoops_ - 1) {
                curLoopCols = lastLoopCols_;
                curLoopScaleHCols = Ops::Base::CeilDiv<int64_t>(curLoopCols, SCALE_BLOCK_SIZE) *
                                    SCALE_THIRD_DIM_SIZE;
            }
            if (isInputScale_ == 1) {
                CopyScaleIn(scaleSrcOffset + colsLoop * perLoopScaleHCols, curLoopScaleHCols);
                CopyScaleOut(scaleDstOffset + colsLoop * perLoopScaleHCols, curLoopScaleHCols);
            }
            int64_t colsLoopOffset = colsLoop * perLoopCols_;
            CopyXIn(xSrcOffset + colsLoopOffset, curLoopCols);
            CopyXOut(xDstOffset + colsLoopOffset, curLoopCols);
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3GatherOutMxfp8<T>::GatherCopyExpandedXandMXQuant(int64_t curExpertLoopOffset,
                                                                              int64_t curLoopElements,
                                                                              LocalTensor<int32_t> &subRowIdxLocal)
{
    // Gather 模式（非 EP）：外层按列分块，内层按源行遍历
    // 每行 CopyIn 一次，CopyOut 到所有映射到该行的目标位置，避免重复读取
    int64_t scaleHCols = Ops::Base::CeilDiv<int64_t>(cols_, SCALE_BLOCK_SIZE) * SCALE_THIRD_DIM_SIZE;

    int64_t currentLoopStartRow = curExpertLoopOffset / k_;
    int64_t currentLoopLastRow = (curExpertLoopOffset + curLoopElements - 1) / k_;

    for (int64_t colsLoop = 0; colsLoop < colsLoops_; colsLoop++) {
        int64_t curLoopCols = (colsLoop == colsLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
        int64_t perLoopScaleHCols = perLoopCols_ / SCALE_FACTOR_WITH_X;
        int64_t curLoopScaleHCols = (colsLoop == colsLoops_ - 1) ?
            Ops::Base::CeilDiv<int64_t>(curLoopCols, SCALE_BLOCK_SIZE) * SCALE_THIRD_DIM_SIZE :
            perLoopScaleHCols;

        int64_t colsLoopOffset = colsLoop * perLoopCols_;

        // 每次 colLoop 迭代重置行追踪变量，确保每个列分块都能正确处理所有索引
        int64_t globalSortIdx = curExpertLoopOffset;
        int64_t curLoopRow = 0;

        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            // CopyIn 源行数据一次
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            CopyXIn(row * cols_ + colsLoopOffset, curLoopCols);
            if (isInputScale_ == 1) {
                CopyScaleIn(row * scaleHCols + colsLoop * perLoopScaleHCols, curLoopScaleHCols);
            }

            // 获取本地 tensor
            LocalTensor<uint8_t> xLocal = xCopyInQueue_.DeQue<uint8_t>();
            LocalTensor<uint8_t> scaleLocal;
            if (isInputScale_ == 1) {
                scaleLocal = scaleCopyInQueue_.DeQue<uint8_t>();
            }

            // CopyOut 到所有映射到该源行的目标位置
            while (curLoopRow < curLoopElements && globalSortIdx / k_ == row) {
                int32_t outIndex = subRowIdxLocal.GetValue(curLoopRow);
                curLoopRow++;
                globalSortIdx++;
                if (outIndex < 0) {
                    continue;
                }
                int64_t xDstOffset = static_cast<int64_t>(outIndex) * cols_ + colsLoopOffset;
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLoopCols * sizeof(uint8_t)), 0, 0, 0};
                DataCopyPad(expandedXGm_[xDstOffset], xLocal, copyParams);

                if (isInputScale_ == 1) {
                    int64_t scaleDstOffset = static_cast<int64_t>(outIndex) * scaleHCols +
                                             colsLoop * perLoopScaleHCols;
                    DataCopyExtParams scaleCopyParams{1,
                                                      static_cast<uint32_t>(curLoopScaleHCols * sizeof(uint8_t)),
                                                      0, 0, 0};
                    DataCopyPad(expandedScaleGm_[scaleDstOffset], scaleLocal, scaleCopyParams);
                }
            }

            xCopyInQueue_.FreeTensor(xLocal);
            if (isInputScale_ == 1) {
                scaleCopyInQueue_.FreeTensor(scaleLocal);
            }
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_OUT_MXFP8_H_REGBASE