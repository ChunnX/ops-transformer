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
 * \file moe_v3_row_idx_gather_droppad.h
 * \brief
 */
#ifndef MOE_V3_ROW_IDX_GATHER_DROPPAD_H_REGBASE
#define MOE_V3_ROW_IDX_GATHER_DROPPAD_H_REGBASE

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3RowIdxGatherDropPad {
public:
    __aicore__ inline MoeV3RowIdxGatherDropPad(){};
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR expandedScale, GM_ADDR workspace,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void CopyOut(int64_t progress);
    __aicore__ inline void CopyOutRemain();
    __aicore__ inline void SyncAll();
    __aicore__ inline void AssistInit();
    __aicore__ inline void CopyScaleZeroOut(int64_t index);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> copyInQueue_;
    TQue<QuePosition::VECOUT, 1> copyOutQueue_;
    TQue<QuePosition::VECOUT, 1> copyOutZeroQueue_;
    TQue<QuePosition::VECOUT, 1> scaleZeroOutQueue_;

    GlobalTensor<int32_t> expandDstToSrcRowGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expertIdxValueGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<T> expandedXGm_;
    GlobalTensor<uint8_t> expandedXUint8Gm_;
    GlobalTensor<float> expandedScaleGm_;

    LocalTensor<T> outTmpLocal_;
    LocalTensor<uint8_t> outTmpUint8Local_;
    LocalTensor<float> scaleZeroLocal_;

    const MoeV3Arch35SrcToDstCapacityComputeTilingData *srcToDstTilingData_;
    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t totalLength_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t expertCapacity_;
    int64_t expertNum_;
    int64_t cols_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t isInputScale_;
    int64_t quantMode_;

    int64_t tokenCount_ = 0;
    int32_t lastExpertId_ = -1;
    int32_t lastCoreExpertId_ = 0;
    int32_t lastCoreExpertIdNum_ = 0;
};

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::AssistInit()
{
    if constexpr (IsSameType<T, hifloat8_t>::value) {
        LocalTensor<uint8_t> outLocal = copyOutZeroQueue_.AllocTensor<uint8_t>();
        Duplicate<uint8_t>(outLocal, static_cast<uint8_t>(0), this->perLoopCols_);
        copyOutZeroQueue_.EnQue<uint8_t>(outLocal);
    } else {
        LocalTensor<T> outLocal = copyOutZeroQueue_.AllocTensor<T>();
        Duplicate<T>(outLocal, static_cast<T>(0), this->perLoopCols_);
        copyOutZeroQueue_.EnQue<T>(outLocal);
    }

    if (this->isInputScale_ == 1) {
        LocalTensor<float> scaleLocal = scaleZeroOutQueue_.AllocTensor<float>();
        Duplicate<float>(scaleLocal, static_cast<float>(0), 1);
        scaleZeroOutQueue_.EnQue<float>(scaleLocal);
    }

    if (this->blockIdx_ != 0) {
        this->lastCoreExpertId_ = expertIdxValueGm_.GetValue((this->blockIdx_ - 1) * 2);
        this->lastCoreExpertIdNum_ = expertIdxValueGm_.GetValue((this->blockIdx_ - 1) * 2 + 1);
        for (int64_t i = this->blockIdx_ - 2; i >= 0; i--) {
            int32_t lastExpertIdx = expertIdxValueGm_.GetValue(i * 2);
            if (lastExpertIdx < this->lastCoreExpertId_) {
                break;
            }
            int32_t lastExpertNum = expertIdxValueGm_.GetValue(i * 2 + 1);
            this->lastCoreExpertIdNum_ += lastExpertNum;
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::CopyScaleZeroOut(int64_t index)
{
    if (this->isInputScale_ != 1) {
        return;
    }
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(expandedScaleGm_[index], this->scaleZeroLocal_, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::CopyIn(int64_t progress)
{
    LocalTensor<int32_t> inLocal = copyInQueue_.AllocTensor<int32_t>();
    int64_t length = Align(currentLoopRows_, sizeof(int32_t));
    DataCopy(inLocal, expandDstToSrcRowGm_[progress * perLoopRows_], length);
    DataCopy(inLocal[length], expandedExpertIdxGm_[progress * perLoopRows_], length);
    copyInQueue_.EnQue<int32_t>(inLocal);
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::CopyOut(int64_t progress)
{
    LocalTensor<int32_t> inLocal = copyInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> outLocal = copyOutQueue_.AllocTensor<int32_t>();
    int64_t length = Align(currentLoopRows_, sizeof(int32_t));
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    if (this->lastExpertId_ == -1) {
        this->lastExpertId_ = this->lastCoreExpertId_;
        this->tokenCount_ = this->lastCoreExpertIdNum_;
    }
    for (int64_t idx = 0; idx < currentLoopRows_; idx++) {
        int32_t expertIdx = inLocal[length].GetValue(idx);
        int32_t index = 0;
        while (this->lastExpertId_ < expertIdx) {
            while (this->tokenCount_ < this->expertCapacity_) {
                index = this->lastExpertId_ * this->expertCapacity_ + this->tokenCount_;
                int64_t col = this->perLoopCols_;
                for (int64_t i = 0; i < this->colLoops_; i++) {
                    if (i == this->colLoops_ - 1) {
                        col = this->lastLoopCols_;
                    }
                    DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(col * sizeof(T)), 0,
                                                   0, 0};
                    if constexpr (IsSameType<T, hifloat8_t>::value) {
                        DataCopyPad(expandedXUint8Gm_[index * this->cols_ + i * this->perLoopCols_],
                                    this->outTmpUint8Local_, copyParams1);
                    } else {
                        DataCopyPad(expandedXGm_[index * this->cols_ + i * this->perLoopCols_], this->outTmpLocal_,
                                    copyParams1);
                    }
                }
                CopyScaleZeroOut(index);
                this->tokenCount_++;
            }
            this->tokenCount_ = 0;
            this->lastExpertId_++;
        }

        if (this->tokenCount_ < this->expertCapacity_) {
            int32_t outOffset = inLocal.GetValue(idx);
            index = expertIdx * this->expertCapacity_ + this->tokenCount_;
            outLocal.SetValue(0, index);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(expandedRowIdxGm_[outOffset], outLocal, copyParams);
            this->tokenCount_++;
        }
    }
    copyInQueue_.FreeTensor(inLocal);
    copyOutQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::CopyOutRemain()
{
    if (this->blockIdx_ != this->srcToDstTilingData_->needCoreNum - 1) {
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            copyOutZeroQueue_.FreeTensor(this->outTmpUint8Local_);
        } else {
            copyOutZeroQueue_.FreeTensor(this->outTmpLocal_);
        }
        if (this->isInputScale_ == 1) {
            scaleZeroOutQueue_.FreeTensor(this->scaleZeroLocal_);
        }
        return;
    }
    while (this->lastExpertId_ < this->expertNum_) {
        while (this->tokenCount_ < this->expertCapacity_) {
            int32_t index = this->lastExpertId_ * this->expertCapacity_ + this->tokenCount_;
            int64_t col = this->perLoopCols_;
            for (int64_t i = 0; i < this->colLoops_; i++) {
                if (i == this->colLoops_ - 1) {
                    col = this->lastLoopCols_;
                }
                DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(col * sizeof(T)),
                                             0, 0, 0};
                if constexpr (IsSameType<T, hifloat8_t>::value) {
                    DataCopyPad(expandedXUint8Gm_[index * this->cols_ + i * this->perLoopCols_],
                                this->outTmpUint8Local_, copyParams);
                } else {
                    DataCopyPad(expandedXGm_[index * this->cols_ + i * this->perLoopCols_],
                                this->outTmpLocal_, copyParams);
                }
            }
            CopyScaleZeroOut(index);
            this->tokenCount_++;
        }
        this->tokenCount_ = 0;
        this->lastExpertId_++;
    }
    if constexpr (IsSameType<T, hifloat8_t>::value) {
        copyOutZeroQueue_.FreeTensor(this->outTmpUint8Local_);
    } else {
        copyOutZeroQueue_.FreeTensor(this->outTmpLocal_);
    }
    if (this->isInputScale_ == 1) {
        scaleZeroOutQueue_.FreeTensor(this->scaleZeroLocal_);
    }
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::SyncAll()
{
    if (coreNum_ == 1) {
        return;
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                         GM_ADDR expandedScale, GM_ADDR workspace,
                                                         const MoeInitRoutingV3Arch35TilingData *tilingData,
                                                         TPipe *tPipe)
{
    int64_t blockNum = GetBlockNum();
    pipe_ = tPipe;
    this->blockIdx_ = GetBlockIdx();

    this->coreNum_ = tilingData->coreNum;
    this->totalLength_ = tilingData->n * tilingData->k;
    this->srcToDstTilingData_ = &(tilingData->srcToDstDropPadParamsOp);
    this->expertNum_ = tilingData->expertNum;
    this->expertCapacity_ = tilingData->expertCapacity;
    this->cols_ = tilingData->cols;
    this->quantMode_ = tilingData->quantMode;
    this->isInputScale_ = tilingData->isInputScale;

    if (this->blockIdx_ == this->srcToDstTilingData_->needCoreNum - 1) {
        this->coreRows_ = this->srcToDstTilingData_->lastCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData_->lastCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData_->lastCoreLastLoopRows;
        this->rowLoops_ = this->srcToDstTilingData_->lastCoreLoops;
    } else {
        this->coreRows_ = this->srcToDstTilingData_->perCoreRows;
        this->perLoopRows_ = this->srcToDstTilingData_->perCorePerLoopRows;
        this->lastLoopRows_ = this->srcToDstTilingData_->perCoreLastLoopRows;
        this->rowLoops_ = this->srcToDstTilingData_->perCoreLoops;
    }
    this->perLoopCols_ = this->srcToDstTilingData_->perLoopCols;
    this->lastLoopCols_ = this->srcToDstTilingData_->lastLoopCols;
    this->colLoops_ = this->srcToDstTilingData_->colLoops;

    int64_t length = Align(this->totalLength_, sizeof(int32_t));
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, length);

    if constexpr (IsSameType<T, hifloat8_t>::value) {
        expandedXUint8Gm_.SetGlobalBuffer((__gm__ uint8_t *)expandedX,
                                          this->expertNum_ * this->expertCapacity_ * this->cols_);
    } else {
        expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX,
                                     this->expertNum_ * this->expertCapacity_ * this->cols_);
    }

    if (this->isInputScale_ == 1) {
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale,
                                         this->expertNum_ * this->expertCapacity_);
    }

    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                             this->blockIdx_ * this->srcToDstTilingData_->perCoreRows,
                                         Align(this->coreRows_, sizeof(int32_t)));
    expandDstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + length +
                                             this->blockIdx_ * this->srcToDstTilingData_->perCoreRows,
                                         Align(this->coreRows_, sizeof(int32_t)));
    // expertIdxValueGm偏移地址必须与expert_tokens_count.h保持一致
    // expert_tokens_count.h偏移: Align(n*k)*2 + Align(actualExpertNum)*2
    int64_t actualExpertNumOffset = Align(tilingData->actualExpertNum, sizeof(int32_t)) * 2;
    expertIdxValueGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + length * 2 + actualExpertNumOffset,
        this->coreNum_ * 2);

    pipe_->InitBuffer(copyInQueue_, 1, AlignBytes(this->perLoopRows_, sizeof(int32_t)) * 2);
    pipe_->InitBuffer(copyOutQueue_, 1, AlignBytes(INT32_ONE_BLOCK_NUM, sizeof(int32_t)));
    if constexpr (IsSameType<T, hifloat8_t>::value) {
        pipe_->InitBuffer(copyOutZeroQueue_, 1, AlignBytes(this->perLoopCols_, sizeof(uint8_t)));
    } else {
        pipe_->InitBuffer(copyOutZeroQueue_, 1, AlignBytes(this->perLoopCols_, sizeof(T)));
    }
    if (this->isInputScale_ == 1) {
        pipe_->InitBuffer(scaleZeroOutQueue_, 1, AlignBytes(1, sizeof(float)));
    }
}

template <typename T>
__aicore__ inline void MoeV3RowIdxGatherDropPad<T>::Process()
{
    if (this->blockIdx_ < this->srcToDstTilingData_->needCoreNum) {
        AssistInit();
        if constexpr (IsSameType<T, hifloat8_t>::value) {
            this->outTmpUint8Local_ = copyOutZeroQueue_.DeQue<uint8_t>();
        } else {
            this->outTmpLocal_ = copyOutZeroQueue_.DeQue<T>();
        }
        if (this->isInputScale_ == 1) {
            this->scaleZeroLocal_ = scaleZeroOutQueue_.DeQue<float>();
        }
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < this->rowLoops_; loop++) {
            if (loop == this->rowLoops_ - 1) {
                currentLoopRows_ = lastLoopRows_;
            }
            CopyIn(loop);
            CopyOut(loop);
        }
        CopyOutRemain();
    }
    this->SyncAll();
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_ROW_IDX_GATHER_DROPPAD_H_REGBASE