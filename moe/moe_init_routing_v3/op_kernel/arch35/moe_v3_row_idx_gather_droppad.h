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
class MoeV3SrcToDstWithCapacity {
public:
    __aicore__ inline MoeV3SrcToDstWithCapacity(){};
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
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::AssistInit()
{
    // ========== DropPad填充初始化：创建零值tensor（关键步骤4）==========
    // DropPad模式需要填充不足expertCapacity的专家位置为0
    // 预先初始化一个perLoopCols大小的零值tensor用于后续填充
    // ==============================================================
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

    // ========== DropPad跨核专家处理：读取前序核信息（关键步骤5）==========
    // DropPad模式下，需要处理跨核专家的连续性问题：
    // 问题：同一个专家的token可能分布在不同核上，需要确保写入位置连续
    // 
    // 处理步骤：
    // 1. 读取上一个核的最后专家ID（lastCoreExpertId）和token数量（lastCoreExpertIdNum）
    // 2. 向前遍历所有更早的核，累计处理同一个专家的token数量
    //    - 如果上个核的最后专家ID等于当前核的首个专家ID，说明专家跨核
    //    - 需要将所有前序核处理该专家的token数量累加到lastCoreExpertIdNum_
    // 3. 当前核开始写入时，起始位置为：lastExpertId * expertCapacity + lastCoreExpertIdNum_
    // 
    // 示例：
    // 核0: 处理专家5的10个token → 写入位置: 5*expertCapacity + 0-9
    // 核1: 处理专家5的20个token → 需读取核0信息，写入位置: 5*expertCapacity + 10-29
    // ==============================================================
    if (this->blockIdx_ != 0) {
        this->lastCoreExpertId_ = expertIdxValueGm_.GetValue((this->blockIdx_ - 1) * 2);
        this->lastCoreExpertIdNum_ = expertIdxValueGm_.GetValue((this->blockIdx_ - 1) * 2 + 1);
        // 向前遍历所有处理同一专家的前序核
        for (int64_t i = this->blockIdx_ - 2; i >= 0; i--) {
            int32_t lastExpertIdx = expertIdxValueGm_.GetValue(i * 2);
            if (lastExpertIdx < this->lastCoreExpertId_) {
                // 如果上个核的专家ID小于当前核的最后专家ID，说明不再连续，停止累加
                break;
            }
            int32_t lastExpertNum = expertIdxValueGm_.GetValue(i * 2 + 1);
            this->lastCoreExpertIdNum_ += lastExpertNum;
        }
    }
}

template <typename T>
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::CopyOutRemain()
{
    if (this->isInputScale_ != 1) {
        return;
    }
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(expandedScaleGm_[index], this->scaleZeroLocal_, copyParams);
}

template <typename T>
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::CopyIn(int64_t progress)
{
    LocalTensor<int32_t> inLocal = copyInQueue_.AllocTensor<int32_t>();
    int64_t length = Align(currentLoopRows_, sizeof(int32_t));
    DataCopy(inLocal, expandDstToSrcRowGm_[progress * perLoopRows_], length);
    DataCopy(inLocal[length], expandedExpertIdxGm_[progress * perLoopRows_], length);
    copyInQueue_.EnQue<int32_t>(inLocal);
}

template <typename T>
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::CopyOut(int64_t progress)
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
        // ========== DropPad填充处理：补齐剩余专家容量（关键步骤6）==========
        // DropPad模式下，最后一个核需要补齐所有专家的剩余容量
        // 确保每个专家都达到expertCapacity，不足部分填充为0
        // 
        // 处理逻辑：
        // 1. 从lastExpertId开始（当前核处理的最后专家）
        // 2. 遍历到expertNum-1的所有专家
        // 3. 对每个专家，从tokenCount开始填充到expertCapacity
        // 4. 填充内容为预先初始化的零值tensor
        // 
        // 示例（假设expertCapacity=128，专家5只处理了100个token）：
        // - 专家5写入位置：5*128 + 0-99（实际token）
        // - 需要填充：5*128 + 100-127（填充0）
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
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::CopyOutRemain()
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
                DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(col * sizeof(T)), 0, 0, 0};
                if constexpr (IsSameType<T, hifloat8_t>::value) {
                    DataCopyPad(expandedXUint8Gm_[index * this->cols_ + i * this->perLoopCols_],
                                this->outTmpUint8Local_, copyParams);
                } else {
                    DataCopyPad(expandedXGm_[index * this->cols_ + i * this->perLoopCols_], this->outTmpLocal_, copyParams);
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
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::SyncAll()
{
    if (coreNum_ == 1) {
        return;
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T>
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::Init(GM_ADDR expandedRowIdx, GM_ADDR expandedX,
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
        // ========== DropPad输出布局：expandedX为3D tensor（关键步骤1）==========
        // DropPad模式输出为[expertNum, expertCapacity, cols]的三维tensor
        // 每个专家最多处理expertCapacity个token，不足部分填充0
        // 写入位置：expertIdx * expertCapacity + tokenIdx
        // ==============================================================
        expandedXGm_.SetGlobalBuffer((__gm__ T *)expandedX,
                                     this->expertNum_ * this->expertCapacity_ * this->cols_);
    }

    if (this->isInputScale_ == 1) {
        expandedScaleGm_.SetGlobalBuffer((__gm__ float *)expandedScale,
                                         this->expertNum_ * this->expertCapacity_);
    }

    // ========== DropPad workspace复用策略（关键步骤2）==========
    // workspace通过时序复用实现空间优化：
    // 1. expandedExpertIdxGm：复用expertCountTempGm的位置（Sort阶段结束后释放）
    //    位置Align(n*k)*2开始，每个核perCoreRows个int32
    // 2. expandDstToSrcRowGm：复用workspaceGms[0]的位置（Sort阶段结束后释放）
    //    位置Align(n*k)*2 + Align(actualExpertNum)开始，每个核perCoreRows个int32
    // 3. expertIdxValueGm：读取expert_tokens_count输出的跨核专家信息
    //    位置Align(n*k)*2 + Align(actualExpertNum)*2，大小coreNum*2个int32
    // ==============================================================
    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                             this->blockIdx_ * this->srcToDstTilingData_->perCoreRows,
                                         Align(this->coreRows_, sizeof(int32_t)));
    expandDstToSrcRowGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + length +
                                             this->blockIdx_ * this->srcToDstTilingData_->perCoreRows,
                                         Align(this->coreRows_, sizeof(int32_t)));
    
    // ========== DropPad关键数据读取：expertIdxValueGm（关键步骤3）==========
    // expertIdxValueGm存储每个核的最后专家信息（由expert_tokens_count写入）
    // 每个核2个int32：[finalExpertId, expertTokenValue]
    // 用于处理跨核专家的连续性问题：
    // - 当前核的首个专家可能与上个核的最后专家相同
    // - 需要读取上个核的expertTokenValue来计算正确的起始位置
    // ==============================================================
    expertIdxValueGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + length * 2 + Align(tilingData->actualExpertNum, sizeof(int32_t)) * 2,
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
__aicore__ inline void MoeV3SrcToDstWithCapacity<T>::Process()
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