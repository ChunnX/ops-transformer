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
 * \file tri_inverse_mix.h
 * \brief MixInverse 64x64 lower-triangular inverse kernel.
 */

#ifndef TRI_INVERSE_MIX_KERNEL_H
#define TRI_INVERSE_MIX_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "tri_inverse_tiling_data.h"

namespace TriInverse {
using namespace AscendC;
using namespace matmul;

using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using cT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using TriInverseMT = matmul::MatmulImpl<aT1, bT1, cT1>;

constexpr uint32_t BUFFER_NUM_ONE = 1;
constexpr uint32_t TASK_RATIO = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BROADCAST_AXIS = 2;
constexpr uint32_t INVERSE_SHAPE = 32;
constexpr uint32_t INVERSE_COUNT = 5;
constexpr uint32_t TRI_INVERSE_UB_BYTES = 64 * 1024;

class MixInverseOp {
public:
    __aicore__ inline MixInverseOp(TPipe *pipe, const TriInverseTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR ccIn, GM_ADDR ccOut)
    {
        tileNum_ = static_cast<uint32_t>(tiling_->tileNum);
        chunkSize_ = static_cast<uint32_t>(tiling_->chunkSize);
        paraNum_ = static_cast<uint32_t>(tiling_->stageOneParaNum);
        coreNum_ = static_cast<uint32_t>(tiling_->aiCoreNum);
        halfChunkSize_ = chunkSize_ / TASK_RATIO;
        ccOffset_ = chunkSize_ * chunkSize_;

        ccInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ccIn), tileNum_ * ccOffset_);
        ccOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ccOut), tileNum_ * ccOffset_);

        coreIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreIdx_ /= TASK_RATIO;
            subBlockIdx_ = GetSubBlockIdx();
            subOffset_ = subBlockIdx_ * halfChunkSize_;
            InitLocalBuffers();
            InitGatherBuffer();
        }
        if ASCEND_IS_AIC {
            mm_.Init(&tiling_->matmulTilingFp32, pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t tailTileNum = tileNum_ / coreNum_;
        uint32_t formerTileNum = tailTileNum + 1;
        uint32_t formerCoreNum = tileNum_ % coreNum_;
        uint32_t start = 0;
        uint32_t end = 0;
        if (coreIdx_ < formerCoreNum) {
            start = coreIdx_ * formerTileNum;
            end = start + formerTileNum;
        } else {
            start = formerCoreNum * formerTileNum + (coreIdx_ - formerCoreNum) * tailTileNum;
            end = start + tailTileNum;
        }

        for (uint32_t taskId = start; taskId < end; taskId += paraNum_) {
            uint32_t curParaNum = paraNum_ < end - taskId ? paraNum_ : end - taskId;
            ProcessParaTiles(taskId, curParaNum);
        }
    }

private:
    __aicore__ inline void InitLocalBuffers()
    {
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, chunkSize_ * halfChunkSize_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, chunkSize_ * halfChunkSize_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(tmpBuff_, TRI_INVERSE_UB_BYTES);

        uint32_t buffOffset = 0;
        attnUbFloat_ = tmpBuff_.GetWithOffset<float>(chunkSize_ * halfChunkSize_, buffOffset);
        buffOffset += chunkSize_ * halfChunkSize_ * sizeof(float);

        inverseBuffer_ = tmpBuff_.GetWithOffset<float>(halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT, buffOffset);
        buffOffset += halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT * sizeof(float);

        inverseGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(INVERSE_SHAPE, buffOffset);
        buffOffset += INVERSE_SHAPE * sizeof(uint32_t);

        inverseRes_ = tmpBuff_.GetWithOffset<float>(chunkSize_ * halfChunkSize_, buffOffset);
    }

    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < INVERSE_SHAPE; ++i) {
            inverseGatherBuffer_.SetValue<uint32_t>(i, (i * chunkSize_) * sizeof(float));
        }
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void ProcessParaTiles(uint32_t taskId, uint32_t curParaNum)
    {
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < curParaNum; ++i) {
                MixInverseAIVTile(taskId + i);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(0x6);
            for (uint32_t i = 0; i < curParaNum; ++i) {
                MixInverseMMCompute((taskId + i) * ccOffset_);
            }
        }
    }

    __aicore__ inline void MixInverseAIVTile(uint32_t tileId)
    {
        uint64_t tileOffset = static_cast<uint64_t>(tileId) * ccOffset_;
        uint64_t halfOffset = subOffset_ * chunkSize_;
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;

        DataCopyInBf16(curVecLen, ccInGm_[tileOffset + halfOffset]);
        auto bf16Local = inQueue_.DeQue<bfloat16_t>();
        Cast(attnUbFloat_, bf16Local, RoundMode::CAST_NONE, curVecLen);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Local);

        Muls(inverseRes_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();

        MixInverseAIV(subOffset_, INVERSE_SHAPE);
        auto tmp = outQueue_.AllocTensor<bfloat16_t>();
        Cast(tmp, inverseRes_, RoundMode::CAST_RINT, curVecLen);
        outQueue_.EnQue(tmp);
        DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, ccOutGm_[tileOffset + halfOffset]);
    }

    __aicore__ inline void MixInverseAIV(uint64_t offset, uint32_t inverseVecLen)
    {
        PipeBarrier<PIPE_V>();
        uint64_t inverseBufferOffset = 0;
        auto row = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto col = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen + inverseVecLen;
        auto yLocal = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto ei = inverseBuffer_[inverseBufferOffset];

        Duplicate(ei, static_cast<float>(0.0), inverseVecLen);
        Duplicate(yLocal, static_cast<float>(0.0), inverseVecLen * inverseVecLen);
        inverseRes_.SetValue(offset, static_cast<float>(1.0));

        uint32_t srcShape[2] = {1, inverseVecLen};
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
        for (int i = 1; i < static_cast<int>(inverseVecLen); ++i) {
            uint32_t curI = i - 1;
            uint32_t validRows = inverseVecLen - i;
            Gather(col, attnUbFloat_[offset + i * chunkSize_ + curI], inverseGatherBuffer_, static_cast<uint32_t>(0),
                   validRows);
            PipeBarrier<PIPE_V>();
            uint32_t dstShape[2] = {validRows, inverseVecLen};
            uint32_t colSrcShape[2] = {validRows, 1};
            Broadcast<float, BROADCAST_AXIS, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
            Broadcast<float, BROADCAST_AXIS, 0>(row, inverseRes_[offset + curI * chunkSize_], dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
            PipeBarrier<PIPE_V>();
            ei.SetValue(i - 1, static_cast<float>(0.0));
            ei.SetValue(i, static_cast<float>(1.0));
            eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventID);
            WaitFlag<HardEvent::S_V>(eventID);
            Sub(inverseRes_[offset + i * chunkSize_], ei, yLocal[i * inverseVecLen], inverseVecLen);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DataCopyInBf16(uint64_t len, GlobalTensor<bfloat16_t> src)
    {
        DataCopyPadExtParams<bfloat16_t> padParams;
        DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(bfloat16_t)), 0, 0, 0};
        auto local = inQueue_.AllocTensor<bfloat16_t>();
        DataCopyPad(local, src, params, padParams);
        inQueue_.EnQue(local);
    }

    __aicore__ inline void DataCopyOutBf16(uint32_t rows, uint32_t cols, uint32_t colsAligned,
                                           GlobalTensor<bfloat16_t> dst)
    {
        auto tmpLocal = outQueue_.DeQue<bfloat16_t>();
        uint32_t srcStride = (colsAligned - cols) * sizeof(bfloat16_t) / BLOCK_SIZE;
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                 srcStride, 0, 0};
        DataCopyPad(dst, tmpLocal, params);
        outQueue_.FreeTensor(tmpLocal);
    }

    __aicore__ inline void MixInverseMMCompute(uint64_t offset)
    {
        uint64_t leftDown = offset + chunkSize_ * INVERSE_SHAPE;
        uint64_t rightDown = leftDown + INVERSE_SHAPE;
        MixInverseMMProcess(ccOutGm_[leftDown], ccOutGm_[offset], ccOutGm_[leftDown]);
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
        MixInverseMMProcess(ccOutGm_[rightDown], ccOutGm_[leftDown], ccOutGm_[leftDown]);
        eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    __aicore__ inline void MixInverseMMProcess(GlobalTensor<bfloat16_t> x, GlobalTensor<bfloat16_t> y,
                                               GlobalTensor<bfloat16_t> z)
    {
        mm_.SetOrgShape(chunkSize_, chunkSize_, chunkSize_);
        mm_.SetSingleShape(INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE);
        mm_.SetTensorA(x);
        mm_.SetTensorB(y);
        mm_.IterateAll(z);
        mm_.End();
    }

    TPipe *pipe_;
    const TriInverseTilingData *tiling_;
    TriInverseMT mm_;

    uint32_t tileNum_;
    uint32_t chunkSize_;
    uint32_t halfChunkSize_;
    uint32_t coreNum_;
    uint32_t paraNum_;
    uint32_t subBlockIdx_;
    uint32_t subOffset_;
    uint32_t coreIdx_;
    uint64_t ccOffset_;

    GlobalTensor<bfloat16_t> ccInGm_;
    GlobalTensor<bfloat16_t> ccOutGm_;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;

    LocalTensor<float> attnUbFloat_;
    LocalTensor<float> inverseBuffer_;
    LocalTensor<float> inverseRes_;
    LocalTensor<uint32_t> inverseGatherBuffer_;
};

using TriInverseOp = MixInverseOp;
} // namespace TriInverse

#endif // TRI_INVERSE_MIX_KERNEL_H
