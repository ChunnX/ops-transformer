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
constexpr uint32_t BF16_CACHELINE_ELEMS = BLOCK_SIZE / sizeof(bfloat16_t);
constexpr uint32_t HALF_CACHELINE_ELEMS = BLOCK_SIZE / sizeof(half);
constexpr uint32_t BROADCAST_AXIS = 2;
constexpr uint32_t INVERSE_SHAPE = 32;
constexpr uint32_t INVERSE_COUNT = 5;
constexpr uint32_t INVERSE_BUFFER_ELEMS =
    INVERSE_SHAPE * INVERSE_SHAPE + (INVERSE_SHAPE * INVERSE_SHAPE + INVERSE_SHAPE) +
    INVERSE_SHAPE * INVERSE_SHAPE + INVERSE_SHAPE;
constexpr uint32_t TRI_INVERSE_UB_BYTES = 48 * 1024;
constexpr uint32_t DIAG16_SHAPE = 16;
constexpr uint32_t DIAG16_ELEMS = DIAG16_SHAPE * DIAG16_SHAPE;
constexpr uint32_t RAW_HALF_FRACTAL_ELEMS = 512 / sizeof(half);
constexpr uint32_t AIC_RAW_L0_AB_BYTES = 32 * 1024;
constexpr uint32_t AIC_RAW_L0_AB_HALF_PINGPONG_BYTES = AIC_RAW_L0_AB_BYTES / 2;
constexpr uint32_t AIC_RAW_L0_C_BYTES = 64 * 1024;
constexpr uint32_t COMPOSE32_SHAPE = INVERSE_SHAPE;
constexpr uint32_t COMPOSE32_ELEMS = COMPOSE32_SHAPE * COMPOSE32_SHAPE;
constexpr uint32_t AIC_MIX_L1_BYTES = (COMPOSE32_ELEMS * 5 + DIAG16_ELEMS * 4) * sizeof(half);
constexpr uint8_t RAW_MMAD_NON_FINAL_UNIT_FLAG = 2;
constexpr uint8_t RAW_MMAD_FINAL_UNIT_FLAG = 3;
constexpr bool RAW_MMAD_CMATRIX_SOURCE_L0C = false;
constexpr bool RAW_MMAD_INIT_C = true;
constexpr bool RAW_MMAD_ACCUMULATE_C = false;
constexpr int16_t FP16_ONE_BITS = 15360;

#define TRI_INV_I16_NZ_ROW0 FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW1 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW2 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW3 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW4 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW5 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW6 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW7 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW8 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW9 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW10 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW11 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0
#define TRI_INV_I16_NZ_ROW12 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0
#define TRI_INV_I16_NZ_ROW13 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0
#define TRI_INV_I16_NZ_ROW14 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0
#define TRI_INV_I16_NZ_ROW15 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS
#define TRI_INV_I16_NZ_FRACTAL \
    TRI_INV_I16_NZ_ROW0, TRI_INV_I16_NZ_ROW1, TRI_INV_I16_NZ_ROW2, TRI_INV_I16_NZ_ROW3, \
    TRI_INV_I16_NZ_ROW4, TRI_INV_I16_NZ_ROW5, TRI_INV_I16_NZ_ROW6, TRI_INV_I16_NZ_ROW7, \
    TRI_INV_I16_NZ_ROW8, TRI_INV_I16_NZ_ROW9, TRI_INV_I16_NZ_ROW10, TRI_INV_I16_NZ_ROW11, \
    TRI_INV_I16_NZ_ROW12, TRI_INV_I16_NZ_ROW13, TRI_INV_I16_NZ_ROW14, TRI_INV_I16_NZ_ROW15

static constexpr volatile __gm__ int16_t TRI_INV_IDENTITY16_NZ[RAW_HALF_FRACTAL_ELEMS] = {
    TRI_INV_I16_NZ_FRACTAL};

#undef TRI_INV_I16_NZ_ROW0
#undef TRI_INV_I16_NZ_ROW1
#undef TRI_INV_I16_NZ_ROW2
#undef TRI_INV_I16_NZ_ROW3
#undef TRI_INV_I16_NZ_ROW4
#undef TRI_INV_I16_NZ_ROW5
#undef TRI_INV_I16_NZ_ROW6
#undef TRI_INV_I16_NZ_ROW7
#undef TRI_INV_I16_NZ_ROW8
#undef TRI_INV_I16_NZ_ROW9
#undef TRI_INV_I16_NZ_ROW10
#undef TRI_INV_I16_NZ_ROW11
#undef TRI_INV_I16_NZ_ROW12
#undef TRI_INV_I16_NZ_ROW13
#undef TRI_INV_I16_NZ_ROW14
#undef TRI_INV_I16_NZ_ROW15
#undef TRI_INV_I16_NZ_FRACTAL

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
        ccOutHalfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(ccOut), tileNum_ * ccOffset_);

        coreIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreIdx_ /= TASK_RATIO;
            subBlockIdx_ = GetSubBlockIdx();
            subOffset_ = subBlockIdx_ * halfChunkSize_;
            InitLocalBuffers();
            InitGatherBuffer();
        }
        if ASCEND_IS_AIC {
            identity16Prepared_ = false;
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

        inverseBuffer_ = tmpBuff_.GetWithOffset<float>(INVERSE_BUFFER_ELEMS, buffOffset);
        buffOffset += INVERSE_BUFFER_ELEMS * sizeof(float);

        inverseGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(INVERSE_SHAPE, buffOffset);
        buffOffset += INVERSE_SHAPE * sizeof(uint32_t);

        inverseRes_ = tmpBuff_.GetWithOffset<float>(chunkSize_ * halfChunkSize_, buffOffset);
        buffOffset += chunkSize_ * halfChunkSize_ * sizeof(float);
        buffOffset = AlignUp(buffOffset, BLOCK_SIZE);

        aivHalfPower_ = tmpBuff_.GetWithOffset<half>(chunkSize_ * halfChunkSize_, buffOffset);
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
            InitRawMmadLocalBuffers();
            for (uint32_t i = 0; i < curParaNum; ++i) {
                MixInverseCubeDiag16((taskId + i) * ccOffset_);
            }
            RestoreRawMmadState();
            pipe_->Reset();
        }
    }

    __aicore__ inline void MixInverseAIVTile(uint32_t tileId)
    {
        uint64_t tileOffset = static_cast<uint64_t>(tileId) * ccOffset_;
        uint64_t halfOffset = subOffset_ * chunkSize_;
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;

        DataCopyInBf16(curVecLen, ccInGm_[tileOffset + halfOffset]);
        CastHalfPowerFromInput(curVecLen);
        if (subBlockIdx_ == 0) {
            SyncMte3ToV();
            WriteCompose16LowerScratch(tileOffset, 0);
            PrepareDiag16PowerScratch(tileOffset);
            WriteDiag16IdentityScratch(tileOffset, 0, 0);
            WriteDiag16IdentityScratch(tileOffset, DIAG16_SHAPE, DIAG16_SHAPE);
            ReloadTopDiag16PowerScratch(tileOffset);
        } else {
            SyncMte3ToV();
            WriteCompose16LowerScratch(tileOffset, INVERSE_SHAPE);
            WriteCompose32LowerScratch(tileOffset);
            WriteDiag16IdentityScratch(tileOffset, 0, INVERSE_SHAPE);
            WriteDiag16IdentityScratch(tileOffset, DIAG16_SHAPE, INVERSE_SHAPE + DIAG16_SHAPE);
        }
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

    __aicore__ inline uint64_t DiagOffset(uint64_t tileOffset, uint32_t base)
    {
        return tileOffset + static_cast<uint64_t>(base) * chunkSize_ + base;
    }

    __aicore__ inline uint64_t PowerScratchOffset(uint64_t tileOffset, uint32_t base)
    {
        uint32_t blockIdx = base / DIAG16_SHAPE;
        uint32_t scratchRow = (blockIdx / 2) * DIAG16_SHAPE;
        uint32_t scratchCol = INVERSE_SHAPE + (blockIdx % 2) * DIAG16_SHAPE;
        return tileOffset + static_cast<uint64_t>(scratchRow) * chunkSize_ + scratchCol;
    }

    __aicore__ inline uint64_t Compose16LowerOffset(uint64_t tileOffset, uint32_t topBase)
    {
        return tileOffset + static_cast<uint64_t>(topBase + DIAG16_SHAPE) * chunkSize_ + topBase;
    }

    __aicore__ inline uint64_t Compose32LowerOffset(uint64_t tileOffset)
    {
        return tileOffset + static_cast<uint64_t>(INVERSE_SHAPE) * chunkSize_;
    }

    __aicore__ inline uint64_t Compose32LowerSubOffset(uint64_t tileOffset, uint32_t rowBase, uint32_t colBase)
    {
        return Compose32LowerOffset(tileOffset) + static_cast<uint64_t>(rowBase) * chunkSize_ + colBase;
    }

    __aicore__ inline void DataCopyHalfBlockToGm(uint64_t dstOffset, LocalTensor<half> src, uint32_t srcOffset,
                                                 uint32_t blockSize)
    {
        DataCopyExtParams params{1,
                                 static_cast<uint32_t>(blockSize * sizeof(half)),
                                 0, 0, 0};
        for (uint32_t row = 0; row < blockSize; ++row) {
            DataCopyPad(ccOutHalfGm_[dstOffset + static_cast<uint64_t>(row) * chunkSize_],
                        src[srcOffset + row * chunkSize_], params);
        }
    }

    __aicore__ inline void DataCopyHalfBlockToGm(uint64_t dstOffset, LocalTensor<half> src, uint32_t srcOffset)
    {
        DataCopyHalfBlockToGm(dstOffset, src, srcOffset, DIAG16_SHAPE);
    }

    __aicore__ inline void DataCopyDiag16PowerScratch(uint64_t tileOffset, uint32_t localRowBase,
                                                      uint32_t globalBase)
    {
        DataCopyHalfBlockToGm(PowerScratchOffset(tileOffset, globalBase), aivHalfPower_,
                              localRowBase * chunkSize_ + globalBase);
    }

    __aicore__ inline void WriteCompose16LowerScratch(uint64_t tileOffset, uint32_t topBase)
    {
        SyncVToMte3();
        DataCopyHalfBlockToGm(Compose16LowerOffset(tileOffset, topBase), aivHalfPower_,
                              DIAG16_SHAPE * chunkSize_ + topBase);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void WriteCompose32LowerScratch(uint64_t tileOffset)
    {
        SyncVToMte3();
        DataCopyHalfBlockToGm(Compose32LowerOffset(tileOffset), aivHalfPower_, 0, COMPOSE32_SHAPE);
        SyncMte3ToMte2();
        SyncMte3ToS();
        FlushHalfMatrixCacheLines(ccOutHalfGm_[Compose32LowerOffset(tileOffset)],
                                  COMPOSE32_SHAPE, COMPOSE32_SHAPE, chunkSize_);
        SyncMte3ToV();
    }

    __aicore__ inline void WriteDiag16IdentityScratch(uint64_t tileOffset, uint32_t localRowBase,
                                                      uint32_t globalBase)
    {
        uint32_t localOffset = localRowBase * chunkSize_ + globalBase;
        for (uint32_t row = 0; row < DIAG16_SHAPE; ++row) {
            for (uint32_t col = 0; col < DIAG16_SHAPE; ++col) {
                half value = (row == col) ? half(1.0f) : half(0.0f);
                aivHalfPower_.SetValue(localOffset + row * chunkSize_ + col, value);
            }
        }
        SyncVToMte3();
        DataCopyHalfBlockToGm(DiagOffset(tileOffset, globalBase), aivHalfPower_, localOffset);
        SyncMte3ToV();
    }

    __aicore__ inline void CastHalfPowerFromInput(uint64_t len)
    {
        auto bf16Local = inQueue_.DeQue<bfloat16_t>();
        Cast(attnUbFloat_, bf16Local, RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Local);

        Muls(inverseRes_, attnUbFloat_, static_cast<float>(-1.0), len);
        PipeBarrier<PIPE_V>();
        Cast(aivHalfPower_, inverseRes_, RoundMode::CAST_ROUND, len);
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
    }

    __aicore__ inline void PrepareDiag16PowerScratch(uint64_t tileOffset)
    {
        DataCopyDiag16PowerScratch(tileOffset, 0, 0);
        DataCopyDiag16PowerScratch(tileOffset, DIAG16_SHAPE, DIAG16_SHAPE);
        SyncMte3ToV();

        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        DataCopyInBf16(curVecLen, ccInGm_[tileOffset + static_cast<uint64_t>(INVERSE_SHAPE) * chunkSize_]);
        CastHalfPowerFromInput(curVecLen);
        DataCopyDiag16PowerScratch(tileOffset, 0, INVERSE_SHAPE);
        DataCopyDiag16PowerScratch(tileOffset, DIAG16_SHAPE, INVERSE_SHAPE + DIAG16_SHAPE);

        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline void ReloadTopDiag16PowerScratch(uint64_t tileOffset)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        DataCopyInBf16(curVecLen, ccInGm_[tileOffset]);
        CastHalfPowerFromInput(curVecLen);
        DataCopyDiag16PowerScratch(tileOffset, 0, 0);
        DataCopyDiag16PowerScratch(tileOffset, DIAG16_SHAPE, DIAG16_SHAPE);
        SyncMte3ToMte2();
        SyncMte3ToV();
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline uint64_t CompactMatrixNzOffset(uint32_t row, uint32_t col, uint32_t matrixSize)
    {
        return static_cast<uint64_t>(col / BLOCK_CUBE) * matrixSize * BLOCK_CUBE +
               static_cast<uint64_t>(row) * BLOCK_CUBE + (col % BLOCK_CUBE);
    }

    __aicore__ inline uint32_t Compact16Slot(uint32_t slot)
    {
        return slot * DIAG16_ELEMS;
    }

    __aicore__ inline uint32_t Compact32BlockOffset(uint32_t rowBase, uint32_t colBase)
    {
        return static_cast<uint32_t>(CompactMatrixNzOffset(rowBase, colBase, COMPOSE32_SHAPE));
    }

    __aicore__ inline void InitRawMmadLocalBuffers()
    {
        pipe_->Reset();
        pipe_->InitBuffer(aicHalfL1Buff_, AIC_MIX_L1_BYTES);
        pipe_->InitBuffer(aicL0ABuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0BBuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0CBuff_, AIC_RAW_L0_C_BYTES);

        uint32_t l1Offset = 0;
        aicHalfL1Power_ = aicHalfL1Buff_.GetWithOffset<half>(DIAG16_ELEMS, l1Offset);
        l1Offset += DIAG16_ELEMS * sizeof(half);
        aicHalfL1Inv_ = aicHalfL1Buff_.GetWithOffset<half>(DIAG16_ELEMS, l1Offset);
        l1Offset += DIAG16_ELEMS * sizeof(half);
        aicHalfL1Scratch_ = aicHalfL1Buff_.GetWithOffset<half>(COMPOSE32_ELEMS, l1Offset);
        l1Offset += COMPOSE32_ELEMS * sizeof(half);
        aicHalfL1Identity_ = aicHalfL1Buff_.GetWithOffset<half>(DIAG16_ELEMS, l1Offset);
        l1Offset += DIAG16_ELEMS * sizeof(half);
        aicHalfL1Zero_ = aicHalfL1Buff_.GetWithOffset<half>(DIAG16_ELEMS, l1Offset);
        l1Offset += DIAG16_ELEMS * sizeof(half);
        aicHalfL1Top32_ = aicHalfL1Buff_.GetWithOffset<half>(COMPOSE32_ELEMS, l1Offset);
        l1Offset += COMPOSE32_ELEMS * sizeof(half);
        aicHalfL1Bottom32_ = aicHalfL1Buff_.GetWithOffset<half>(COMPOSE32_ELEMS, l1Offset);
        l1Offset += COMPOSE32_ELEMS * sizeof(half);
        aicHalfL1Lower32_ = aicHalfL1Buff_.GetWithOffset<half>(COMPOSE32_ELEMS, l1Offset);
        l1Offset += COMPOSE32_ELEMS * sizeof(half);
        aicHalfL1Identity32_ = aicHalfL1Buff_.GetWithOffset<half>(COMPOSE32_ELEMS, l1Offset);

        aicHalfL0A_ = aicL0ABuff_.GetWithOffset<half>(
            AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half), 0);
        aicHalfL0B_ = aicL0BBuff_.GetWithOffset<half>(
            AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half), 0);
        aicHalfL0AAlt_ = aicL0ABuff_.GetWithOffset<half>(
            AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half), AIC_RAW_L0_AB_HALF_PINGPONG_BYTES);
        aicHalfL0BAlt_ = aicL0BBuff_.GetWithOffset<half>(
            AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half), AIC_RAW_L0_AB_HALF_PINGPONG_BYTES);
        aicL0C_ = aicL0CBuff_.Get<float>();
        identity16Prepared_ = false;
        PrepareRawMmadState();
    }

    __aicore__ inline void PrepareRawMmadState()
    {
        SetLoadDataPaddingValue<uint64_t>(0);
        SetNdParaImpl(0x1);
        SetHF32Mode(HF32Mode::DISABLE);
        SetHF32TransMode(HF32TransMode::NEAREST_EVEN);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void RestoreRawMmadState()
    {
        SetAtomicNone();
        SetNdParaImpl(0x1);
        SetHF32Mode(HF32Mode::DISABLE);
        SetHF32TransMode(HF32TransMode::NEAREST_EVEN);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void SyncMte2ToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(eventID);
        WaitFlag<HardEvent::MTE2_MTE1>(eventID);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    __aicore__ inline void SyncMte3ToS()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventID);
        WaitFlag<HardEvent::MTE3_S>(eventID);
    }

    __aicore__ inline void SyncVToMte3()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventID);
        WaitFlag<HardEvent::MTE3_V>(eventID);
    }

    __aicore__ inline void SyncMte2ToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_M));
        SetFlag<HardEvent::MTE2_M>(eventID);
        WaitFlag<HardEvent::MTE2_M>(eventID);
    }

    __aicore__ inline void SyncMte1ToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(eventID);
        WaitFlag<HardEvent::MTE1_M>(eventID);
    }

    __aicore__ inline void SyncMToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::M_MTE1));
        SetFlag<HardEvent::M_MTE1>(eventID);
        WaitFlag<HardEvent::M_MTE1>(eventID);
    }

    __aicore__ inline void SyncMToFix()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(eventID);
        WaitFlag<HardEvent::M_FIX>(eventID);
    }

    __aicore__ inline void SyncFixToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_M));
        SetFlag<HardEvent::FIX_M>(eventID);
        WaitFlag<HardEvent::FIX_M>(eventID);
    }

    __aicore__ inline void SyncFixToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE1));
        SetFlag<HardEvent::FIX_MTE1>(eventID);
        WaitFlag<HardEvent::FIX_MTE1>(eventID);
    }

    __aicore__ inline void SyncFixToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    __aicore__ inline void SyncFixToS()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_S));
        SetFlag<HardEvent::FIX_S>(eventID);
        WaitFlag<HardEvent::FIX_S>(eventID);
    }

    __aicore__ inline void FlushBf16MatrixCacheLines(GlobalTensor<bfloat16_t> gm, uint32_t rows, uint32_t cols,
                                                     uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; col += BF16_CACHELINE_ELEMS) {
                Barrier();
                DataCacheCleanAndInvalid<bfloat16_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    gm[row * stride + col]);
            }
        }
        Barrier();
    }

    __aicore__ inline void FlushHalfMatrixCacheLines(GlobalTensor<half> gm, uint32_t rows, uint32_t cols,
                                                     uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; col += HALF_CACHELINE_ELEMS) {
                Barrier();
                DataCacheCleanAndInvalid<half, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    gm[row * stride + col]);
            }
        }
        Barrier();
    }

    __aicore__ inline void CleanHalfMatrixCacheLines(GlobalTensor<half> gm, uint32_t rows, uint32_t cols,
                                                     uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; col += HALF_CACHELINE_ELEMS) {
                Barrier();
                DataCacheCleanAndInvalid<half, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_UB>(
                    gm[row * stride + col]);
            }
        }
        Barrier();
    }

    __aicore__ inline void PrepareIdentity16(uint64_t tileOffset, uint32_t base)
    {
        LoadHalfBlockFromGm(DiagOffset(tileOffset, base), aicHalfL1Identity_);
    }

    __aicore__ inline void ZeroHalfL1(LocalTensor<half> l1)
    {
        ZeroHalfL1(l1, DIAG16_ELEMS);
    }

    __aicore__ inline void ZeroHalfL1(LocalTensor<half> l1, uint32_t elems)
    {
        SyncMte2ToM();
        SyncMte1ToM();
        SyncFixToM();
        InitConstValueParams<half> zeroParams(
            1, static_cast<uint16_t>(AlignUp(elems * sizeof(half), BLOCK_SIZE) / BLOCK_SIZE),
            0, half(0.0f));
        InitConstValue(l1, zeroParams);
        SyncMToMte1();
    }

    __aicore__ inline void LoadHalfBlockFromGm(uint64_t offset, LocalTensor<half> dst)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = DIAG16_SHAPE;
        copyParams.dValue = DIAG16_SHAPE;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = chunkSize_;
        copyParams.dstNzC0Stride = DIAG16_SHAPE;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(dst, ccOutHalfGm_[offset], copyParams);
        SyncMte2ToMte1();
    }

    __aicore__ inline void LoadDiag16PowerScratch(uint64_t tileOffset, uint32_t base)
    {
        LoadHalfBlockFromGm(PowerScratchOffset(tileOffset, base), aicHalfL1Power_);
    }

    __aicore__ inline void LoadDiag32PowerScratch(uint64_t tileOffset, uint32_t base,
                                                  LocalTensor<half> dstPower)
    {
        ZeroHalfL1(dstPower, COMPOSE32_ELEMS);
        LoadHalfBlockFromGm(PowerScratchOffset(tileOffset, base),
                            dstPower[Compact32BlockOffset(0, 0)]);
        LoadHalfBlockFromGm(Compose16LowerOffset(tileOffset, base),
                            dstPower[Compact32BlockOffset(DIAG16_SHAPE, 0)]);
        LoadHalfBlockFromGm(PowerScratchOffset(tileOffset, base + DIAG16_SHAPE),
                            dstPower[Compact32BlockOffset(DIAG16_SHAPE, DIAG16_SHAPE)]);
    }

    __aicore__ inline void LoadIdentity32Scratch(uint64_t tileOffset, uint32_t base,
                                                 LocalTensor<half> dstIdentity)
    {
        ZeroHalfL1(dstIdentity, COMPOSE32_ELEMS);
        LoadHalfBlockFromGm(DiagOffset(tileOffset, base),
                            dstIdentity[Compact32BlockOffset(0, 0)]);
        LoadHalfBlockFromGm(DiagOffset(tileOffset, base + DIAG16_SHAPE),
                            dstIdentity[Compact32BlockOffset(DIAG16_SHAPE, DIAG16_SHAPE)]);
    }

    __aicore__ inline void LoadHalfBlockFromGm(uint64_t offset, LocalTensor<half> dst, uint32_t blockSize)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = blockSize;
        copyParams.dValue = blockSize;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = chunkSize_;
        copyParams.dstNzC0Stride = AlignUp(blockSize, BLOCK_CUBE);
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(dst, ccOutHalfGm_[offset], copyParams);
        SyncMte2ToMte1();
    }

    __aicore__ inline void CopyHalfL1MatrixToL0A(LocalTensor<half> l1, LocalTensor<half> l0, uint32_t blockSize)
    {
        uint32_t blockCount = AlignUp(blockSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = static_cast<uint16_t>(blockCount);
        loadParams.srcStride = static_cast<uint16_t>(blockCount);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = false;
        loadParams.addrMode = 0;
        for (uint32_t rowBlock = 0; rowBlock < blockCount; ++rowBlock) {
            LoadData(l0[rowBlock * blockCount * RAW_HALF_FRACTAL_ELEMS],
                     l1[rowBlock * RAW_HALF_FRACTAL_ELEMS], loadParams);
        }
    }

    __aicore__ inline void CopyHalfL1MatrixToL0A(LocalTensor<half> l1, LocalTensor<half> l0)
    {
        CopyHalfL1MatrixToL0A(l1, l0, DIAG16_SHAPE);
    }

    __aicore__ inline void CopyHalfL1MatrixToL0B(LocalTensor<half> l1, LocalTensor<half> l0, uint32_t blockSize)
    {
        uint32_t blockCount = AlignUp(blockSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = static_cast<uint16_t>(blockCount);
        loadParams.srcStride = static_cast<uint16_t>(blockCount);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = true;
        loadParams.addrMode = 0;
        for (uint32_t colBlock = 0; colBlock < blockCount; ++colBlock) {
            LoadData(l0[colBlock * blockCount * RAW_HALF_FRACTAL_ELEMS],
                     l1[colBlock * RAW_HALF_FRACTAL_ELEMS], loadParams);
        }
    }

    __aicore__ inline void CopyHalfL1MatrixToL0B(LocalTensor<half> l1, LocalTensor<half> l0)
    {
        CopyHalfL1MatrixToL0B(l1, l0, DIAG16_SHAPE);
    }

    __aicore__ inline void CopyPackedHalfL1BlockToL0(LocalTensor<half> l1, LocalTensor<half> l0,
                                                     uint32_t srcRow, uint32_t srcCol,
                                                     uint32_t blockSize, uint32_t packedSize,
                                                     bool transpose)
    {
        uint32_t blockCount = AlignUp(blockSize, BLOCK_CUBE) / BLOCK_CUBE;
        uint32_t srcBlockStride = AlignUp(packedSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = static_cast<uint16_t>(blockCount);
        loadParams.srcStride = static_cast<uint16_t>(srcBlockStride);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;
        for (uint32_t rowBlock = 0; rowBlock < blockCount; ++rowBlock) {
            uint32_t row = rowBlock * BLOCK_CUBE;
            uint32_t srcOffset = static_cast<uint32_t>(CompactMatrixNzOffset(srcRow + row, srcCol, packedSize));
            uint32_t dstOffset = rowBlock * blockCount * RAW_HALF_FRACTAL_ELEMS;
            LoadData(l0[dstOffset], l1[srcOffset], loadParams);
        }
    }

    __aicore__ inline MmadParams MakeRawMmadParams(bool initCMatrix, uint8_t unitFlag)
    {
        return MakeRawMmadParams(DIAG16_SHAPE, DIAG16_SHAPE, DIAG16_SHAPE, initCMatrix, unitFlag);
    }

    __aicore__ inline MmadParams MakeRawMmadParams(uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                                   bool initCMatrix, uint8_t unitFlag)
    {
        MmadParams params(static_cast<uint16_t>(mSize), static_cast<uint16_t>(nSize),
                          static_cast<uint16_t>(kSize), unitFlag, RAW_MMAD_CMATRIX_SOURCE_L0C, initCMatrix);
        params.kDirectionAlign = false;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
        params.disableGemv = true;
#endif
        return params;
    }

    __aicore__ inline void RawMmadHalfL1AccumulateTile(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                       LocalTensor<float> cTile, bool initCMatrix,
                                                       uint8_t unitFlag, uint32_t l0BufferIdx = 0)
    {
        LocalTensor<half> l0A = (l0BufferIdx == 0) ? aicHalfL0A_ : aicHalfL0AAlt_;
        LocalTensor<half> l0B = (l0BufferIdx == 0) ? aicHalfL0B_ : aicHalfL0BAlt_;
        SyncMToMte1();
        CopyHalfL1MatrixToL0A(lhsL1, l0A);
        CopyHalfL1MatrixToL0B(rhsL1, l0B);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, l0A, l0B, MakeRawMmadParams(initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void RawMmadHalfL1AccumulateTile(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                       LocalTensor<float> cTile, uint32_t blockSize,
                                                       bool initCMatrix, uint8_t unitFlag,
                                                       uint32_t l0BufferIdx = 0)
    {
        LocalTensor<half> l0A = (l0BufferIdx == 0) ? aicHalfL0A_ : aicHalfL0AAlt_;
        LocalTensor<half> l0B = (l0BufferIdx == 0) ? aicHalfL0B_ : aicHalfL0BAlt_;
        SyncMToMte1();
        CopyHalfL1MatrixToL0A(lhsL1, l0A, blockSize);
        CopyHalfL1MatrixToL0B(rhsL1, l0B, blockSize);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, l0A, l0B, MakeRawMmadParams(blockSize, blockSize, blockSize, initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void RawMmadPackedHalfL1AccumulateTile(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                             LocalTensor<float> cTile,
                                                             uint32_t lhsRow, uint32_t lhsCol,
                                                             uint32_t rhsRow, uint32_t rhsCol,
                                                             uint32_t blockSize,
                                                             uint32_t lhsPackedSize,
                                                             uint32_t rhsPackedSize,
                                                             bool initCMatrix, uint8_t unitFlag,
                                                             uint32_t l0BufferIdx = 0)
    {
        LocalTensor<half> l0A = (l0BufferIdx == 0) ? aicHalfL0A_ : aicHalfL0AAlt_;
        LocalTensor<half> l0B = (l0BufferIdx == 0) ? aicHalfL0B_ : aicHalfL0BAlt_;
        SyncMToMte1();
        CopyPackedHalfL1BlockToL0(lhsL1, l0A, lhsRow, lhsCol, blockSize, lhsPackedSize, false);
        CopyPackedHalfL1BlockToL0(rhsL1, l0B, rhsRow, rhsCol, blockSize, rhsPackedSize, true);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, l0A, l0B, MakeRawMmadParams(blockSize, blockSize, blockSize, initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void CopyL0CMatrixToHalfL1(LocalTensor<half> l1, LocalTensor<float> l0c,
                                                 uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = DIAG16_SHAPE;
        fixParams.mSize = DIAG16_SHAPE;
        fixParams.srcStride = DIAG16_SHAPE;
        fixParams.dstStride = DIAG16_SHAPE * DIAG16_SHAPE * sizeof(half) / BLOCK_SIZE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        Fixpipe<half, float, CFG_NZ>(l1, l0c, fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToHalfL1(LocalTensor<half> l1, LocalTensor<float> l0c,
                                                 uint32_t blockSize, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = blockSize;
        fixParams.mSize = blockSize;
        fixParams.srcStride = AlignUp(blockSize, BLOCK_CUBE);
        fixParams.dstStride = AlignUp(blockSize, BLOCK_CUBE) * BLOCK_CUBE * sizeof(half) / BLOCK_SIZE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        Fixpipe<half, float, CFG_NZ>(l1, l0c, fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToGmBf16(GlobalTensor<bfloat16_t> gm, LocalTensor<float> l0c,
                                                 uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.mSize = DIAG16_SHAPE;
        fixParams.nSize = DIAG16_SHAPE;
        fixParams.srcStride = DIAG16_SHAPE;
        fixParams.dstStride = chunkSize_;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322BF16;
        fixParams.unitFlag = unitFlag;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(gm, l0c, fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToGmBf16(GlobalTensor<bfloat16_t> gm, LocalTensor<float> l0c,
                                                 uint32_t blockSize, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.mSize = blockSize;
        fixParams.nSize = blockSize;
        fixParams.srcStride = AlignUp(blockSize, BLOCK_CUBE);
        fixParams.dstStride = chunkSize_;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322BF16;
        fixParams.unitFlag = unitFlag;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(gm, l0c, fixParams);
    }

    __aicore__ inline void HalfL1PowerBlock(LocalTensor<half> lhs, LocalTensor<half> rhs,
                                            LocalTensor<half> dst, LocalTensor<float> cTile)
    {
        RawMmadHalfL1AccumulateTile(lhs, rhs, cTile, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(dst, cTile, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void HalfL1PowerBlock32(LocalTensor<half> lhs, LocalTensor<half> rhs,
                                              LocalTensor<half> dst, LocalTensor<float> cTile)
    {
        RawMmadHalfL1AccumulateTile(lhs, rhs, cTile, COMPOSE32_SHAPE,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(dst, cTile, COMPOSE32_SHAPE, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void HalfL1InvAddBlock(LocalTensor<half> inv, LocalTensor<half> power,
                                             LocalTensor<float> cTile)
    {
        RawMmadHalfL1AccumulateTile(inv, aicHalfL1Identity_, cTile,
                                    RAW_MMAD_INIT_C, RAW_MMAD_NON_FINAL_UNIT_FLAG, 0);
        RawMmadHalfL1AccumulateTile(inv, power, cTile,
                                    RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG, 1);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(inv, cTile, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void BuildInvInitFromPower(LocalTensor<float> cTile)
    {
        RawMmadHalfL1AccumulateTile(aicHalfL1Identity_, aicHalfL1Identity_, cTile,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(aicHalfL1Inv_, cTile, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
        HalfL1InvAddBlock(aicHalfL1Inv_, aicHalfL1Power_, cTile);
    }

    __aicore__ inline void HalfL1InvAddBlock32(LocalTensor<half> inv, LocalTensor<half> power,
                                               LocalTensor<half> identity, LocalTensor<float> cTile)
    {
        RawMmadHalfL1AccumulateTile(inv, identity, cTile, COMPOSE32_SHAPE,
                                    RAW_MMAD_INIT_C, RAW_MMAD_NON_FINAL_UNIT_FLAG, 0);
        RawMmadHalfL1AccumulateTile(inv, power, cTile, COMPOSE32_SHAPE,
                                    RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG, 1);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(inv, cTile, COMPOSE32_SHAPE, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void BuildInv32InitFromPower(LocalTensor<half> inv, LocalTensor<half> power,
                                                   LocalTensor<half> identity, LocalTensor<float> cTile)
    {
        RawMmadHalfL1AccumulateTile(identity, identity, cTile, COMPOSE32_SHAPE,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(inv, cTile, COMPOSE32_SHAPE, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
        HalfL1InvAddBlock32(inv, power, identity, cTile);
    }

    __aicore__ inline void SolveDiag16ToGm(uint64_t tileOffset, uint32_t base)
    {
        PrepareIdentity16(tileOffset, base);
        ZeroHalfL1(aicHalfL1Scratch_);
        LoadDiag16PowerScratch(tileOffset, base);

        LocalTensor<float> cInv = aicL0C_;
        LocalTensor<float> cPower = aicL0C_[static_cast<uint32_t>(chunkSize_ * chunkSize_)];
        BuildInvInitFromPower(cInv);

        HalfL1PowerBlock(aicHalfL1Power_, aicHalfL1Power_, aicHalfL1Scratch_, cPower);
        HalfL1InvAddBlock(aicHalfL1Inv_, aicHalfL1Scratch_, cInv);

        HalfL1PowerBlock(aicHalfL1Scratch_, aicHalfL1Scratch_, aicHalfL1Power_, cPower);
        HalfL1InvAddBlock(aicHalfL1Inv_, aicHalfL1Power_, cInv);

        HalfL1PowerBlock(aicHalfL1Power_, aicHalfL1Power_, aicHalfL1Scratch_, cPower);
        RawMmadHalfL1AccumulateTile(aicHalfL1Inv_, aicHalfL1Identity_, cInv,
                                    RAW_MMAD_INIT_C, RAW_MMAD_NON_FINAL_UNIT_FLAG, 0);
        RawMmadHalfL1AccumulateTile(aicHalfL1Inv_, aicHalfL1Scratch_, cInv,
                                    RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG, 1);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(aicHalfL1Inv_, cInv, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
        RawMmadHalfL1AccumulateTile(aicHalfL1Inv_, aicHalfL1Identity_, cInv,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, 0);
        SyncMToFix();
        CopyL0CMatrixToGmBf16(ccOutGm_[tileOffset + static_cast<uint64_t>(base) * chunkSize_ + base],
                              cInv, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
    }

    __aicore__ inline void SolveDiag32ToGmAndL1(uint64_t tileOffset, uint32_t base,
                                                LocalTensor<half> inv32, LocalTensor<half> power32,
                                                LocalTensor<half> scratch32, LocalTensor<half> identity32,
                                                LocalTensor<float> cPower, LocalTensor<float> cInv)
    {
        LoadIdentity32Scratch(tileOffset, base, identity32);
        LoadDiag32PowerScratch(tileOffset, base, power32);
        BuildInv32InitFromPower(inv32, power32, identity32, cInv);

        HalfL1PowerBlock32(power32, power32, scratch32, cPower);
        HalfL1InvAddBlock32(inv32, scratch32, identity32, cInv);

        HalfL1PowerBlock32(scratch32, scratch32, power32, cPower);
        HalfL1InvAddBlock32(inv32, power32, identity32, cInv);

        HalfL1PowerBlock32(power32, power32, scratch32, cPower);
        HalfL1InvAddBlock32(inv32, scratch32, identity32, cInv);

        HalfL1PowerBlock32(scratch32, scratch32, power32, cPower);
        HalfL1InvAddBlock32(inv32, power32, identity32, cInv);

        RawMmadHalfL1AccumulateTile(inv32, identity32, cInv, COMPOSE32_SHAPE,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, 0);
        SyncMToFix();
        CopyL0CMatrixToGmBf16(ccOutGm_[tileOffset + static_cast<uint64_t>(base) * chunkSize_ + base],
                              cInv, COMPOSE32_SHAPE, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
    }

    __aicore__ inline void SaveCurrentInv16ToL1(LocalTensor<half> dst, LocalTensor<float> cTile)
    {
        HalfL1PowerBlock(aicHalfL1Inv_, aicHalfL1Identity_, dst, cTile);
    }

    __aicore__ inline void CopyHalfL1Compact16ToBlock32(LocalTensor<half> src16, LocalTensor<half> dst32,
                                                        uint32_t rowBase, uint32_t colBase)
    {
        SyncMte2ToM();
        SyncFixToM();
        SyncMte1ToM();
        for (uint32_t row = 0; row < DIAG16_SHAPE; ++row) {
            for (uint32_t col = 0; col < DIAG16_SHAPE; ++col) {
                dst32.SetValue(CompactMatrixNzOffset(rowBase + row, colBase + col, COMPOSE32_SHAPE),
                               src16.GetValue(CompactMatrixNzOffset(row, col, DIAG16_SHAPE)));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void CopyL0CMatrixToHalfL1Block32(LocalTensor<half> dst32, LocalTensor<float> l0c,
                                                        uint32_t rowBase, uint32_t colBase)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = DIAG16_SHAPE;
        fixParams.mSize = DIAG16_SHAPE;
        fixParams.srcStride = DIAG16_SHAPE;
        fixParams.dstStride = COMPOSE32_SHAPE * DIAG16_SHAPE * sizeof(half) / BLOCK_SIZE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = RAW_MMAD_FINAL_UNIT_FLAG;
        fixParams.isChannelSplit = false;
        Fixpipe<half, float, CFG_NZ>(dst32[CompactMatrixNzOffset(rowBase, colBase, COMPOSE32_SHAPE)],
                                     l0c, fixParams);
    }

    __aicore__ inline void StoreL0C16ToPacked32(LocalTensor<half> dst32, LocalTensor<float> l0c,
                                                uint32_t rowBase, uint32_t colBase)
    {
        SyncMToFix();
        CopyL0CMatrixToHalfL1(aicHalfL1Identity_, l0c, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
        CopyHalfL1Compact16ToBlock32(aicHalfL1Identity_, dst32, rowBase, colBase);
    }

    __aicore__ inline void Compose16To32Lower(uint64_t tileOffset, uint32_t topBase,
                                              LocalTensor<half> topInv, LocalTensor<half> lowerDst,
                                              LocalTensor<float> cScratch, LocalTensor<float> cOut)
    {
        LoadHalfBlockFromGm(Compose16LowerOffset(tileOffset, topBase), aicHalfL1Power_);
        HalfL1PowerBlock(aicHalfL1Power_, topInv, aicHalfL1Scratch_, cScratch);
        RawMmadHalfL1AccumulateTile(aicHalfL1Inv_, aicHalfL1Scratch_, cOut,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToGmBf16(ccOutGm_[Compose16LowerOffset(tileOffset, topBase)],
                              cOut, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        RawMmadHalfL1AccumulateTile(aicHalfL1Inv_, aicHalfL1Scratch_, cOut,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(lowerDst, cOut, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void Compose32To64Lower(uint64_t tileOffset, LocalTensor<float> cScratch,
                                              LocalTensor<float> cOut)
    {
        RawMmadHalfL1AccumulateTile(aicHalfL1Lower32_, aicHalfL1Top32_, cScratch, COMPOSE32_SHAPE,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, 0);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(aicHalfL1Scratch_, cScratch, COMPOSE32_SHAPE, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();

        RawMmadHalfL1AccumulateTile(aicHalfL1Bottom32_, aicHalfL1Scratch_, cOut, COMPOSE32_SHAPE,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, 0);
        SyncMToFix();
        CopyL0CMatrixToGmBf16(ccOutGm_[Compose32LowerOffset(tileOffset)], cOut,
                              COMPOSE32_SHAPE, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
    }

    __aicore__ inline void ZeroBf16Block(uint64_t offset)
    {
        RawMmadHalfL1AccumulateTile(aicHalfL1Zero_, aicHalfL1Zero_, aicL0C_,
                                    RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToGmBf16(ccOutGm_[offset], aicL0C_, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
    }

    __aicore__ inline void LoadCompose32LowerToL1(uint64_t tileOffset)
    {
        CleanHalfMatrixCacheLines(ccOutHalfGm_[Compose32LowerSubOffset(tileOffset, 0, 0)],
                                  DIAG16_SHAPE, DIAG16_SHAPE, chunkSize_);
        LoadHalfBlockFromGm(Compose32LowerSubOffset(tileOffset, 0, 0),
                            aicHalfL1Lower32_[Compact32BlockOffset(0, 0)]);
        CleanHalfMatrixCacheLines(ccOutHalfGm_[Compose32LowerSubOffset(tileOffset, 0, DIAG16_SHAPE)],
                                  DIAG16_SHAPE, DIAG16_SHAPE, chunkSize_);
        LoadHalfBlockFromGm(Compose32LowerSubOffset(tileOffset, 0, DIAG16_SHAPE),
                            aicHalfL1Lower32_[Compact32BlockOffset(0, DIAG16_SHAPE)]);
        CleanHalfMatrixCacheLines(ccOutHalfGm_[Compose32LowerSubOffset(tileOffset, DIAG16_SHAPE, 0)],
                                  DIAG16_SHAPE, DIAG16_SHAPE, chunkSize_);
        LoadHalfBlockFromGm(Compose32LowerSubOffset(tileOffset, DIAG16_SHAPE, 0),
                            aicHalfL1Lower32_[Compact32BlockOffset(DIAG16_SHAPE, 0)]);
        CleanHalfMatrixCacheLines(ccOutHalfGm_[Compose32LowerSubOffset(tileOffset, DIAG16_SHAPE, DIAG16_SHAPE)],
                                  DIAG16_SHAPE, DIAG16_SHAPE, chunkSize_);
        LoadHalfBlockFromGm(Compose32LowerSubOffset(tileOffset, DIAG16_SHAPE, DIAG16_SHAPE),
                            aicHalfL1Lower32_[Compact32BlockOffset(DIAG16_SHAPE, DIAG16_SHAPE)]);
    }

    __aicore__ inline void MixInverseCubeDiag16(uint64_t tileOffset)
    {
        LocalTensor<float> cScratch = aicL0C_[static_cast<uint32_t>(chunkSize_ * chunkSize_)];
        SolveDiag32ToGmAndL1(tileOffset, 0, aicHalfL1Top32_, aicHalfL1Scratch_,
                             aicHalfL1Lower32_, aicHalfL1Identity32_, cScratch, aicL0C_);
        SolveDiag32ToGmAndL1(tileOffset, INVERSE_SHAPE, aicHalfL1Bottom32_, aicHalfL1Scratch_,
                             aicHalfL1Lower32_, aicHalfL1Identity32_, cScratch, aicL0C_);
        LoadCompose32LowerToL1(tileOffset);
        Compose32To64Lower(tileOffset, cScratch, aicL0C_);
        ZeroTopRightPowerScratch(tileOffset);
    }

    __aicore__ inline void ZeroTopRightPowerScratch(uint64_t tileOffset)
    {
        ZeroHalfL1(aicHalfL1Zero_);
        ZeroBf16Block(tileOffset + DIAG16_SHAPE);
        ZeroBf16Block(tileOffset + static_cast<uint64_t>(INVERSE_SHAPE) * chunkSize_ +
                      INVERSE_SHAPE + DIAG16_SHAPE);
        for (uint32_t idx = 0; idx < 4; ++idx) {
            uint32_t scratchRow = (idx / 2) * DIAG16_SHAPE;
            uint32_t scratchCol = INVERSE_SHAPE + (idx % 2) * DIAG16_SHAPE;
            ZeroBf16Block(tileOffset + static_cast<uint64_t>(scratchRow) * chunkSize_ + scratchCol);
        }
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
    GlobalTensor<half> ccOutHalfGm_;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    TBuf<TPosition::TSCM> aicHalfL1Buff_;
    TBuf<TPosition::A2> aicL0ABuff_;
    TBuf<TPosition::B2> aicL0BBuff_;
    TBuf<TPosition::CO1> aicL0CBuff_;

    LocalTensor<float> attnUbFloat_;
    LocalTensor<float> inverseBuffer_;
    LocalTensor<float> inverseRes_;
    LocalTensor<uint32_t> inverseGatherBuffer_;
    LocalTensor<half> aivHalfPower_;
    LocalTensor<half> aicHalfL1Power_;
    LocalTensor<half> aicHalfL1Inv_;
    LocalTensor<half> aicHalfL1Scratch_;
    LocalTensor<half> aicHalfL1Identity_;
    LocalTensor<half> aicHalfL1Zero_;
    LocalTensor<half> aicHalfL1Top32_;
    LocalTensor<half> aicHalfL1Bottom32_;
    LocalTensor<half> aicHalfL1Lower32_;
    LocalTensor<half> aicHalfL1Identity32_;
    LocalTensor<half> aicHalfL0A_;
    LocalTensor<half> aicHalfL0B_;
    LocalTensor<half> aicHalfL0AAlt_;
    LocalTensor<half> aicHalfL0BAlt_;
    LocalTensor<float> aicL0C_;
    bool identity16Prepared_;
};

using TriInverseOp = MixInverseOp;
} // namespace TriInverse

#endif // TRI_INVERSE_MIX_KERNEL_H
