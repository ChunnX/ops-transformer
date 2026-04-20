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
 * \file mhc_post_backward.h
 * \brief
 */

#ifndef MHC_POST_BACKWARD_H
#define MHC_POST_BACKWARD_H
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

namespace MhcPostBackward {
using namespace AscendC;

// Constants for memory alignment and buffer configuration
constexpr uint32_t BF16_FP16_ALIGN_SIZE = 16; // 16 elements = 32 bytes for bf16/fp16
constexpr uint32_t FLOAT32_ALIGN_SIZE = 8; // 8 elements = 32 bytes for float32

// Queue depth configuration
constexpr uint32_t QUEUE_DEPTH = 1; // Single Buffer depth for all queues

// Template macro definitions - 用于编译时优化
// IS_N_ALIGNED: n == 8 (only n=8 is 32-byte aligned for DataCopy)
// IS_NN_ALIGNED: (n*n) % 8 == 0 (16,64: true; 36: false)
// IS_D_ALIGNED: D % 16 == 0 && nTilesD == 1
#define TEMPLATE_DECLARE template<typename T, uint16_t IS_N_ALIGNED, uint16_t IS_NN_ALIGNED, uint16_t IS_D_ALIGNED>
#define TEMPLATE_ARGS T, IS_N_ALIGNED, IS_NN_ALIGNED, IS_D_ALIGNED

TEMPLATE_DECLARE
class MhcPostBackwardKernel {
public:
    __aicore__ inline MhcPostBackwardKernel() {}
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                            GM_ADDR gradX, GM_ADDR gradHRes, GM_ADDR gradHOut, GM_ADDR gradHPost,
                            GM_ADDR workspace,
                            const MhcPostBackwardTilingData *tilingData,
                            TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInWeights(uint32_t globalItemIdx);
    __aicore__ inline void ComputeInitGrads();
    __aicore__ inline void CopyInTile(uint32_t globalItemIdx, uint32_t tileId);
    __aicore__ inline void ComputeTile(uint32_t tileId);
    __aicore__ inline void CopyOutGradHOut(uint32_t globalItemIdx, uint32_t tileId);
    __aicore__ inline void CopyOutGradX(uint32_t globalItemIdx, uint32_t tileId);

    __aicore__ inline void CopyOutGradHPost(uint32_t globalItemIdx, uint32_t tileId);

    __aicore__ inline void CopyOutWeightGrads(uint32_t globalItemIdx);
    __aicore__ inline void SetMMParaAndCompute(uint32_t globalItemIdx);
    template<uint16_t N>
    __aicore__ inline void ComputeGradHPost(LocalTensor<float> &gradHPostLocal, LocalTensor<T> &hOutLocal,
                                            LocalTensor<float> &gradOutLocal, uint32_t dataLen);
    __aicore__ inline void ComputeGradHOutTile(LocalTensor<float> gradOutF32UB, LocalTensor<float> hPostLocalUB, 
                                            LocalTensor<T> gradHOutTileUB, uint32_t actualTileD);
    __aicore__ inline void ComputeGradXTile(LocalTensor<float> gradOutF32UB, 
                                            LocalTensor<float> hResLocalUB, 
                                            LocalTensor<T> gradXTileUB, 
                                            uint32_t actualTileD);

private:
    TPipe *pipe_;
    constexpr static AscendC::MicroAPI::CastTrait castB32ToB16 = { AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    // Input queues
    TQue<QuePosition::VECIN, QUEUE_DEPTH> gradOutTileQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> hOutTileQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> hPostQueue_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> hResQueue_;

    // Output queues
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> gradHOutTileQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> gradXTileQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> gradHPostQueue_;

    // Intermediate buffers for float computation
    TBuf<QuePosition::VECCALC> gradOutF32Buf_;

    // Global memory tensors - inputs (bf16/fp16)
    GlobalTensor<T> gradOutputGm_;
    GlobalTensor<T> hOutGm_;
    GlobalTensor<T> xGm_;
    // Global memory tensors - inputs (float32)
    GlobalTensor<float> hPostGm_;
    GlobalTensor<float> hResGm_;

    // Global memory tensors - outputs (bf16/fp16)
    GlobalTensor<T> gradHOutGm_;
    GlobalTensor<T> gradXGm_;
    // Global memory tensors - outputs (float32)
    GlobalTensor<float> gradHPostGm_;
    GlobalTensor<float> gradHResGm_;

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    matmul::MatmulImpl<aType, bType, cType> mm;

    // Tiling parameters
    uint32_t totalItems_;
    uint32_t itemsPerCore_;
    uint32_t remainderItems_;
    uint32_t usedCores_;
    uint32_t S_;
    uint32_t n_;
    uint32_t D_;
    uint32_t tileD_;
    uint32_t nTilesD_;
    uint32_t alignedD_;
    uint32_t lastTileD_;
    uint32_t alignedN_;      // n aligned to 8 for float32 vector ops
    uint32_t alignedNN_;     // n*n aligned to 8 for float32 vector ops
    uint32_t usedAic_;
    uint32_t itemsPerAic_;
    uint32_t remainderItemsAic_;
    uint32_t singleCoreK_;

    uint32_t blockIdx_;
    uint32_t myItemCount_;
    uint32_t itemStart_;
};

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::Init(
    GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR gradX, GM_ADDR gradHRes, GM_ADDR gradHOut, GM_ADDR gradHPost,
    GM_ADDR workspace,
    const MhcPostBackwardTilingData *tilingData,
    TPipe *tPipe)
{
    pipe_ = tPipe;

    // Get tiling data using direct member access
    totalItems_ = tilingData->totalItems;
    itemsPerCore_ = tilingData->itemsPerCore;
    remainderItems_ = tilingData->remainderItems;
    usedCores_ = tilingData->usedCores;
    S_ = tilingData->S;
    n_ = tilingData->n;
    D_ = tilingData->D;
    tileD_ = tilingData->tileD;
    nTilesD_ = tilingData->nTilesD;
    alignedD_ = tilingData->alignedD;
    lastTileD_ = tilingData->lastTileD;
    usedAic_ = tilingData->matmulTiling.usedCoreNum;
    itemsPerAic_ = tilingData->itemsPerAic;
    remainderItemsAic_ = tilingData->remainderItemsAic;
    singleCoreK_ = tilingData->matmulTiling.singleCoreK;

    alignedN_ = tilingData->alignedN;
    alignedNN_ = tilingData->alignedNN;

    blockIdx_ = GetBlockIdx();

    // Set global memory buffers for inputs (bf16/fp16)
    gradOutputGm_.SetGlobalBuffer((__gm__ T *)gradOutput, totalItems_ * n_ * D_);
    hOutGm_.SetGlobalBuffer((__gm__ T *)hOut, totalItems_ * D_);
    xGm_.SetGlobalBuffer((__gm__ T *)x, totalItems_ * n_ * D_);
    // Set global memory buffers for inputs (float32)
    hPostGm_.SetGlobalBuffer((__gm__ float *)hPost, totalItems_ * n_);
    hResGm_.SetGlobalBuffer((__gm__ float *)hRes, totalItems_ * n_ * n_);

    // Set global memory buffers for outputs (bf16/fp16)
    gradHOutGm_.SetGlobalBuffer((__gm__ T *)gradHOut, totalItems_ * D_);
    gradXGm_.SetGlobalBuffer((__gm__ T *)gradX, totalItems_ * n_ * D_);
    // Set global memory buffers for outputs (float32)
    // gradHPostGm_.SetGlobalBuffer((__gm__ float *)gradHPost, totalItems_ * n_);
    gradHPostGm_.SetGlobalBuffer((__gm__ float *)gradHPost, totalItems_ * n_ * D_);
    gradHResGm_.SetGlobalBuffer((__gm__ float *)gradHRes, totalItems_ * n_ * n_);

    if ASCEND_IS_AIV {
        // Calculate work distribution with remainder handling
        if (blockIdx_ < remainderItems_) {
            myItemCount_ = itemsPerCore_ + 1;
            itemStart_ = blockIdx_ * (itemsPerCore_ + 1);
        } else {
            myItemCount_ = itemsPerCore_;
            itemStart_ = remainderItems_ * (itemsPerCore_ + 1) + (blockIdx_ - remainderItems_) * itemsPerCore_;
        }

        // Initialize input queues (bf16/fp16 for grad_output, hOut, x)
        pipe_->InitBuffer(gradOutTileQueue_, QUEUE_DEPTH, n_ * tileD_ * sizeof(T));
        pipe_->InitBuffer(hOutTileQueue_, QUEUE_DEPTH, tileD_ * sizeof(T));
        // float32 for hPost, hRes (use aligned sizes for vector ops)
        pipe_->InitBuffer(hPostQueue_, QUEUE_DEPTH, alignedN_ * sizeof(float));
        pipe_->InitBuffer(hResQueue_, QUEUE_DEPTH, alignedNN_ * sizeof(float));

        // Initialize output queues (bf16/fp16 for gradHOut, gradX)
        pipe_->InitBuffer(gradHOutTileQueue_, QUEUE_DEPTH, tileD_ * sizeof(T));
        pipe_->InitBuffer(gradXTileQueue_, QUEUE_DEPTH, n_ * tileD_ * sizeof(T));
        // float32 for gradHPost, gradHRes (use aligned sizes for vector ops)
        // pipe_->InitBuffer(gradHPostQueue_, QUEUE_DEPTH, alignedN_ * sizeof(float));
        pipe_->InitBuffer(gradHPostQueue_, QUEUE_DEPTH, n_ * tileD_ * sizeof(float));

        // Initialize intermediate buffers for float32 computation
        pipe_->InitBuffer(gradOutF32Buf_, n_ * tileD_ * sizeof(float));
    }

    if ASCEND_IS_AIC {
        if (blockIdx_ < remainderItemsAic_) {
            myItemCount_ = itemsPerAic_ + 1;
            itemStart_ = blockIdx_ * (itemsPerAic_ + 1);
        } else {
            myItemCount_ = itemsPerAic_;
            itemStart_ = remainderItemsAic_ * (itemsPerAic_ + 1) + (blockIdx_ - remainderItemsAic_) * itemsPerAic_;
        }

        mm.Init(&tilingData->matmulTiling, pipe_);
    }
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::Process()
{
    if ASCEND_IS_AIV {
        if (blockIdx_ >= usedCores_) {
            return;
        }

        for (uint32_t itemIdx = 0; itemIdx < myItemCount_; itemIdx++) {
            uint32_t globalItemIdx = itemStart_ + itemIdx;
            CopyInWeights(globalItemIdx);
            ComputeInitGrads();

            for (uint32_t tileId = 0; tileId < nTilesD_; tileId++) {
                CopyInTile(globalItemIdx, tileId);
                ComputeTile(tileId);
                // Split CopyOut to enable pipeline overlap
                CopyOutGradHOut(globalItemIdx, tileId);  // Can overlap with gradX computation
                CopyOutGradX(globalItemIdx, tileId);

                CopyOutGradHPost(globalItemIdx, tileId);
            }

            CopyOutWeightGrads(globalItemIdx);
        }
    }
    if ASCEND_IS_AIC {
        if (blockIdx_ >= usedAic_) {
            return;
        }

        for (uint32_t itemIdx = 0; itemIdx < myItemCount_; itemIdx++) {
            uint32_t globalItemIdx = itemStart_ + itemIdx;
            SetMMParaAndCompute(globalItemIdx);
        }
    }
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::CopyInWeights(uint32_t globalItemIdx)
{
    uint32_t hPostOffset = globalItemIdx * n_;
    uint32_t hResOffset = globalItemIdx * n_ * n_;

    LocalTensor<float> hPostLocal = hPostQueue_.AllocTensor<float>();
    if constexpr (IS_N_ALIGNED == 1) {
        // Fast path: n is aligned, direct copy
        DataCopy(hPostLocal, hPostGm_[hPostOffset], n_);
    } else {
        // Slow path: need padding, zero out first then copy
        uint8_t rightPad = static_cast<uint8_t>(alignedN_ - n_);
        DataCopyPadParams padParams = {true, 0, rightPad, 0};
        DataCopyPad(hPostLocal, hPostGm_[hPostOffset], {1, static_cast<uint16_t>(n_ * sizeof(float)), 0, 0}, padParams);
    }
    hPostQueue_.EnQue(hPostLocal);

    LocalTensor<float> hResLocal = hResQueue_.AllocTensor<float>();
    if constexpr (IS_NN_ALIGNED == 1) {
        // Fast path: n*n is aligned, direct copy
        DataCopy(hResLocal, hResGm_[hResOffset], n_ * n_);
    } else {
        // Slow path: need padding, zero out first then copy
        uint8_t rightPad = static_cast<uint8_t>(alignedNN_ - n_ * n_);
        DataCopyPadParams padParams = {true, 0, rightPad, 0};
        DataCopyPad(hResLocal, hResGm_[hResOffset], {1, static_cast<uint16_t>(n_ * n_ * sizeof(float)), 0, 0}, padParams);
    }
    hResQueue_.EnQue(hResLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::ComputeInitGrads()
{
    LocalTensor<float> gradHPostLocal = gradHPostQueue_.AllocTensor<float>();

    // Initialize gradHPost and gradHRes to zero
    // Use actual size when aligned, aligned size when not
    // if constexpr (IS_N_ALIGNED == 1) {
    //     Duplicate(gradHPostLocal, 0.0f, n_);
    // } else {
    //     Duplicate(gradHPostLocal, 0.0f, alignedN_);
    // }
    Duplicate(gradHPostLocal, 0.0f, n_ * tileD_);

    gradHPostQueue_.EnQue(gradHPostLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::CopyInTile(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint64_t gradOutBase = globalItemIdx * n_ * D_;
    uint64_t hOutBase = globalItemIdx * D_;

    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;
    uint32_t gmCopyD = (dStart + actualTileD > D_) ? (D_ - dStart) : actualTileD;

    LocalTensor<T> gradOutTileLocal = gradOutTileQueue_.AllocTensor<T>();
    LocalTensor<T> hOutTileLocal = hOutTileQueue_.AllocTensor<T>();

    if constexpr (IS_D_ALIGNED == 1) {
        // Fast path: D is aligned and single tile, use direct DataCopy
        DataCopy(hOutTileLocal, hOutGm_[hOutBase], tileD_);
        DataCopy(gradOutTileLocal, gradOutputGm_[gradOutBase], n_ * tileD_);
    } else {
        // Slow path: multi-tile or non-aligned
        // 数据在 buffer 中连续存放，第 i 行起始位置为 i * actualTileD

        // Load hOut tile
        if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
            DataCopy(hOutTileLocal, hOutGm_[hOutBase + dStart], gmCopyD);
        } else {
            // 需要 padding 到 actualTileD（已对齐到 16）
            uint8_t rightPad = static_cast<uint8_t>(actualTileD - gmCopyD);
            DataCopyPadParams padParams = {true, 0, rightPad, 0};
            DataCopyPad(hOutTileLocal, hOutGm_[hOutBase + dStart],
                    {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0}, padParams);
        }

        // Load grad_output and x row by row，连续存放
        for (uint32_t i = 0; i < n_; i++) {
            if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
                DataCopy(gradOutTileLocal[i * actualTileD], gradOutputGm_[gradOutBase + i * D_ + dStart], gmCopyD);
            } else {
                uint8_t rightPad = static_cast<uint8_t>(actualTileD - gmCopyD);
                DataCopyPadParams padParams = {true, 0, rightPad, 0};
                DataCopyPad(gradOutTileLocal[i * actualTileD], gradOutputGm_[gradOutBase + i * D_ + dStart],
                        {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0}, padParams);
            }
        }
    }

    gradOutTileQueue_.EnQue(gradOutTileLocal);
    hOutTileQueue_.EnQue(hOutTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::ComputeTile(uint32_t tileId)
{
    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;

    LocalTensor<T> gradOutTile = gradOutTileQueue_.DeQue<T>();
    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();
    LocalTensor<float> hPostLocal = hPostQueue_.DeQue<float>();
    LocalTensor<float> hResLocal = hResQueue_.DeQue<float>();
    LocalTensor<float> gradHPostLocal = gradHPostQueue_.DeQue<float>();

    LocalTensor<T> gradHOutTile = gradHOutTileQueue_.AllocTensor<T>();
    LocalTensor<T> gradXTile = gradXTileQueue_.AllocTensor<T>();

    // ===== Common path for bf16 and fp16: Cast to f32 for computation =====
    LocalTensor<float> gradOutF32 = gradOutF32Buf_.Get<float>();

    Cast(gradOutF32, gradOutTile, RoundMode::CAST_NONE, n_ * actualTileD);

    // Step 2: gradHPost[i] = sum_d(grad_output[i] * hOut)
    if (n_ == 4) {
        ComputeGradHPost<4>(gradHPostLocal, hOutTile, gradOutF32, actualTileD);
    } else if (n_ == 6) {
        ComputeGradHPost<6>(gradHPostLocal, hOutTile, gradOutF32, actualTileD);
    } else {
        ComputeGradHPost<8>(gradHPostLocal, hOutTile, gradOutF32, actualTileD);
    }

    gradHPostQueue_.EnQue(gradHPostLocal);

    ComputeGradHOutTile(gradOutF32, hPostLocal, gradHOutTile, actualTileD);
    gradHOutTileQueue_.EnQue(gradHOutTile);

    ComputeGradXTile(gradOutF32, hResLocal, gradXTile, actualTileD);
    gradXTileQueue_.EnQue(gradXTile);
    
    hPostQueue_.EnQue(hPostLocal);
    hResQueue_.EnQue(hResLocal);

    gradOutTileQueue_.FreeTensor(gradOutTile);
    hOutTileQueue_.FreeTensor(hOutTile);
}

TEMPLATE_DECLARE
template<uint16_t N>
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::ComputeGradHPost(LocalTensor<float> &gradHPostLocal, 
    LocalTensor<T> &hOutLocal, LocalTensor<float> &gradOutLocal, uint32_t dataLen)
{
    uint32_t vfLen = 256 / sizeof(float);
    __ubuf__ float *gradHPostAddr = (__ubuf__ float*)gradHPostLocal.GetPhyAddr();
    __ubuf__ T *hOutAddr = (__ubuf__ T*)hOutLocal.GetPhyAddr();
    __ubuf__ float *gradOutAddr = (__ubuf__ float*)gradOutLocal.GetPhyAddr();
    uint16_t loopNum = (dataLen + vfLen - 1) / vfLen;
    static constexpr Reg::CastTrait halfToFloatTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> hOutInReg;
        Reg::RegTensor<float> hOut32Reg;
        Reg::RegTensor<float> tmpReg;
        Reg::RegTensor<float> accumReg0;
        Reg::RegTensor<float> accumReg1;
        Reg::RegTensor<float> accumReg2;
        Reg::RegTensor<float> accumReg3;
        Reg::RegTensor<float> accumReg4;
        Reg::RegTensor<float> accumReg5;
        Reg::RegTensor<float> accumReg6;
        Reg::RegTensor<float> accumReg7;

        Reg::MaskReg curMask;
        Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        uint32_t curLen = dataLen;

        Reg::Duplicate(accumReg0, 0.0, allMask);
        Reg::Duplicate(accumReg1, 0.0, allMask);
        Reg::Duplicate(accumReg2, 0.0, allMask);
        Reg::Duplicate(accumReg3, 0.0, allMask);
        if constexpr (N >= 6) {
            Reg::Duplicate(accumReg4, 0.0, allMask);
            Reg::Duplicate(accumReg5, 0.0, allMask);
        }
        if constexpr (N == 8) {
            Reg::Duplicate(accumReg6, 0.0, allMask);
            Reg::Duplicate(accumReg7, 0.0, allMask);
        }
        auto curGradOutAddr = gradOutAddr;
        for (uint16_t i = 0; i < loopNum; i++) {
            curMask = Reg::UpdateMask<float>(curLen);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_US_B16>(hOutInReg, hOutAddr + i * vfLen);
            Reg::Cast<float, T, halfToFloatTrait>(hOut32Reg, hOutInReg, allMask);

            auto curGradHPostAddr = gradHPostAddr + i * vfLen;
            
            Reg::LoadAlign(tmpReg, curGradOutAddr);
            Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
            Reg::StoreAlign(curGradHPostAddr, tmpReg, curMask);

            Reg::LoadAlign(tmpReg, curGradOutAddr + dataLen);
            Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
            Reg::StoreAlign(curGradHPostAddr + tileD_, tmpReg, curMask);

            Reg::LoadAlign(tmpReg, curGradOutAddr + 2 *dataLen);
            Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
            Reg::StoreAlign(curGradHPostAddr + 2 * tileD_, tmpReg, curMask);

            Reg::LoadAlign(tmpReg, curGradOutAddr + 3 *dataLen);
            Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
            Reg::StoreAlign(curGradHPostAddr + 3 * tileD_, tmpReg, curMask);

            if constexpr (N >= 6) {
                Reg::LoadAlign(tmpReg, curGradOutAddr + 4 *dataLen);
                Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
                Reg::StoreAlign(curGradHPostAddr + 4 * tileD_, tmpReg, curMask);

                Reg::LoadAlign(tmpReg, curGradOutAddr + 5 *dataLen);
                Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
                Reg::StoreAlign(curGradHPostAddr + 5 * tileD_, tmpReg, curMask);
            }
            if constexpr (N == 8) {
                Reg::LoadAlign(tmpReg, curGradOutAddr + 6 *dataLen);
                Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
                Reg::StoreAlign(curGradHPostAddr + 6 * tileD_, tmpReg, curMask);

                Reg::LoadAlign(tmpReg, curGradOutAddr + 7 *dataLen);
                Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
                Reg::StoreAlign(curGradHPostAddr + 7 * tileD_, tmpReg, curMask);
            }
            curGradOutAddr += vfLen;
        }
    }
}

// TEMPLATE_DECLARE
// template<uint16_t N>
// __aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::ComputeGradHPost(LocalTensor<float> &gradHPostLocal, 
//     LocalTensor<T> &hOutLocal, LocalTensor<float> &gradOutLocal, uint32_t dataLen)
// {
//     uint32_t vfLen = 256 / sizeof(float);
//     __ubuf__ float *gradHPostAddr = (__ubuf__ float*)gradHPostLocal.GetPhyAddr();
//     __ubuf__ T *hOutAddr = (__ubuf__ T*)hOutLocal.GetPhyAddr();
//     __ubuf__ float *gradOutAddr = (__ubuf__ float*)gradOutLocal.GetPhyAddr();
//     uint16_t loopNum = (dataLen + vfLen - 1) / vfLen;
//     static constexpr Reg::CastTrait halfToFloatTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
//                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
//     __VEC_SCOPE__
//     {
//         Reg::RegTensor<T> hOutInReg;
//         Reg::RegTensor<float> hOut32Reg;
//         Reg::RegTensor<float> tmpReg;
//         Reg::RegTensor<float> accumReg0;
//         Reg::RegTensor<float> accumReg1;
//         Reg::RegTensor<float> accumReg2;
//         Reg::RegTensor<float> accumReg3;
//         Reg::RegTensor<float> accumReg4;
//         Reg::RegTensor<float> accumReg5;
//         Reg::RegTensor<float> accumReg6;
//         Reg::RegTensor<float> accumReg7;

//         Reg::MaskReg curMask;
//         Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
//         uint32_t curLen = dataLen;

//         Reg::Duplicate(accumReg0, 0.0, allMask);
//         Reg::Duplicate(accumReg1, 0.0, allMask);
//         Reg::Duplicate(accumReg2, 0.0, allMask);
//         Reg::Duplicate(accumReg3, 0.0, allMask);
//         if constexpr (N >= 6) {
//             Reg::Duplicate(accumReg4, 0.0, allMask);
//             Reg::Duplicate(accumReg5, 0.0, allMask);
//         }
//         if constexpr (N == 8) {
//             Reg::Duplicate(accumReg6, 0.0, allMask);
//             Reg::Duplicate(accumReg7, 0.0, allMask);
//         }
//         auto curGradOutAddr = gradOutAddr;
//         for (uint16_t i = 0; i < loopNum; i++) {
//             curMask = Reg::UpdateMask<float>(curLen);
//             Reg::LoadAlign<T, Reg::LoadDist::DIST_US_B16>(hOutInReg, hOutAddr + i * vfLen);
//             Reg::Cast<float, T, halfToFloatTrait>(hOut32Reg, hOutInReg, allMask);
            
//             Reg::LoadAlign(tmpReg, curGradOutAddr);
//             Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//             Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg0, accumReg0, tmpReg, curMask);
//             Reg::LoadAlign(tmpReg, curGradOutAddr + dataLen);
//             Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//             Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg1, accumReg1, tmpReg, curMask);
//             Reg::LoadAlign(tmpReg, curGradOutAddr + 2 *dataLen);
//             Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//             Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg2, accumReg2, tmpReg, curMask);
//             Reg::LoadAlign(tmpReg, curGradOutAddr + 3 *dataLen);
//             Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//             Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg3, accumReg3, tmpReg, curMask);
//             if constexpr (N >= 6) {
//                 Reg::LoadAlign(tmpReg, curGradOutAddr + 4 *dataLen);
//                 Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//                 Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg4, accumReg4, tmpReg, curMask);
//                 Reg::LoadAlign(tmpReg, curGradOutAddr + 5 *dataLen);
//                 Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//                 Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg5, accumReg5, tmpReg, curMask);
//             }
//             if constexpr (N == 8) {
//                 Reg::LoadAlign(tmpReg, curGradOutAddr + 6 *dataLen);
//                 Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//                 Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg6, accumReg6, tmpReg, curMask);
//                 Reg::LoadAlign(tmpReg, curGradOutAddr + 7 *dataLen);
//                 Reg::Mul(tmpReg, hOut32Reg, tmpReg, curMask);
//                 Reg::Add<float, Reg::MaskMergeMode::MERGING>(accumReg7, accumReg7, tmpReg, curMask);
//             }
//             curGradOutAddr += vfLen;
//         }
//         Reg::Reduce<Reg::ReduceType::SUM>(accumReg0, accumReg0, allMask);
//         Reg::Reduce<Reg::ReduceType::SUM>(accumReg1, accumReg1, allMask);
//         Reg::Reduce<Reg::ReduceType::SUM>(accumReg2, accumReg2, allMask);
//         Reg::Reduce<Reg::ReduceType::SUM>(accumReg3, accumReg3, allMask);
//         if constexpr (N >= 6) {
//             Reg::Reduce<Reg::ReduceType::SUM>(accumReg4, accumReg4, allMask);
//             Reg::Reduce<Reg::ReduceType::SUM>(accumReg5, accumReg5, allMask);
//         }
//         if constexpr (N == 8) {
//             Reg::Reduce<Reg::ReduceType::SUM>(accumReg6, accumReg6, allMask);
//             Reg::Reduce<Reg::ReduceType::SUM>(accumReg7, accumReg7, allMask);
//         }

//         if constexpr (N == 4) {
//             Interleave(accumReg0, accumReg2, accumReg0, accumReg2);
//             Interleave(accumReg1, accumReg3, accumReg1, accumReg3);
//             Interleave(accumReg0, accumReg1, accumReg0, accumReg1);
//         }
//         if constexpr (N == 6) {
//             Interleave(accumReg0, accumReg4, accumReg0, accumReg4);
//             Interleave(accumReg0, accumReg2, accumReg0, accumReg2);
//             Interleave(accumReg1, accumReg5, accumReg1, accumReg5);
//             Interleave(accumReg1, accumReg3, accumReg1, accumReg3);
//             Interleave(accumReg0, accumReg1, accumReg0, accumReg1);
//         }
//         if constexpr (N == 8) {
//             Interleave(accumReg0, accumReg4, accumReg0, accumReg4);
//             Interleave(accumReg2, accumReg6, accumReg2, accumReg6);
//             Interleave(accumReg0, accumReg2, accumReg0, accumReg2);
//             Interleave(accumReg1, accumReg5, accumReg1, accumReg5);
//             Interleave(accumReg3, accumReg7, accumReg3, accumReg7);
//             Interleave(accumReg1, accumReg3, accumReg1, accumReg3);
//             Interleave(accumReg0, accumReg1, accumReg0, accumReg1);
//         }
//         Reg::LoadAlign(tmpReg, gradHPostAddr);
//         Reg::Add<float, Reg::MaskMergeMode::MERGING>(tmpReg, tmpReg, accumReg0, allMask);
//         Reg::StoreAlign(gradHPostAddr, tmpReg, allMask);
//     }
// }

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::ComputeGradHOutTile(LocalTensor<float> gradOutF32UB, LocalTensor<float> hPostLocalUB, LocalTensor<T> gradHOutTileUB, uint32_t actualTileD){
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (actualTileD + vfLen - 1) / vfLen;
    uint16_t ntimes = n_;

    auto gradOutTileAddr = (__ubuf__ float*)gradOutF32UB.GetPhyAddr();
    auto hPostLocalAddr = (__ubuf__ float*)hPostLocalUB.GetPhyAddr();
    auto gradHOutTileAddr = (__ubuf__ T*)gradHOutTileUB.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t gradOutDealNum = static_cast<uint32_t>(actualTileD);
        AscendC::MicroAPI::RegTensor<float> gradOutF32Reg;
        AscendC::MicroAPI::RegTensor<float> hPostLocalReg;
        AscendC::MicroAPI::RegTensor<float> gradHOutTileReg;
        AscendC::MicroAPI::RegTensor<T> gradHOutTileB16Reg;
         AscendC::MicroAPI::MaskReg pMask;
        AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for(uint16_t k = 0; k < repeatTimes; k ++ ){
            pMask = AscendC::MicroAPI::UpdateMask<float>(gradOutDealNum);
            AscendC::MicroAPI::Duplicate(gradHOutTileReg, 0.0f);
            for (uint16_t i = 0; i < ntimes; i++) {
                AscendC::MicroAPI::LoadAlign(gradOutF32Reg , gradOutTileAddr + i * actualTileD + k * vfLen);
                AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(hPostLocalReg , hPostLocalAddr + i);
                AscendC::MicroAPI::MulAddDst(gradHOutTileReg, gradOutF32Reg, hPostLocalReg, pMask);
            }
            AscendC::MicroAPI::Cast<T, float, castB32ToB16>(gradHOutTileB16Reg , gradHOutTileReg, pMask);
            AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(gradHOutTileAddr + k * vfLen, gradHOutTileB16Reg, pMask);                
        }
    }
}


TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::ComputeGradXTile(LocalTensor<float> gradOutF32UB, LocalTensor<float> hResLocalUB, LocalTensor<T> gradXTileUB, uint32_t actualTileD){
    uint32_t vfLen = 256 / sizeof(float);
    uint16_t repeatTimes = (actualTileD + vfLen - 1) / vfLen;
    uint16_t ntimes = n_;

    auto gradOutTileAddr = (__ubuf__ float*)gradOutF32UB.GetPhyAddr();
    auto hResLocalAddr = (__ubuf__ float*)hResLocalUB.GetPhyAddr();
    auto gradXTileAddr = (__ubuf__ T*)gradXTileUB.GetPhyAddr();
    
    __VEC_SCOPE__
    {
        uint32_t curDealNum = vfLen;
        AscendC::MicroAPI::RegTensor<float> gradOutF32Reg;
        AscendC::MicroAPI::RegTensor<float> hResLocalReg;
        AscendC::MicroAPI::RegTensor<float> gradXTileReg;
        AscendC::MicroAPI::RegTensor<T> gradXTileB16Reg;
        AscendC::MicroAPI::MaskReg pMask;
        AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < ntimes; i++) {
            for(uint16_t k = 0; k < repeatTimes; k ++ ){
                curDealNum = (k == repeatTimes - 1) ? (actualTileD % vfLen == 0 ? vfLen : actualTileD % vfLen) : vfLen;
                pMask = AscendC::MicroAPI::UpdateMask<float>(curDealNum);
                AscendC::MicroAPI::Duplicate(gradXTileReg, 0.0f, pregMain);
                for (uint16_t j = 0; j < ntimes; j++) {
                    AscendC::MicroAPI::LoadAlign(gradOutF32Reg , gradOutTileAddr + j * actualTileD + k * vfLen);
                    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(hResLocalReg , hResLocalAddr + i * n_ + j);
                    AscendC::MicroAPI::MulAddDst(gradXTileReg, gradOutF32Reg, hResLocalReg, pMask);
                }
                AscendC::MicroAPI::Cast<T, float, castB32ToB16>(gradXTileB16Reg , gradXTileReg, pMask);
                AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(gradXTileAddr + i * actualTileD + k * vfLen, gradXTileB16Reg, pMask);                
            }
        }

    }
}



TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::CopyOutGradHOut(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint64_t hOutBase = globalItemIdx * D_;

    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;
    uint32_t gmCopyD = (dStart + actualTileD > D_) ? (D_ - dStart) : actualTileD;

    LocalTensor<T> gradHOutTile = gradHOutTileQueue_.DeQue<T>();

    if constexpr (IS_D_ALIGNED == 1) {
        // Fast path: D is aligned and single tile
        DataCopy(gradHOutGm_[hOutBase], gradHOutTile, tileD_);
    } else {
        // Copy gradHOut
        if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
            DataCopy(gradHOutGm_[hOutBase + dStart], gradHOutTile, gmCopyD);
        } else {
            DataCopyPad(gradHOutGm_[hOutBase + dStart], gradHOutTile,
                    {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0});
        }
    }

    gradHOutTileQueue_.FreeTensor(gradHOutTile);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::CopyOutGradX(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint64_t gradXBase = globalItemIdx * n_ * D_;

    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;
    uint32_t gmCopyD = (dStart + actualTileD > D_) ? (D_ - dStart) : actualTileD;

    LocalTensor<T> gradXTile = gradXTileQueue_.DeQue<T>();

    if constexpr (IS_D_ALIGNED == 1) {
        // Fast path: D is aligned and single tile
        DataCopy(gradXGm_[gradXBase], gradXTile, n_ * tileD_);
    } else {
        // Copy gradX row by row，从 buffer 的连续位置读取
        for (uint32_t i = 0; i < n_; i++) {
            if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
                DataCopy(gradXGm_[gradXBase + i * D_ + dStart], gradXTile[i * actualTileD], gmCopyD);
            } else {
                DataCopyPad(gradXGm_[gradXBase + i * D_ + dStart], gradXTile[i * actualTileD],
                        {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0});
            }
        }
    }

    gradXTileQueue_.FreeTensor(gradXTile);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::CopyOutGradHPost(uint32_t globalItemIdx, uint32_t tileId)
{
    uint32_t dStart = tileId * tileD_;
    uint64_t gradHPostBase = globalItemIdx * n_ * D_;

    uint32_t actualTileD = (tileId == nTilesD_ - 1) ? lastTileD_ : tileD_;
    uint32_t gmCopyD = (dStart + actualTileD > D_) ? (D_ - dStart) : actualTileD;

    LocalTensor<float> gradHPostTile = gradHPostQueue_.DeQue<T>();

    if constexpr (IS_D_ALIGNED == 1) {
        // Fast path: D is aligned and single tile
        DataCopy(gradHPostGm_[gradHPostBase], gradHPostTile, n_ * tileD_);
    } else {
        // Copy gradX row by row，从 buffer 的连续位置读取
        for (uint32_t i = 0; i < n_; i++) {
            if (gmCopyD % BF16_FP16_ALIGN_SIZE == 0) {
                DataCopy(gradHPostGm_[gradHPostBase + i * D_ + dStart], gradHPostTile[i * actualTileD], gmCopyD);
            } else {
                DataCopyPad(gradHPostGm_[gradHPostBase + i * D_ + dStart], gradHPostTile[i * actualTileD],
                        {1, static_cast<uint16_t>(gmCopyD * sizeof(T)), 0, 0});
            }
        }
    }

    gradHPostQueue_.FreeTensor(gradHPostTile);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::CopyOutWeightGrads(uint32_t globalItemIdx)
{
    // uint32_t hPostOffset = globalItemIdx * n_;

    LocalTensor<float> hPostLocal = hPostQueue_.DeQue<float>();
    LocalTensor<float> hResLocal = hResQueue_.DeQue<float>();
    hPostQueue_.FreeTensor(hPostLocal);
    hResQueue_.FreeTensor(hResLocal);

    // LocalTensor<float> gradHPostLocal = gradHPostQueue_.DeQue<float>();
    // if constexpr (IS_N_ALIGNED == 1) {
    //     DataCopy(gradHPostGm_[hPostOffset], gradHPostLocal, n_);
    // } else {
    //     DataCopyPad(gradHPostGm_[hPostOffset], gradHPostLocal, {1, static_cast<uint16_t>(n_ * sizeof(float)), 0, 0});
    // }
    // gradHPostQueue_.FreeTensor(gradHPostLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostBackwardKernel<TEMPLATE_ARGS>::SetMMParaAndCompute(
    uint32_t globalItemIdx)
{
    uint64_t inputBase = globalItemIdx * n_ * D_;
    uint32_t outputBase = globalItemIdx * n_ * n_;
    uint32_t curSingleCoreK = singleCoreK_;

    for (uint32_t offsetK = 0; offsetK < D_; offsetK += singleCoreK_) {
        if (offsetK + singleCoreK_ > D_) {
            curSingleCoreK = D_ - offsetK;
        }
        mm.SetSingleShape(n_, n_, curSingleCoreK);
        mm.SetTensorA(xGm_[inputBase + offsetK]);
        mm.SetTensorB(gradOutputGm_[inputBase + offsetK], true);
        mm.IterateAll(gradHResGm_[outputBase], offsetK == 0 ? 0 : 1);
    }
    mm.End();
}

} // namespace MhcPostBackward

#endif // MHC_POST_BACKWARD_H