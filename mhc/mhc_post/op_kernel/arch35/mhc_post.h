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
 * \file mhc_post.h
 * \brief MhcPost kernel implementation
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 *          where: (H_{l}^{res})^{T} * x_l represents matrix multiplication with transposed h_res
 *                h_{l}^{out} * H_{t}^{post} represents element-wise multiplication and broadcasting
 */

#ifndef ASCENDC_MHC_POST_H
#define ASCENDC_MHC_POST_H

#include "kernel_operator.h"
#include "basic_api/kernel_operator_utils_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_post_tiling_data.h"
#include "mhc_post_tiling_key.h"

namespace MhcPost {
using namespace AscendC;

// Double Buffer configuration - Double Buffer提升Memory Bound算子性能
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2;

#define TEMPLATE_DECLARE template <typename T, uint16_t USE_PERMANENT_X>
#define TEMPLATE_ARGS T, USE_PERMANENT_X

TEMPLATE_DECLARE
class MhcPostKernel {
public:
    __aicore__ inline MhcPostKernel(TPipe *tPipe, const MhcPostTilingData *__restrict tilingData)
        : pipe_(tPipe),
          tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR output,
                                GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void CopyInX(int64_t bsIdx, int64_t nIdx, int64_t nNum, int64_t dIdx, int64_t dNum);
    __aicore__ inline void CopyInHPost(int64_t bsIdx, int64_t nIdx, int64_t nNum);
    __aicore__ inline void CopyInHRes(int64_t bsIdx, int64_t nIdx, int64_t nNum);
    __aicore__ inline void DoMulAndAdd(LocalTensor<float> hPostUb, LocalTensor<float> hResUb, int64_t nNum, int64_t bsIdx, int64_t dIdx, int64_t dNum);
    __aicore__ inline void CopyOutY(int64_t bsIdx, int64_t nIdx, int64_t nNum, int64_t dIdx, int64_t dNum);
    __aicore__ inline uint16_t CeilDiv(uint32_t a, uint16_t b);

private:
    TPipe *pipe_;
    const MhcPostTilingData *tilingData_;

    // Input queues - Double Buffer enabled (queue depth = DOUBLE_BUFFER_DEPTH)
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> hOutTileQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> xTileQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> hResTileQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> hPostTileQueue_;

    // Output queues - Double Buffer enabled (queue depth = DOUBLE_BUFFER_DEPTH)
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_DEPTH> yTileQueue_;
    TBuf<QuePosition::VECCALC> yTileBuf_;

    // Global memory tensors - inputs (bf16/fp16)
    GlobalTensor<T> xGm_;
    GlobalTensor<T> hOutGm_;
    // Global memory tensors - inputs (float32)
    GlobalTensor<float> hResGm_;
    GlobalTensor<float> hPostGm_;

    // Global memory tensors - outputs (bf16/fp16)
    GlobalTensor<T> outputGm_;
    static constexpr uint16_t VECTOR_LENGTH = 256;
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);

    int64_t curCoreItemCount_;
    int64_t curBS_;
    int64_t curN_;
    int64_t curD_;
    int64_t itemStart_;
    uint32_t blockIdx_;
    constexpr static AscendC::MicroAPI::CastTrait castB16ToB32 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
    constexpr static AscendC::MicroAPI::CastTrait castB32ToB16 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::MERGING, AscendC::RoundMode::CAST_RINT };
};

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::Init(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                                                          GM_ADDR output, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    // Calculate work distribution with remainder handling
    if (blockIdx_ < tilingData_->usedCoreNum - 1) {
        curBS_ = tilingData_->bsInner;
    } else {
        curBS_ = tilingData_->bsTail;
    }
    // itemStart_ = blockIdx_ * tilingData_->normalCoreProcessNum;

    // Set global memory buffers
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    hOutGm_.SetGlobalBuffer((__gm__ T *)hOut);
    hResGm_.SetGlobalBuffer((__gm__ float *)hRes);
    hPostGm_.SetGlobalBuffer((__gm__ float *)hPost);
    outputGm_.SetGlobalBuffer((__gm__ T *)output);

    // Initialize input queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH for data tiles
    pipe_->InitBuffer(hResTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->n * tilingData_->nInner * sizeof(float));
    pipe_->InitBuffer(hPostTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->nInner * sizeof(float));

    pipe_->InitBuffer(hOutTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->dInner * sizeof(T));
    pipe_->InitBuffer(xTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->nInner * tilingData_->dInner * sizeof(T));


    // Initialize output queues - Double Buffer with depth=DOUBLE_BUFFER_DEPTH
    pipe_->InitBuffer(yTileBuf_, tilingData_->nInner * tilingData_->dInner * sizeof(float));
    pipe_->InitBuffer(yTileQueue_, DOUBLE_BUFFER_DEPTH, tilingData_->n * tilingData_->dInner * sizeof(T));
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::DoMulAndAdd(LocalTensor<float> hPostUb, LocalTensor<float> hResUb, int64_t nNum, int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    uint16_t nTimes = static_cast<uint16_t>(nNum);
    // if (blockIdx_ == 0){
    //     // printf("nNum %ld\n", nNum);
    //     printf("2222\n");
    // }
    uint16_t n_ = static_cast<uint16_t>(tilingData_->n);
    uint16_t dRepeatTimes = CeilDiv(dNum, VL_FP32);
    uint16_t dNumTmp = static_cast<uint16_t>(dNum);
    uint16_t dInner = static_cast<uint16_t>(tilingData_->dInner);
    uint16_t dAlign = static_cast<uint16_t>(CeilDiv(dNum, 16) * 16);
    uint16_t nAlign = static_cast<uint16_t>(CeilDiv(nTimes, 8) * 8);

    LocalTensor<T> xUb = xTileQueue_.DeQue<T>();
    LocalTensor<T> hOutUb = hOutTileQueue_.DeQue<T>();
    
    LocalTensor<float> yTileBuf = yTileBuf_.Get<float>();

    auto xAddr = (__ubuf__ T*)xUb.GetPhyAddr();
    auto hResAddr = (__ubuf__ float*)hResUb.GetPhyAddr();
    auto hPostAddr = (__ubuf__ float*)hPostUb.GetPhyAddr();
    auto hOutAddr = (__ubuf__ T*)hOutUb.GetPhyAddr();
    // auto yAddr = (__ubuf__ T*)yTileLocal.GetPhyAddr();
    auto yAddr = (__ubuf__ float*)yTileBuf.GetPhyAddr();

    // if (blockIdx_ == 0){
    //     // printf("x %f\n", static_cast<float>(xUb.GetValue(30)));
    //     // printf("x %f\n", static_cast<float>(xUb.GetValue(31)));
    //     // printf("x %f\n", static_cast<float>(xUb.GetValue(32)));
    //     printf("dAlign %f\n", static_cast<float>(dAlign));
    //     printf("nTimes %f\n", static_cast<float>(nTimes));
    //     printf("nAlign %f\n", static_cast<float>(nAlign));
    // }

     __VEC_SCOPE__
    {
		AscendC::MicroAPI::RegTensor<T> xReg;
        AscendC::MicroAPI::RegTensor<T> hOutReg;
        AscendC::MicroAPI::RegTensor<float> xRegFloat;
        AscendC::MicroAPI::RegTensor<float> hOutRegFloat;
        AscendC::MicroAPI::RegTensor<float> hResReg;
        AscendC::MicroAPI::RegTensor<float> hPostReg;
        AscendC::MicroAPI::RegTensor<float> outRegFloat;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t nIndex = 0; nIndex < nTimes; nIndex++) {
            uint32_t dNumU32 = static_cast<uint32_t>(dAlign);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(hPostReg, hPostAddr + nIndex);
            for (uint16_t j = 0; j < dRepeatTimes; j++) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(dNumU32);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(hOutReg, hOutAddr + j * VL_FP32);
                AscendC::MicroAPI::Cast<float, T, castB16ToB32>(hOutRegFloat, hOutReg, pMask);
                AscendC::MicroAPI::Mul(outRegFloat, hOutRegFloat, hPostReg, pMask);
                for (uint16_t i = 0; i < n_; i++) {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + i * dAlign + j * VL_FP32);
                    // AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddr + i * dInner + j * VL_FP32);
                    AscendC::MicroAPI::Cast<float, T, castB16ToB32>(xRegFloat, xReg, pMask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(hResReg, hResAddr + i * nAlign + nIndex);
                    AscendC::MicroAPI::MulAddDst(outRegFloat, xRegFloat, hResReg, pMask);
                }
                AscendC::MicroAPI::DataCopy(yAddr + nIndex * dAlign + j * VL_FP32, outRegFloat, pMask);
                // AscendC::MicroAPI::DataCopy(yAddr + nIndex * dNumTmp + j * VL_FP32, outRegFloat, pMask);
            }
        }
    }
    
    // if (blockIdx_ == 0){
    //     printf("x %f\n", static_cast<float>(xUb.GetValue(0)));
    //     printf("x %f\n", static_cast<float>(xUb.GetValue(16)));
    //     printf("x %f\n", static_cast<float>(xUb.GetValue(128)));
    //     printf("x %f\n", static_cast<float>(hResUb.GetValue(0)));
    //     printf("x %f\n", static_cast<float>(hResUb.GetValue(8)));
    //     printf("x %f\n", static_cast<float>(hResUb.GetValue(9)));
    //     printf("x %f\n", static_cast<float>(hResUb.GetValue(16)));
    // }
    hOutTileQueue_.FreeTensor(hOutUb);
    xTileQueue_.FreeTensor(xUb); 
    
    LocalTensor<T> yTileLocal = yTileQueue_.AllocTensor<T>();
    AscendC::Cast(yTileLocal, yTileBuf, AscendC::RoundMode::CAST_RINT, nTimes * dAlign);
    yTileQueue_.EnQue(yTileLocal);

}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t nLoopTime = tilingData_->nOuter;
    int64_t dLoopTime = tilingData_->dOuter;

    int64_t bsStart = blockIdx_ * tilingData_->bsInner;
    for (int64_t itemIdx = 0; itemIdx < curBS_; itemIdx++) {
        int64_t globalItemIdx = itemStart_ + itemIdx;
        int64_t bsIdx = bsStart + itemIdx;
        // int64_t dIdx = globalItemIdx - bsIdx * tilingData_->dOuter;
        for (int64_t nIdx = 0; nIdx < nLoopTime; nIdx++) {
            int64_t nNum = (nIdx < nLoopTime - 1) ? tilingData_->nInner : tilingData_->nTail;
            CopyInHRes(bsIdx, nIdx, nNum);
            CopyInHPost(bsIdx, nIdx, nNum);
            LocalTensor<float> hPostUb = hPostTileQueue_.DeQue<float>();
            LocalTensor<float> hResUb = hResTileQueue_.DeQue<float>();
            for (int64_t dIdx = 0; dIdx < dLoopTime; dIdx++) {
                int64_t dNum = (dIdx < dLoopTime - 1) ? tilingData_->dInner : tilingData_->dTail;
                CopyInHOut(bsIdx, dIdx, dNum);
                CopyInX(bsIdx, nIdx, nNum, dIdx, dNum);
                DoMulAndAdd(hPostUb, hResUb, nNum, bsIdx, dIdx, dNum);

                CopyOutY(bsIdx, nIdx, nNum, dIdx, dNum);
            }
            hPostTileQueue_.FreeTensor(hPostUb);
            hResTileQueue_.FreeTensor(hResUb); 
        }
    }
    // printf("6\n");
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInHOut(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hOutOffset = bsIdx * tilingData_->d + dIdx * tilingData_->dInner;
    LocalTensor<T> hOutTileLocal = hOutTileQueue_.AllocTensor<T>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(dNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(hOutTileLocal, hOutGm_[hOutOffset], copyParams, {false, 0, 0, 0});

    hOutTileQueue_.EnQue<T>(hOutTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInHPost(int64_t bsIdx, int64_t nIdx, int64_t nNum)
{
    int64_t hPostOffset = bsIdx * tilingData_->n + nIdx * tilingData_->nInner;
    LocalTensor<float> hPostTileLocal = hPostTileQueue_.AllocTensor<float>();

    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(nNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(hPostTileLocal, hPostGm_[hPostOffset], copyParams, {false, 0, 0, 0});
    hPostTileQueue_.EnQue<float>(hPostTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInHRes(int64_t bsIdx, int64_t nIdx, int64_t nNum)
{
    // int64_t nStart = nIdx * tilingData_->nInner * tilingData_->d;
    int64_t hResBase = bsIdx * tilingData_->n * tilingData_->n;
    int64_t nStart = nIdx * tilingData_->nInner;
    int64_t hResOffset = hResBase + nStart;
    // if (blockIdx_ == 0){
    //     // printf("x %f\n", static_cast<float>(xUb.GetValue(30)));
    //     // printf("x %f\n", static_cast<float>(xUb.GetValue(31)));
    //     // printf("x %f\n", static_cast<float>(xUb.GetValue(32)));
    //     printf("hResOffset %ld\n", static_cast<int64_t>(hResOffset));
    //     printf("nNum %ld\n", static_cast<int64_t>(nNum));
    //     printf("tilingData_->n %ld\n", static_cast<int64_t>(tilingData_->n));
    // }
    LocalTensor<float> hResTileLocal = hResTileQueue_.AllocTensor<float>();
     DataCopyExtParams copyParams = {static_cast<uint16_t>(tilingData_->n), static_cast<uint32_t>(nNum * sizeof(float)),
                                    static_cast<uint32_t>((tilingData_->n - nNum) * sizeof(float)), 0, 0};

    // DataCopyExtParams copyParams = {1, static_cast<uint32_t>(nNum  * tilingData_->n * sizeof(float)), 0, 0, 0};
    DataCopyPad(hResTileLocal, hResGm_[hResOffset], copyParams, {false, 0, 0, 0});

    hResTileQueue_.EnQue<float>(hResTileLocal);
}

// TEMPLATE_DECLARE
// __aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInHRes(int64_t bsIdx, int64_t nIdx, int64_t nNum)
// {
//     int64_t hResOffset = bsIdx * tilingData_->n * tilingData_->n + nIdx * tilingData_->nInner * tilingData_->n;
//     LocalTensor<float> hResTileLocal = hResTileQueue_.AllocTensor<float>();

//     DataCopyExtParams copyParams = {1, static_cast<uint32_t>(nNum  * tilingData_->n * sizeof(float)), 0, 0, 0};
//     DataCopyPad(hResTileLocal, hResGm_[hResOffset], copyParams, {false, 0, 0, 0});

//     hResTileQueue_.EnQue<float>(hResTileLocal);
// }

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyInX(int64_t bsIdx, int64_t nIdx, int64_t nNum, int64_t dIdx, int64_t dNum)
{
    int64_t dStart = dIdx * tilingData_->dInner;
    // int64_t nStart = nIdx * tilingData_->nInner * tilingData_->d;
    int64_t xBase = bsIdx * tilingData_->n * tilingData_->d;
    int64_t nStart = nIdx * tilingData_->nInner * tilingData_->d;
    int64_t xOffset = xBase + dStart;
    LocalTensor<T> xTileLocal = xTileQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(tilingData_->n), static_cast<uint32_t>(dNum * sizeof(T)),
                                    static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)), 0, 0};
    // DataCopyExtParams copyParams = {static_cast<uint16_t>(tilingData_->n), static_cast<uint32_t>(dNum * sizeof(T)),
    //                             static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)), offset, 0};
    DataCopyPad(xTileLocal, xGm_[xOffset], copyParams, {false, 0, 0, 0});

    xTileQueue_.EnQue<T>(xTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::CopyOutY(int64_t bsIdx, int64_t nIdx, int64_t nNum, int64_t dIdx, int64_t dNum)
{
    int64_t dStart = dIdx * tilingData_->dInner;
    int64_t nStart = nIdx * tilingData_->nInner * tilingData_->d;
    int64_t yBase = bsIdx * tilingData_->n * tilingData_->d;
    int64_t yOffset = yBase + nStart + dStart;
    int32_t offset = CeilDiv(static_cast<uint32_t>((tilingData_->dInner - dNum) * sizeof(T)), 32);
    LocalTensor<T> yTileLocal = yTileQueue_.DeQue<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(nNum), static_cast<uint32_t>(dNum * sizeof(T)),
                                    0, static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)), 0};
    // DataCopyExtParams copyParams = {static_cast<uint16_t>(nNum), static_cast<uint32_t>(dNum * sizeof(T)),
    //                             offset, static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)), 0};
    // if (blockIdx_ == 0){
    //     printf("yOffset %ld\n", yOffset);
    // }
    
    // printf("nNum %ld\n", nNum);
    // printf("dNum %ld\n", dNum);
    DataCopyPad(outputGm_[yOffset], yTileLocal, copyParams);
        
    yTileQueue_.FreeTensor(yTileLocal);
}

TEMPLATE_DECLARE
__aicore__ inline uint16_t MhcPostKernel<TEMPLATE_ARGS>::CeilDiv(uint32_t a, uint16_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

}  // namespace MhcPost

#endif  // ASCENDC_MHC_POST_H