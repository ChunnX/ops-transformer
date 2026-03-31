/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_backward.h
 * \brief
 */

#ifndef __mhc_pre_backward_KERNEL_H_
#define __mhc_pre_backward_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MhcPreBackward {

template <typename T>
__aicore__ auto CeilAlign(T a, T b) -> T
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ auto CeilDiv(T a, T b) -> T
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T lhs, T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint16_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

using namespace matmul;
using namespace AscendC;

constexpr MicroAPI::CastTrait ctFp32To16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
 	                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait ctHalf2Fp32Zero = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint64_t SYNC_MAX_NUM = 15;
constexpr uint32_t INOUT_QUEUE_SIZE = 16 * 1024;          // 16KB
constexpr uint32_t FP32_BUF_SIZE = (248 - 16 * 5) * 1024; // 128KB
constexpr uint32_t PROCESS_V2_CHUNK_SIZE = 64;            // ProcessV2函数使用的chunk大小
constexpr uint32_t SINGLE_M = 1024;
constexpr uint32_t ND_BLOCK_SIZE = 128;
constexpr uint32_t ALPHA_GRAD_PADDING = 24;

struct InitParams {
    GM_ADDR x;
    GM_ADDR phi;
    GM_ADDR alpha;
    GM_ADDR h_in_grad;
    GM_ADDR h_post_grad;
    GM_ADDR h_comb_before_grad;
    GM_ADDR inv_rms;
    GM_ADDR mm_res;
    GM_ADDR h_pre;
    GM_ADDR h_post;
    GM_ADDR gamma;

    GM_ADDR x_grad;
    GM_ADDR hc_weight_grad;
    GM_ADDR alpha_grad;
    GM_ADDR bias_post_grad;
    GM_ADDR gamma_grad;
    GM_ADDR workspace;
    TPipe *tPipeIn;
    MhcPreBackwardTilingData *tilingData;
};

struct OpRunInfo {
    uint64_t totalLength_ = 0; // 总长度 (batch * sequence 或 T)
    uint64_t nD = 0;           // n * D
    uint64_t fusionSize = 0;   // 2N + N^2
};

template <class P>
struct V0V1Buffers {
    LocalTensor<P> hPreGradBuf;
    LocalTensor<P> hPreBufS1;
    LocalTensor<P> hPreBufS2;
    LocalTensor<P> hPostBufS1;
    LocalTensor<P> hPostBufS2;
    LocalTensor<P> hCombBufS1;
    LocalTensor<P> hFusionBuf1;
    LocalTensor<P> gatherFusionBuf;
    LocalTensor<P> invRmsBuf;
    LocalTensor<P> calcTmpBuf;
    LocalTensor<uint8_t> brcbTmpBuf;
    uint32_t stepLength;
    uint32_t gatherLength;
    uint32_t preProcessUbOffset; // Offset after hPreGradBuf for PreProcessV0 temporary buffers
};

using aT_C0 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using bT_C0 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT_C0 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT_C0 = matmul::MatmulImpl<aT_C0, bT_C0, cT_C0>;

using aT_C1 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using bT_C1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT_C1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT_C1 = matmul::MatmulImpl<aT_C1, bT_C1, cT_C1>;

/**
 * @brief Workspace buffer管理结构体
 * 内存布局：
 * 1. h_mix_grad: [B, S, 2N + N*N] = [B * S * fusionSize]
 * 2. alpha_grad: [3, coreNum] = [3 * coreNum]
 * 3. bias_grad: [coreNum, 2N + N*N] = [coreNum * fusionSize]
 * 4. inv_rms_grad: [B, S] = [B * S]
 */
template <class P>
struct WorkspaceBuffer {
    uint64_t totalLength; // B * S
    uint64_t fusionSize;  // 2N + N*N
    uint64_t coreNum;     // 核心数量,此处为vecCoreNum

    // 各区域的起始偏移（以元素为单位）
    uint64_t hMixGradOffset;
    uint64_t alphaGradOffset;
    uint64_t biasGradOffset;
    uint64_t invRmsGradOffset;
    uint64_t xRsGradOffset;
    uint64_t xRsOffset; // V2输出，C1输入
    __aicore__ inline void Init(uint64_t bs, uint64_t fs, uint64_t cn)
    {
        totalLength = bs;
        fusionSize = fs;
        coreNum = cn;

        uint64_t alphaGradSize = ALPHA_GRAD_PADDING * coreNum; // 3个数padding
        uint64_t biasGradRows = coreNum;

        // 计算各区域偏移
        hMixGradOffset = 0;
        alphaGradOffset = totalLength * fusionSize;
        biasGradOffset = alphaGradOffset + alphaGradSize;
        invRmsGradOffset = biasGradOffset + biasGradRows * fusionSize;
        xRsGradOffset = invRmsGradOffset + totalLength;
        xRsOffset = xRsGradOffset + SINGLE_M * ND_BLOCK_SIZE * 2 * 32;
    }

    /**
     * @brief 获取h_mix_grad在指定bs索引位置的偏移
     * @param bsIndex batch*sequence的线性索引 (bsIndex = b * S + s)
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetHMixGradOffset(uint64_t bsIndex)
    {
        return hMixGradOffset + bsIndex * fusionSize;
    }

    /**
     * @brief 获取alpha_grad在指定(alphaIdx, coreId)位置的偏移
     * @param alphaIdx alpha索引 [0, 2]，对应3个alpha值
     * @param coreId 核心ID索引 [0, coreNum-1]
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetAlphaGradOffset(uint32_t coreId)
    {
        return alphaGradOffset + coreId * ALPHA_GRAD_PADDING;
    }

    /**
     * @brief 获取bias_grad在指定coreId的偏移
     * @param coreId 核心ID索引 [0, coreNum-1]
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetBiasGradOffset(uint32_t coreId)
    {
        return biasGradOffset + coreId * fusionSize;
    }

    /**
     * @brief 获取inv_rms_grad在指定bs索引位置的偏移
     * @param bsIndex batch*sequence的线性索引 (bsIndex = b * S + s)
     * @return uint64_t 偏移值（以元素为单位）
     */
    __aicore__ inline uint64_t GetInvRmsGradOffset(uint64_t bsIndex)
    {
        return invRmsGradOffset + bsIndex;
    }

    __aicore__ inline uint64_t GetXRsGradOffset(uint32_t coreId, uint32_t buffId)
    {
        return CeilAlign(xRsGradOffset, uint64_t(32)) + (coreId * 2 + buffId) * SINGLE_M * ND_BLOCK_SIZE;
    }

    __aicore__ inline uint64_t GetXRsOffset(uint32_t coreId, uint32_t buffId)
    {
        return CeilAlign(xRsOffset, uint64_t(32)) + (coreId * 2 + buffId) * SINGLE_M * ND_BLOCK_SIZE;
    }
};

template <class T, class P>
class MhcPreBackwardKernel {
public:
    __aicore__ inline MhcPreBackwardKernel(MT_C0 &matmul0, MT_C1 &matmul1) : mm0(matmul0), mm1(matmul1)
    {
    }
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void InitStage2();
    __aicore__ inline void Process();

    __aicore__ inline void PreProcessV0(LocalTensor<P> &hPreGradBuf, uint32_t dealBsNum, uint32_t ubOffset,
                                        uint32_t runBSStart, uint32_t runBSEnd);
    __aicore__ inline void ProcessV0Main();
    __aicore__ inline void AllocV0V1Buffers(uint32_t runBSStart, uint32_t runBSEnd, V0V1Buffers<P> &buffers);
    __aicore__ inline void ProcessV0(uint32_t runBSStart, V0V1Buffers<P> &buffers);
    __aicore__ inline void VFDoV0ProcessHPostGrad(__ubuf__ P *hPostIn, __ubuf__ P *PostGradIn, __ubuf__ P *hPostGradOut,
                                                uint32_t stepLen);
    __aicore__ inline void ProcessV1(uint32_t runBSStart, uint32_t runBSEnd, V0V1Buffers<P> &buffers,
                                     uint32_t vecRuntimesId, LocalTensor<P> &sumBuf);
    __aicore__ inline void VFDoV1ProcessInvRmsGrad(__ubuf__ P *h1GradIn, __ubuf__ P *hMixIn, uint16_t dealBSSize);
    __aicore__ inline void InitCube();
    __aicore__ inline void AICProcess(GlobalTensor<P> x, GlobalTensor<P> y, GlobalTensor<P> z, uint64_t m, uint64_t n,
                                      uint64_t k);
    __aicore__ inline void ProcessC0C1Pipeline();
    __aicore__ inline void ProcessV2Pipeline();
    __aicore__ inline void ProcessC0(uint32_t offsetND, uint32_t currentNDBlock, uint32_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);
    __aicore__ inline void ProcessC1(uint32_t offsetND, uint32_t currentNDBlock, uint32_t offsetBS,
                                     uint32_t currentBSBlock, uint32_t buffId);
    __aicore__ inline void ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint32_t bsStart, uint32_t bsEnd,
                                     LocalTensor<P> &sumBuf, uint32_t buffId);
    __aicore__ inline void ProcessV3();

    __aicore__ inline void VFDoPreProcessV0(__ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr, __ubuf__ T* hInGradB16InAddr, uint16_t curLenN, uint32_t lenD,
                                                                    uint32_t i, uint32_t j, uint32_t runBSStart);
    __aicore__ inline void VFDoV0HPreGrad(__ubuf__ P* hPreBufS1Addr, __ubuf__ P* hPreAddr, __ubuf__ P* hInGradAddr, uint32_t totalElem);
    
    template <bool hasGamma, bool isFirstChunk>
    __aicore__ inline void VFDoV2XCastAndMulGamma(__ubuf__ P* xRsFp32Addr, __ubuf__ P* gammaOutAddr, __ubuf__ T* bf16InputAddr, 
                                                  __ubuf__ P* gammaSrcAddr, __ubuf__ P* gammaBroadAddr, uint32_t currentChunkSize, uint32_t copySizeND);
    
    template <bool hasGamma>
    __aicore__ inline void VFDoV2GammaMulXRsGradMm(__ubuf__ P* xRsGradUbAddr, __ubuf__ P* xRsGradMmAddr, __ubuf__ P* gammaBroadAddr,
                                                   uint32_t currentChunkSize, uint32_t copySizeND);

private:
    MT_C0 &mm0;
    MT_C1 &mm1;
    uint32_t coreNum_;
    uint64_t totalLength_;
    uint64_t vecDealBSPeCore_;
    uint32_t vecCoreNum_;
    uint32_t dealStartBS_;
    uint32_t dealEndBS_;
    uint32_t usedVecCoreNum_;
    uint32_t D_;
    uint32_t N_;
    uint32_t fusionSize_;
    uint32_t onceDealMixBSNum_;
    float hcEps_;

    uint32_t blockIdx_;
    uint32_t nD_;
    uint32_t cubeCoreNum_; // AIC CoreNum
    uint32_t v1UsedCubeCoreNum_;
    uint32_t cubeDealnDPeCore_;
    uint32_t dealStartND_;
    uint32_t dealEndND_;


    GlobalTensor<T> xGm_;               // 输入 x
    GlobalTensor<P> phiGm_;             // 输入 phi
    GlobalTensor<P> alphaGm_;           // 输入 alpha
    GlobalTensor<P> gammaGm_;           // 输入 gamma
    GlobalTensor<T> hInGradGm_;         // 输入 h_in_grad
    GlobalTensor<P> hPostGradGm_;       // 输入 h_post_grad
    GlobalTensor<P> hCombBeforeGradGm_; // 输入 h_comb_before_grad
    GlobalTensor<P> invRmsGm_;          // 前向预计算 inv_rms
    GlobalTensor<P> mmResGm_;           // 前向预计算 h_mix
    GlobalTensor<P> hPreGm_;            // 前向预计算 h_pre
    GlobalTensor<P> hPostGm_;           // 前向输出 h_post
    GlobalTensor<P> workSpaceGm_;
    WorkspaceBuffer<P> workspaceBuf_; // Workspace buffer管理接口

    GlobalTensor<T> xGradGm_;        // 输出 x_grad
    GlobalTensor<P> hcWeightGradGm_; // 输出 hc_weight_grad
    GlobalTensor<P> alphaGradGm_;    // 输出 alpha_grad
    GlobalTensor<P> biasPostGradGm_; // 输出 bias_post_grad
    GlobalTensor<P> gammaGradGm_;    // 输出 gamma_grad

    /* V1 LocalTensor*/
    LocalTensor<P> gammaUb;      // gamma
    LocalTensor<T> xRsUb;        // x_rs
    LocalTensor<T> hInGradUb;    // h_in_grad
    LocalTensor<P> hPreUb;       // h_pre
    LocalTensor<P> invRmsGradUb; // inv_rms_grad
    LocalTensor<P> invRmsUb;     // inv_rms
    LocalTensor<P> xRsGradMmUb;  // x_rs_grad_mm
    LocalTensor<T> xGradUb;      // x_grad_bf16
    LocalTensor<P> xRsFp32Out;   // x_rs_fp32

    GlobalTensor<P> invRmsGradGm_; // V1输出 inv_rms_grad
    GlobalTensor<P> xRsGradMmGm_;  // C0输出 x_rs_grad_mm
    GlobalTensor<P> xRsFp32Gm_;    // V2输出 & C1输入 x_rs

    LocalTensor<uint32_t> gatherOffsetBuf_;
    LocalTensor<uint32_t> hPreGatherOffsetBuf_;
    LocalTensor<P> alphaBuf_;
    LocalTensor<P> alphaTmpBuf_;
    TPipe *pipe_;
    OpRunInfo runInfo_;
    const MhcPreBackwardTilingData *tiling_;
    TQue<QuePosition::VECIN, 0> bf16InQueue_;
    TQue<QuePosition::VECOUT, 1> bf16OutQueue_;
    TQue<QuePosition::VECIN, 1> fp32InQueue_;
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECOUT, 1> fp32OutQueue_;

    TBuf<TPosition::VECCALC> fp32TBuf_;

    uint32_t hPreMaxBufLen_;
    uint32_t hPostMaxBufLen_;
    uint32_t hCombBeforeGradBufLen_;
    uint32_t hPostOffset_;
    uint32_t hMixOffset_;
    uint32_t hFusionBufLen_;
    uint32_t hFusionOffset_;
    uint32_t globalUbOffset_;
    uint32_t vecDealChunk_;
    uint16_t eleNumPerVf_;
    DataCopyParams dataCopyParams_;
    DataCopyPadParams dataCopyPadParams_;
    bool alphaBufInitialized_;
    bool withGamma_;
    float scaleMean_;
};

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::Init(InitParams initParams)
{
    // 1. 绑定GlobalTensor
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));

    hInGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.h_in_grad));
    hPostGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post_grad));
    hCombBeforeGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_comb_before_grad));

    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
    mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.mm_res));
    hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));
    hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));

    xGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x_grad));
    hcWeightGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.hc_weight_grad));
    alphaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha_grad));
    biasPostGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.bias_post_grad));

    workSpaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));

    withGamma_ = (initParams.gamma != nullptr);

    if (withGamma_) {
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
        gammaGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma_grad));
    }

    // 2. 获取TilingData进行初始化
    tiling_ = initParams.tilingData;
    runInfo_.totalLength_ = tiling_->totalLength;
    nD_ = tiling_->nD;
    D_ = tiling_->D;
    N_ = tiling_->N;
    fusionSize_ = tiling_->fusionSize;
    coreNum_ = tiling_->coreNum;
    totalLength_ = tiling_->totalLength;
    hcEps_ = tiling_->hcEps;
    vecCoreNum_ = tiling_->vecCoreNum;
    scaleMean_ = 1.0f / nD_;
    eleNumPerVf_ = GetVRegSize() / sizeof(P);

    // 初始化WorkspaceBuffer接口（需要在coreNum_初始化之后）
    workspaceBuf_.Init(totalLength_, fusionSize_, vecCoreNum_);
    blockIdx_ = GetBlockIdx();

    // 3. 申请UB
    pipe_ = initParams.tPipeIn;
    vecDealChunk_ = N_ > 6 ? 32 : 64; // 当N较大时为tempbuff保留空间，取32
    hPreMaxBufLen_ = vecDealChunk_ * N_;
    hPostMaxBufLen_ = hPreMaxBufLen_;
    hMixOffset_ = (hPreMaxBufLen_ + hPostMaxBufLen_) * sizeof(P);
    hPostOffset_ = hPreMaxBufLen_ * sizeof(P);
    hCombBeforeGradBufLen_ = vecDealChunk_ * N_ * N_;
    hFusionBufLen_ = hPreMaxBufLen_ + hPostMaxBufLen_ + hCombBeforeGradBufLen_;
    onceDealMixBSNum_ = INOUT_QUEUE_SIZE / (fusionSize_ * sizeof(P));

    pipe_->InitBuffer(gradInQueue_, 1, INOUT_QUEUE_SIZE);

    if ASCEND_IS_AIV {
        pipe_->InitBuffer(bf16InQueue_, 1, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(bf16OutQueue_, 1, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(fp32InQueue_, 1, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(fp32OutQueue_, 1, INOUT_QUEUE_SIZE);
        pipe_->InitBuffer(fp32TBuf_, FP32_BUF_SIZE);


        gatherOffsetBuf_ = fp32TBuf_.GetWithOffset<uint32_t>(hFusionBufLen_, 0);
        globalUbOffset_ = hFusionBufLen_ * sizeof(uint32_t);
        alphaBuf_ = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, globalUbOffset_);
        globalUbOffset_ += hFusionBufLen_ * sizeof(P);
        alphaTmpBuf_ = fp32TBuf_.GetWithOffset<P>(fusionSize_, globalUbOffset_);
        globalUbOffset_ += fusionSize_ * sizeof(P);

        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        for (uint32_t i = 0; i < vecDealChunk_; i++) {
            for (uint32_t j = 0; j < N_; j++) {
                gatherOffsetBuf_.SetValue(offset1++, (i * N_ + j) * sizeof(P));
                if (i == 0) {
                    alphaTmpBuf_.SetValue(offset2++, alphaGm_.GetValue(0));
                }
            }

            for (uint32_t j = 0; j < N_; j++) {
                gatherOffsetBuf_.SetValue(offset1++, (i * N_ + j) * sizeof(P) + hPostOffset_);
                if (i == 0) {
                    alphaTmpBuf_.SetValue(offset2++, alphaGm_.GetValue(1));
                }
            }
            for (uint32_t j = 0; j < (N_ * N_); j++) {
                gatherOffsetBuf_.SetValue(offset1++, (i * N_ * N_ + j) * sizeof(P) + hMixOffset_);
                if (i == 0) {
                    alphaTmpBuf_.SetValue(offset2++, alphaGm_.GetValue(2));
                }
            }
        }

        vecDealBSPeCore_ = totalLength_ / vecCoreNum_;
        if (vecDealBSPeCore_ == 0) {
            vecDealBSPeCore_ = totalLength_;
            usedVecCoreNum_ = 1;
        } else {
            vecDealBSPeCore_ = CeilAlign(vecDealBSPeCore_, uint64_t(16));
            usedVecCoreNum_ = CeilDiv(totalLength_, vecDealBSPeCore_) > vecCoreNum_ ?
                                  vecCoreNum_ :
                                  CeilDiv(totalLength_, vecDealBSPeCore_);
        }

        dealStartBS_ = vecDealBSPeCore_ * blockIdx_;
        dealEndBS_ = dealStartBS_ + vecDealBSPeCore_;
        if (dealEndBS_ > totalLength_ || blockIdx_ == usedVecCoreNum_ - 1) {
            dealEndBS_ = totalLength_;
        }
    }
    InitStage2();
    // 初始化DataCopyParams和DataCopyPadParams
    dataCopyParams_.blockCount = 1;
    dataCopyParams_.srcStride = 0;
    dataCopyParams_.dstStride = 0;
    dataCopyPadParams_.isPad = false;
    alphaBufInitialized_ = false;
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::InitStage2()
{
    // 4. 初始化 V2 & C0-1 的分核
    v1UsedCubeCoreNum_ = 0;
    cubeDealnDPeCore_ = nD_ / GetBlockNum();
    if (cubeDealnDPeCore_ == 0) {
        cubeDealnDPeCore_ = nD_;
        v1UsedCubeCoreNum_ = 1;
    } else {
        cubeDealnDPeCore_ = CeilAlign(cubeDealnDPeCore_, uint32_t(128));
        v1UsedCubeCoreNum_ = CeilDiv(nD_, cubeDealnDPeCore_);
    }

    if ASCEND_IS_AIC {
        dealStartND_ = cubeDealnDPeCore_ * blockIdx_;

        dealEndND_ = dealStartND_ + cubeDealnDPeCore_;
        if (dealEndND_ > nD_) {
            dealEndND_ = nD_;
        }
    }

    if ASCEND_IS_AIV {
        // 1C2V: 初始化 V2 & C0-1 的分核
        dealStartND_ = cubeDealnDPeCore_ * uint32_t(blockIdx_ / 2);
        dealEndND_ = dealStartND_ + cubeDealnDPeCore_;
        if (dealEndND_ > nD_) {
            dealEndND_ = nD_;
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::Process()
{
    if ASCEND_IS_AIV {
        ProcessV0Main();
    }

    SyncAll<false>();

    if ASCEND_IS_AIC {
        // 执行矩阵乘法 C0 -> C1
        ProcessC0C1Pipeline();
    }

    if ASCEND_IS_AIV {
        // 执行向量计算V2
        ProcessV2Pipeline();
    }

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == (GetBlockNum() - 1)) {
            ProcessV3();
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoPreProcessV0(__ubuf__ P* hPreGradAddr, __ubuf__ T* xB16InAddr, __ubuf__ T* hInGradB16InAddr, uint16_t curLenN, uint32_t lenD,
                                                                    uint32_t i, uint32_t j, uint32_t runBSStart) 
{
    uint16_t eleNumPerVf = 64;
    uint16_t dLoopCnt = (lenD + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> sumReg;
        uint32_t offset = (i - runBSStart) * N_ + j;
        for (uint16_t offsetN = 0; offsetN < curLenN; offsetN++) { // x [curLenN, lenD],  hinGrad [1, lenD]
            MicroAPI::Duplicate(sumReg, 0.0f);
            uint32_t curLenD = lenD;
            // x[1, lenD]  hinGrad [1, lenD]   res[1, lenD]
            MicroAPI::RegTensor<T> xInB16Reg, hInGradB16InReg;
            MicroAPI::RegTensor<P> xFp32Reg, hInGradFp32Reg, mulReg, tmpSumPerVfReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < dLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);

                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInB16Reg, xB16InAddr + offsetN * lenD + vfBlockIdx * eleNumPerVf);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(hInGradB16InReg, hInGradB16InAddr + vfBlockIdx * eleNumPerVf);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(hInGradFp32Reg, hInGradB16InReg, mask);

                MicroAPI::Mul(mulReg, xFp32Reg, hInGradFp32Reg, mask); // mulReg[1,lenD] 0-63
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(tmpSumPerVfReg, mulReg, mask); // 每个vfBlockIdx循环的临时求和，每64个元素的和
                MicroAPI::Add(sumReg, sumReg, tmpSumPerVfReg, mask); // D方向上的reduceSum
            }
                // uint32_t bufIdx = (i - runBSStart) * N_ + (j + offsetN);
                uint32_t bufIdx = offset + offsetN;
                MicroAPI::Store(hPreGradAddr + bufIdx, sumReg, 1);
            }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::PreProcessV0(LocalTensor<P> &hPreGradBuf, uint32_t dealBsNum,
                                                                uint32_t ubOffset, uint32_t runBSStart,
                                                                uint32_t runBSEnd)
{
    uint32_t maxDLen = 10240;
    uint32_t queueCopySize = INOUT_QUEUE_SIZE / sizeof(T);
    uint64_t copySizeD = D_ <= maxDLen ? D_ : maxDLen;
    uint32_t lenN = queueCopySize / copySizeD > N_ ? N_ : queueCopySize / copySizeD;
    AscendC::LocalTensor<T> bf16InputBuf;

    AscendC::LocalTensor<P> xFp32Buf = fp32TBuf_.GetWithOffset<P>(D_ * lenN, ubOffset);
    ubOffset = CeilAlign(uint32_t(ubOffset + D_ * lenN * sizeof(P)), uint32_t(32));
    AscendC::LocalTensor<P> hInGradBuf = fp32TBuf_.GetWithOffset<P>(D_, ubOffset);
    ubOffset = CeilAlign(uint32_t(ubOffset + D_ * sizeof(P)), uint32_t(32));
    AscendC::LocalTensor<P> tmpBuf = fp32TBuf_.GetWithOffset<P>(D_ / 4, ubOffset);

    for (uint32_t i = runBSStart; i < runBSEnd; i++) {
        uint32_t inputOffset = (i * N_) * copySizeD;
        uint64_t ubOffset = 0;
        for (uint32_t j = 0; j < N_; j += lenN) {
            uint32_t curLenN = j + lenN > N_ ? N_ - j : lenN;
            dataCopyParams_.blockLen = copySizeD * sizeof(T);
            dataCopyParams_.blockCount = curLenN;
            dataCopyParams_.srcStride = 0;
            dataCopyParams_.dstStride = 0;
            dataCopyPadParams_.isPad = false;

            bf16InQueue_.AllocTensor<T>(bf16InputBuf);
            DataCopyPad(bf16InputBuf, xGm_[inputOffset], dataCopyParams_, dataCopyPadParams_);
            bf16InQueue_.EnQue(bf16InputBuf);
            bf16InQueue_.DeQue<T>(bf16InputBuf);
            Cast(xFp32Buf[ubOffset], bf16InputBuf, RoundMode::CAST_NONE, copySizeD * curLenN);
            bf16InQueue_.FreeTensor(bf16InputBuf);

            dataCopyParams_.blockCount = 1;
            AscendC::LocalTensor<T> hInputBuf = fp32InQueue_.AllocTensor<T>();
            DataCopyPad(hInputBuf, hInGradGm_[i * copySizeD], dataCopyParams_, dataCopyPadParams_);
            fp32InQueue_.EnQue(hInputBuf);

            LocalTensor<T> hInGradInputBuf = fp32InQueue_.DeQue<T>();
           
            __ubuf__ T* xB16InAddr = (__ubuf__ T*)bf16InputBuf.GetPhyAddr();
            __ubuf__ T* hInGradB16InAddr = (__ubuf__ T*)hInGradInputBuf.GetPhyAddr();
            __ubuf__ P* hPreGradAddr = (__ubuf__ P*)hPreGradBuf.GetPhyAddr();
            VFDoPreProcessV0(hPreGradAddr, xB16InAddr, hInGradB16InAddr, curLenN, copySizeD, i, j, runBSStart);

            inputOffset += copySizeD * curLenN;
            bf16InQueue_.FreeTensor(bf16InputBuf);
            fp32InQueue_.FreeTensor(hInGradInputBuf);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0Main()
{
    if (blockIdx_ >= usedVecCoreNum_) {
        return;
    }

    uint32_t dealBsNum = dealEndBS_ - dealStartBS_; // 不能加一

    uint32_t vecRunTimes = CeilDiv(dealBsNum, vecDealChunk_);

    float alphaPre = alphaGm_.GetValue(0);

    // 临时变量存储alpha. 在V0V1buffer前面加塞了(N^2 + 2N + 24) * 4的空间,24*4用于Padding搬出
    LocalTensor<P> sumBuf = fp32TBuf_.GetWithOffset<P>(fusionSize_, globalUbOffset_);
    globalUbOffset_ += CeilAlign(uint32_t(fusionSize_ * sizeof(P)), uint32_t(32));

    for (uint32_t cIdx = 0; cIdx < vecRunTimes; cIdx++) {
        uint32_t runBSStart = cIdx * vecDealChunk_ + dealStartBS_;
        uint32_t runBSEnd = runBSStart + vecDealChunk_;

        if (runBSEnd > dealEndBS_) {
            runBSEnd = dealEndBS_;
        }

        uint32_t currentDealBsNum = runBSEnd - runBSStart;

        V0V1Buffers<P> buffers;
        AllocV0V1Buffers(runBSStart, runBSEnd, buffers);

        PreProcessV0(buffers.hPreGradBuf, currentDealBsNum, buffers.preProcessUbOffset, runBSStart, runBSEnd);

        ProcessV0(runBSStart, buffers);

        ProcessV1(runBSStart, runBSEnd, buffers, cIdx, sumBuf);
    }

    AscendC::LocalTensor<P> alphaOutBuf = bf16OutQueue_.AllocTensor<P>();
    uint32_t coreId = GetBlockIdx();

    uint32_t srcShape1[] = {1, N_};
    uint32_t srcShape2[] = {1, 2 * N_};
    uint32_t srcShape3[] = {1, fusionSize_};

    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf, sumBuf, srcShape1, true);
    PipeBarrier<PIPE_V>();
    Muls(sumBuf, sumBuf, 0.0f, N_);
    PipeBarrier<PIPE_V>();

    ReduceSum<float, Pattern::Reduce::AR, false>(alphaOutBuf[8], sumBuf, srcShape2, true);
    PipeBarrier<PIPE_V>();
    Muls(sumBuf, sumBuf, 0.0f, N_ * 2);
    PipeBarrier<PIPE_V>();

    ReduceSum<float, Pattern::Reduce::AR, true>(alphaOutBuf[16], sumBuf, srcShape3, true);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

    bf16OutQueue_.EnQue(alphaOutBuf);
    alphaOutBuf = bf16OutQueue_.DeQue<P>();

    uint32_t offset = workspaceBuf_.GetAlphaGradOffset(coreId);

    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = ALPHA_GRAD_PADDING * sizeof(P);
    bsCopyParams.srcStride = 0; // 每个BS元素之间的stride
    bsCopyParams.dstStride = 0; // 目标buffer连续存储
    DataCopyPad(workSpaceGm_[offset], alphaOutBuf, bsCopyParams);

    bf16OutQueue_.FreeTensor(alphaOutBuf);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::AllocV0V1Buffers(uint32_t runBSStart, uint32_t runBSEnd,
                                                                    V0V1Buffers<P> &buffers)
{
    buffers.stepLength = (runBSEnd - runBSStart) * N_;
    uint32_t hMixLength = (runBSEnd - runBSStart) * N_ * N_;
    buffers.gatherLength = buffers.stepLength * 2 + hMixLength;
    uint32_t stepOffset1 = CeilAlign(uint32_t(buffers.stepLength * sizeof(P)), uint32_t(32));

    // Allocate hPreGradBuf first
    uint32_t hpreGradTotlenLen = CeilAlign(buffers.stepLength, uint32_t(32));
    buffers.hPreGradBuf = fp32TBuf_.GetWithOffset<P>(hpreGradTotlenLen, globalUbOffset_);
    buffers.preProcessUbOffset = globalUbOffset_ + hpreGradTotlenLen * sizeof(P);

    uint32_t ubOffset = buffers.preProcessUbOffset;
    uint32_t fusionOffset = ubOffset;

    // V0 buffers
    buffers.hPreBufS1 = fp32TBuf_.GetWithOffset<P>(hPreMaxBufLen_, ubOffset);
    buffers.hPostBufS1 = fp32TBuf_.GetWithOffset<P>(hPostMaxBufLen_, ubOffset + hPostOffset_);
    buffers.hCombBufS1 = fp32TBuf_.GetWithOffset<P>(hCombBeforeGradBufLen_, ubOffset + hMixOffset_);
    buffers.hFusionBuf1 = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, fusionOffset);

    ubOffset += hFusionBufLen_ * sizeof(P);
    fusionOffset = ubOffset;

    buffers.hPreBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    ubOffset += stepOffset1;
    buffers.hPostBufS2 = fp32TBuf_.GetWithOffset<P>(buffers.stepLength, ubOffset);
    buffers.gatherFusionBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, fusionOffset);

    // V1 buffers
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.invRmsBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, ubOffset);
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.calcTmpBuf = fp32TBuf_.GetWithOffset<P>(hFusionBufLen_, ubOffset);
    ubOffset += hFusionBufLen_ * sizeof(P);
    buffers.brcbTmpBuf = fp32TBuf_.GetWithOffset<uint8_t>(FP32_BUF_SIZE - ubOffset, ubOffset);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV0HPreGrad(__ubuf__ P* hPreBufS1Addr, __ubuf__ P* hPreAddr, __ubuf__ P* hInGradAddr, uint32_t totalElem)
{
    uint32_t eleNumPerVf = 64;
    uint16_t loopCnt = Ceil(totalElem, eleNumPerVf);
    uint32_t curElemCnt = totalElem;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> hPreReg, hInGradReg;
        MicroAPI::RegTensor<P> s1Reg, s2Reg, mulReg, resReg;
        MicroAPI::RegTensor<P> oneReg;
        // MicroAPI::RegTensor<P> s1NegReg;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
            MicroAPI::Duplicate(oneReg, 1.0f, mask);

            MicroAPI::LoadAlign(hPreReg, hPreAddr + vfBlockIdx * eleNumPerVf);
            MicroAPI::LoadAlign(hInGradReg, hInGradAddr + vfBlockIdx * eleNumPerVf);
            MicroAPI::Adds(s1Reg, hPreReg, -hcEps_, mask); // s1 = hPre - hcEps 
            MicroAPI::Sub(s2Reg, oneReg, s1Reg, mask);    // s2 = 1 - s1
            // MicroAPI::Muls(s1NegReg, s1Reg, (-1.0f), mask);    // s1 = -s1
            // MicroAPI::Adds(s2Reg, s1NegReg, (1.0f), mask);    // s2 = - s1 + 1
            MicroAPI::Mul(mulReg, s1Reg, s2Reg, mask);    // mulReg = s1 * s2
            MicroAPI::Mul(resReg, mulReg, hInGradReg, mask);    // resReg = mulReg * hInGradReg

            MicroAPI::StoreAlign(hPreBufS1Addr + vfBlockIdx * eleNumPerVf, resReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV0(uint32_t runBSStart, V0V1Buffers<P> &buffers)
{
    AscendC::LocalTensor<P> fp32InputBuf = fp32InQueue_.AllocTensor<P>();

    auto h1GradBuf = buffers.calcTmpBuf;
    dataCopyParams_.blockLen = buffers.stepLength * sizeof(P);

    PipeBarrier<PIPE_MTE2>();
    DataCopyPad(fp32InputBuf, hPreGm_[runBSStart * N_], dataCopyParams_, dataCopyPadParams_);
    fp32InQueue_.EnQue(fp32InputBuf);
    LocalTensor<P> hPre = fp32InQueue_.DeQue<P>();

    __ubuf__ P* hPreAddr = (__ubuf__ P*)hPre.GetPhyAddr();
    __ubuf__ P* hInGradAddr = (__ubuf__ P*)buffers.hPreGradBuf.GetPhyAddr();
    __ubuf__ P* hPreBufS1Addr = (__ubuf__ P*)buffers.hPreBufS1.GetPhyAddr();
    VFDoV0HPreGrad(hPreBufS1Addr, hPreAddr, hInGradAddr, buffers.stepLength);

    // fp32InputBuf = fp32InQueue_.AllocTensor<P>();
    // PipeBarrier<PIPE_MTE2>();
    // DataCopyPad(fp32InputBuf, hPostGm_[runBSStart * N_], dataCopyParams_, dataCopyPadParams_);
    // fp32InQueue_.EnQue(fp32InputBuf);
    // LocalTensor<P> hPost = fp32InQueue_.DeQue<P>();

    // Muls(buffers.hPostBufS1, hPost, 0.5f, buffers.stepLength);
    // PipeBarrier<PIPE_V>();

    // // s2_post = 1 - s1_post
    // Muls(buffers.hPostBufS2, buffers.hPostBufS1, -1.0f, buffers.stepLength);
    // PipeBarrier<PIPE_V>();
    // Adds(buffers.hPostBufS2, buffers.hPostBufS2, 1.0f, buffers.stepLength);
    // PipeBarrier<PIPE_V>();

    // // s3_post = s1_post * s2_post
    // Mul(buffers.hPostBufS1, hPost, buffers.hPostBufS2, buffers.stepLength);
    // PipeBarrier<PIPE_V>();
    // fp32InQueue_.FreeTensor(hPost);

    // fp32InputBuf = fp32InQueue_.AllocTensor<P>();
    // // PipeBarrier<PIPE_MTE2>();
    // DataCopyPad(fp32InputBuf, hPostGradGm_[runBSStart * N_], dataCopyParams_, dataCopyPadParams_);
    // fp32InQueue_.EnQue(fp32InputBuf);
    // LocalTensor<P> hPostGrad = fp32InQueue_.DeQue<P>();

    // Mul(buffers.hPostBufS1, buffers.hPostBufS1, hPostGrad, buffers.stepLength);
    // PipeBarrier<PIPE_V>();
    // fp32InQueue_.FreeTensor(hPostGrad);

    // V0-1
    fp32InputBuf = fp32InQueue_.AllocTensor<P>();
    // PipeBarrier<PIPE_MTE2>(); // ？这为什么需要插PIPE_MTE3同步
    DataCopyPad(fp32InputBuf, hPostGm_[runBSStart * N_], dataCopyParams_, dataCopyPadParams_);
    fp32InQueue_.EnQue(fp32InputBuf);
    AscendC::LocalTensor<P> fp32Buf = gradInQueue_.AllocTensor<P>();
    DataCopyPad(fp32Buf, hPostGradGm_[runBSStart * N_], dataCopyParams_, dataCopyPadParams_);
    gradInQueue_.EnQue(fp32Buf);

    LocalTensor<P> hPost = fp32InQueue_.DeQue<P>();
    LocalTensor<P> hPostGrad = gradInQueue_.DeQue<P>();
    VFDoV0ProcessHPostGrad((__ubuf__ P *)hPost.GetPhyAddr(), (__ubuf__ P *)hPostGrad.GetPhyAddr(),
                        (__ubuf__ P *)buffers.hPostBufS1.GetPhyAddr(), buffers.stepLength);
    fp32InQueue_.FreeTensor(hPost);
    gradInQueue_.FreeTensor(hPostGrad);

    fp32InputBuf = fp32InQueue_.AllocTensor<P>();
    PipeBarrier<PIPE_MTE2>();
    dataCopyParams_.blockLen = hCombBeforeGradBufLen_ * sizeof(P);
    DataCopyPad(fp32InputBuf, hCombBeforeGradGm_[runBSStart * N_ * N_], dataCopyParams_, dataCopyPadParams_);
    fp32InQueue_.EnQue(fp32InputBuf);
    LocalTensor<P> hCombBeforeGradBuf = fp32InQueue_.DeQue<P>();

    Muls(buffers.hCombBufS1, hCombBeforeGradBuf, 1.0f, hCombBeforeGradBufLen_);
    PipeBarrier<PIPE_V>();
    fp32InQueue_.FreeTensor(hCombBeforeGradBuf);

    if (!alphaBufInitialized_) {
        uint32_t xRowSumBroadCastDst[2] = {vecDealChunk_, fusionSize_};
        uint32_t xRowSumBroadCastSrc[2] = {1, fusionSize_};
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        BroadCast<float, 2, 0>(alphaBuf_, alphaTmpBuf_, xRowSumBroadCastDst, xRowSumBroadCastSrc, buffers.brcbTmpBuf);
        PipeBarrier<PIPE_V>();
        alphaBufInitialized_ = true;
    }

    Gather(buffers.gatherFusionBuf, buffers.hFusionBuf1, gatherOffsetBuf_, uint32_t(0), buffers.gatherLength);
    PipeBarrier<PIPE_V>();

    Mul(h1GradBuf, buffers.gatherFusionBuf, alphaBuf_, buffers.gatherLength);
    PipeBarrier<PIPE_V>();
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV0ProcessHPostGrad(__ubuf__ P *hPostIn, __ubuf__ P *hPostGradIn,
    __ubuf__ P *hPostGradOut, uint32_t stepLen)
{
    uint16_t loopCnt = CeilDiv(stepLen, uint32_t(eleNumPerVf_));
    uint32_t curLen = stepLen;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg mask;
        MicroAPI::RegTensor<P> hPostGradReg, tmpReg;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < loopCnt; vfBlockIdx++) {
            mask = MicroAPI::UpdateMask<P>(curLen);
            MicroAPI::LoadAlign(hPostGradReg, hPostIn + vfBlockIdx * eleNumPerVf_);
            // MicroAPI::Load<P>(hPostGradReg, hPostIn + vfBlockIdx * eleNumPerVf_);
            MicroAPI::Muls(tmpReg, hPostGradReg, -0.5f, mask);
            MicroAPI::Adds(tmpReg, tmpReg, 1.0f, mask);
            MicroAPI::Mul(hPostGradReg, hPostGradReg, tmpReg, mask);
            MicroAPI::LoadAlign(tmpReg, hPostGradIn + vfBlockIdx * eleNumPerVf_);
            // MicroAPI::Load<P>(tmpReg, hPostGradIn + vfBlockIdx * eleNumPerVf_);
            MicroAPI::Mul(hPostGradReg, hPostGradReg, tmpReg, mask);
            MicroAPI::StoreAlign(hPostGradOut + vfBlockIdx * eleNumPerVf_, hPostGradReg, mask);
            // MicroAPI::Store<P>(hPostGradOut + vfBlockIdx * eleNumPerVf_, hPostGradReg);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV1(uint32_t runBSStart, uint32_t runBSEnd,
                                                             V0V1Buffers<P> &buffers, uint32_t vecRuntimesId,
                                                             LocalTensor<P> &sumBuf)
{
    uint32_t shapeBiasGrad[] = {runBSEnd - runBSStart, fusionSize_};
    constexpr bool isReuse = false;

    LocalTensor<P> hMixBuf;
    AscendC::LocalTensor<P> fp32OutBuf = bf16OutQueue_.AllocTensor<P>();
    ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(fp32OutBuf, buffers.gatherFusionBuf, buffers.brcbTmpBuf,
                                                            shapeBiasGrad, true);
    bf16OutQueue_.EnQue(fp32OutBuf);
    LocalTensor<P> BiasOutBuf = bf16OutQueue_.DeQue<P>();

    uint32_t coreId = GetBlockIdx();
    dataCopyParams_.blockLen = fusionSize_ * sizeof(P);
    if (vecRuntimesId != 0) {
        SetAtomicAdd<float>();
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetBiasGradOffset(coreId)], BiasOutBuf, dataCopyParams_);
        SetAtomicNone();
    } else {
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetBiasGradOffset(coreId)], BiasOutBuf, dataCopyParams_);
    }
    PipeBarrier<PIPE_V>();
    bf16OutQueue_.FreeTensor(BiasOutBuf);

    auto h1GradBuf = buffers.calcTmpBuf;
    int64_t remainDealBsSize = (runBSEnd - runBSStart);
    uint32_t bsOffset = 0;
    while (remainDealBsSize > 0) {
        fp32OutBuf = bf16OutQueue_.AllocTensor<P>();
        uint32_t dealBSSize = (remainDealBsSize > onceDealMixBSNum_) ? onceDealMixBSNum_ : remainDealBsSize;
        dataCopyParams_.blockLen = dealBSSize * sizeof(P);
        AscendC::LocalTensor<P> fp32InputBuf = fp32InQueue_.AllocTensor<P>();
        DataCopyPad(fp32InputBuf, invRmsGm_[(runBSStart + bsOffset)], dataCopyParams_, dataCopyPadParams_);
        fp32InQueue_.EnQue(fp32InputBuf);
        LocalTensor<P> invRmsBufLocal = fp32InQueue_.DeQue<P>();

        const uint32_t xRowSumBroadCastDst[2] = {dealBSSize, fusionSize_};
        const uint32_t xRowSumBroadCastSrc[2] = {dealBSSize, 1};
        BroadCast<float, 2, 1>(buffers.invRmsBuf, invRmsBufLocal, xRowSumBroadCastDst, xRowSumBroadCastSrc,
                               buffers.brcbTmpBuf);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(invRmsBufLocal);

        PipeBarrier<PIPE_MTE3>();
        Mul(fp32OutBuf, buffers.invRmsBuf, h1GradBuf[bsOffset * fusionSize_], dealBSSize * fusionSize_);
        PipeBarrier<PIPE_V>();

        bf16OutQueue_.EnQue(fp32OutBuf);
        LocalTensor<P> hMixGradOutBuf = bf16OutQueue_.DeQue<P>();

        dataCopyParams_.blockLen = dealBSSize * fusionSize_ * sizeof(P);
        DataCopyPad(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(runBSStart + bsOffset)], hMixGradOutBuf,
                    dataCopyParams_);
        PipeBarrier<PIPE_V>();

        bf16InQueue_.AllocTensor<P>(hMixBuf);
        DataCopyPad(hMixBuf, mmResGm_[(runBSStart + bsOffset) * fusionSize_], dataCopyParams_, dataCopyPadParams_);
        bf16InQueue_.EnQue(hMixBuf);
        bf16InQueue_.DeQue<P>(hMixBuf);

        // hmix * h_1_grad, prepare for inv rms grad
        // Mul(h1GradBuf[bsOffset * fusionSize_], h1GradBuf[bsOffset * fusionSize_], hMixBuf, dealBSSize * fusionSize_);
        VFDoV1ProcessInvRmsGrad((__ubuf__ P *)h1GradBuf[bsOffset * fusionSize_].GetPhyAddr(),
                                (__ubuf__ P *)hMixBuf.GetPhyAddr(), dealBSSize);
        PipeBarrier<PIPE_V>();

        // TODO
        Mul(buffers.hFusionBuf1[bsOffset * fusionSize_], hMixBuf, buffers.invRmsBuf, dealBSSize * fusionSize_);
        PipeBarrier<PIPE_V>();
        bf16InQueue_.FreeTensor(hMixBuf);

        // alpha grad pre
        Mul(buffers.hFusionBuf1[bsOffset * fusionSize_], buffers.hFusionBuf1[bsOffset * fusionSize_],
            buffers.gatherFusionBuf[bsOffset * fusionSize_], dealBSSize * fusionSize_);
        PipeBarrier<PIPE_V>();

        remainDealBsSize -= dealBSSize;
        bsOffset += dealBSSize;
        bf16OutQueue_.FreeTensor(hMixGradOutBuf);
    }
    fp32OutBuf = bf16OutQueue_.AllocTensor<P>();

    uint32_t shape[] = {(runBSEnd - runBSStart), fusionSize_};
    // inv rms grad
    // TODO 暂时存储在invRmsBuf中，后续直接拼接好后搬运出
    ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(fp32OutBuf, h1GradBuf, buffers.brcbTmpBuf, shape, true);
    PipeBarrier<PIPE_V>();
    bf16OutQueue_.EnQue(fp32OutBuf);
    LocalTensor<P> invRmsOutBuf = bf16OutQueue_.DeQue<P>();

    dataCopyParams_.blockLen = (runBSEnd - runBSStart) * sizeof(P);
    DataCopyPad(workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(runBSStart)], invRmsOutBuf, dataCopyParams_);
    PipeBarrier<PIPE_V>();
    bf16OutQueue_.FreeTensor(invRmsOutBuf);

    // alpha grad (inv rms 梯度计算完成，复用h1GradBuf)
    ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(h1GradBuf, buffers.hFusionBuf1, buffers.brcbTmpBuf, shape,
                                                            true);
    PipeBarrier<PIPE_V>();

    if (vecRuntimesId == 0) {
        Adds(sumBuf, h1GradBuf, 0.0f, fusionSize_);
    } else {
        Add(sumBuf, h1GradBuf, sumBuf, fusionSize_);
    }
    PipeBarrier<PIPE_V>();
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV1ProcessInvRmsGrad(__ubuf__ P *h1GradIn, __ubuf__ P *hMixIn,
                                                                           uint16_t dealBSSize)
{
    uint32_t totalElem = dealBSSize * fusionSize_;
    uint32_t regCapacityFP32 = 64; // 256B / sizeof(P)
    uint16_t nLoopCnt = Ceil(totalElem, regCapacityFP32);
    uint32_t curElemCnt = totalElem;

    __VEC_SCOPE__
    {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
            uint32_t elemOffset = vfBlockIdx * regCapacityFP32;
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
            MicroAPI::RegTensor<P> h1GradBuf, hMixReg;
            MicroAPI::RegTensor<P> hMulReg;

            MicroAPI::LoadAlign(h1GradBuf, h1GradIn + elemOffset);
            MicroAPI::LoadAlign(hMixReg, hMixIn + elemOffset);

            MicroAPI::Mul(hMulReg, h1GradBuf, hMixReg, mask);

            MicroAPI::StoreAlign(h1GradIn + elemOffset, hMulReg, mask);
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC0C1Pipeline()
{
    if (GetBlockIdx() >= v1UsedCubeCoreNum_) {
        return;
    }

    uint32_t currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - dealStartND_);
    uint32_t currentBSBlock = Min(SINGLE_M, uint32_t(totalLength_));
    uint32_t buffId = 0;
    // 预执行第一个C0
    ProcessC0(dealStartND_, currentNDBlock, 0, currentBSBlock, buffId);

    for (uint32_t offsetND = dealStartND_; offsetND < dealEndND_; offsetND += ND_BLOCK_SIZE) {
        currentNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - offsetND);

        for (uint32_t offsetBS = 0; offsetBS < totalLength_; offsetBS += SINGLE_M) {
            currentBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - offsetBS));

            // 预计算下一个C0
            uint32_t nextOffsetBS = offsetBS + SINGLE_M;
            if (nextOffsetBS < totalLength_) { // BS还有未计算
                uint32_t nextBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - nextOffsetBS));
                ProcessC0(offsetND, currentNDBlock, nextOffsetBS, nextBSBlock, (buffId + 1));
            } else {
                uint32_t nextOffsetND = offsetND + ND_BLOCK_SIZE;
                if (nextOffsetND < dealEndND_) { // ND 还有未计算
                    uint32_t nextNDBlock = Min(ND_BLOCK_SIZE, dealEndND_ - nextOffsetND);
                    nextOffsetBS = 0;
                    uint32_t nextBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - nextOffsetBS));
                    ProcessC0(nextOffsetND, nextNDBlock, nextOffsetBS, nextBSBlock, (buffId + 1));
                }
            }

            ProcessC1(offsetND, currentNDBlock, offsetBS, currentBSBlock, buffId);
            buffId++;
        }
    }
}


template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC0(uint32_t offsetND, uint32_t currentNDBlock,
                                                             uint32_t offsetBS, uint32_t currentBSBlock,
                                                             uint32_t buffId)
{
    buffId = buffId % 2;
    // 设置矩阵形状
    mm0.SetOrgShape(totalLength_, nD_, fusionSize_);
    mm0.SetSingleShape(currentBSBlock, currentNDBlock, fusionSize_);
    // h_mix_grad @ phi
    mm0.SetTensorA(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)]); // [BS, N^2+2N]
    mm0.SetTensorB(phiGm_[offsetND], false);                                 // [N^2+2N, ND]
    // 计算
    uint64_t writeOffset = workspaceBuf_.GetXRsGradOffset(GetBlockIdx(), buffId);
    while (mm0.Iterate()) {
        mm0.GetTensorC(workSpaceGm_[writeOffset], 0, true); // 连续写
        writeOffset += 256 * 128;                           // baseM * baseN
    }
    mm0.End();

    // 通知V2计算
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x7);
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessC1(uint32_t offsetND, uint32_t currentNDBlock,
                                                             uint32_t offsetBS, uint32_t currentBSBlock,
                                                             uint32_t buffId)
{
    AscendC::CrossCoreWaitFlag(0x8);
    // 1C2V 对应操作两块 workspaceGm_
    buffId = buffId % 2;
    // 设置矩阵形状
    uint64_t offsetXs = workspaceBuf_.GetXRsOffset(GetBlockIdx(), buffId);
    mm1.SetOrgShape(fusionSize_, currentNDBlock, totalLength_, currentBSBlock, nD_); // [M,N,Ka,Kb,orgKc]
    mm1.SetSingleShape(fusionSize_, currentNDBlock, currentBSBlock);                 // [sM, sN, sK]
    // h_mix_grad^T @ x_rs
    mm1.SetTensorA(workSpaceGm_[workspaceBuf_.GetHMixGradOffset(offsetBS)], true); // [BS, N^2+2N]
    mm1.SetTensorB(workSpaceGm_[offsetXs], false);                                 // [BS, NDBlock]
    // 计算
    bool enAtomic = (offsetBS != 0);
    mm1.IterateAll(hcWeightGradGm_[offsetND], enAtomic); // 非连续写
    mm1.End();
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2Pipeline()
{
    if (blockIdx_ >= 2 * v1UsedCubeCoreNum_) {
        return;
    }
    hPreGatherOffsetBuf_ = fp32TBuf_.GetWithOffset<uint32_t>(PROCESS_V2_CHUNK_SIZE, 0);
    globalUbOffset_ = PROCESS_V2_CHUNK_SIZE * sizeof(uint32_t);
    for (uint32_t i = 0; i < PROCESS_V2_CHUNK_SIZE; i++) {
        hPreGatherOffsetBuf_.SetValue(i, i * N_ * sizeof(P));
    }

    // 确定处理范围 （1C2V 模式）
    uint32_t vecIdx = blockIdx_ % 2; // 0 或 1
    uint32_t buffId = 0;
    // ND方向切分：每个ND block大小为nDBlockSize_，在1C2V模式下，每个V核处理一半
    for (uint32_t offsetNDBase = dealStartND_; offsetNDBase < dealEndND_; offsetNDBase += ND_BLOCK_SIZE) {
        // 根据vecIdx选择处理ND block的前半部分(0)或后半部分(1)
        uint32_t offsetND = offsetNDBase + (ND_BLOCK_SIZE / 2) * vecIdx;
        if (offsetND >= dealEndND_) {
            continue; // 超出范围，跳过
        }
        uint32_t copySizeND = Min(ND_BLOCK_SIZE / 2, dealEndND_ - offsetND);

        // BS方向切分：与C0保持一致，按SINGLE_M切分
        LocalTensor<P> sumBuf = fp32OutQueue_.AllocTensor<P>();
        for (uint32_t offsetBS = 0; offsetBS < totalLength_; offsetBS += SINGLE_M) {
            uint32_t currentBSBlock = Min(SINGLE_M, uint32_t(totalLength_ - offsetBS));
            // 等待C0完成计算
            AscendC::CrossCoreWaitFlag(0x7);

            // 执行当前的V2
            ProcessV2(offsetND, copySizeND, offsetBS, offsetBS + currentBSBlock, sumBuf, buffId);
            buffId++;
            // 通知C1计算
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
        }
        if (withGamma_) {
            fp32OutQueue_.EnQue(sumBuf);
            sumBuf = fp32OutQueue_.DeQue<P>();
            DataCopyExtParams fp32ExtCopyParams;
            fp32ExtCopyParams.blockCount = 1;
            fp32ExtCopyParams.blockLen = copySizeND * sizeof(P);
            fp32ExtCopyParams.srcStride = 0;
            fp32ExtCopyParams.dstStride = 0;
            DataCopyPad(gammaGradGm_[offsetND], sumBuf, fp32ExtCopyParams);
        }
        fp32OutQueue_.FreeTensor(sumBuf);
    }
}

template <class T, class P>
template <bool hasGamma, bool isFirstChunk>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2XCastAndMulGamma(__ubuf__ P* xRsFp32Addr, __ubuf__ P* gammaOutAddr, __ubuf__ T* bf16InputAddr,
                                                                        __ubuf__ P* gammaSrcAddr, __ubuf__ P* gammaBroadAddr, uint32_t currentChunkSize, uint32_t copySizeND)
{
    uint32_t eleNumPerVf = 64;
    uint16_t vfLoopCnt = CeilDiv(copySizeND, eleNumPerVf);
    __VEC_SCOPE__
    {
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++) {
            uint32_t lenND = copySizeND;
            MicroAPI::RegTensor<T> xInB16Reg;
            MicroAPI::RegTensor<P> xFp32Reg, gammaReg, resultReg;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(lenND);
                uint32_t offset = tIdx * copySizeND + vfBlockIdx * eleNumPerVf;
                
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInB16Reg, bf16InputAddr + offset);
                MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInB16Reg, mask);

                MicroAPI::StoreAlign(xRsFp32Addr + offset, xFp32Reg, mask); // 后续其他计算需要xFp32

                if constexpr(hasGamma) {
                    MicroAPI::LoadAlign(gammaReg, gammaSrcAddr + vfBlockIdx * eleNumPerVf);
                    if constexpr(isFirstChunk) {
                       MicroAPI::StoreAlign(gammaBroadAddr + offset, gammaReg, mask);
                    }
                    
                    MicroAPI::Mul(resultReg, gammaReg, xFp32Reg, mask); // 逐元素乘：gamma * x
                    MicroAPI::StoreAlign(gammaOutAddr + offset, resultReg, mask);
                } else { // 没有gamma，原始逻辑是 xFp32Reg * 1.0，等价于不处理，这里直接store
                    MicroAPI::StoreAlign(gammaOutAddr + offset, xFp32Reg, mask);
                }
            }
        }
    }
}

template <class T, class P>
template <bool hasGamma>
__aicore__ inline void MhcPreBackwardKernel<T, P>::VFDoV2GammaMulXRsGradMm(__ubuf__ P* xRsGradUbAddr, __ubuf__ P* xRsGradMmAddr, __ubuf__ P* gammaBroadAddr, 
                                                   uint32_t currentChunkSize, uint32_t copySizeND)
{
    uint32_t eleNumPerVf = 64;
    uint16_t vfLoopCnt = CeilDiv(copySizeND, eleNumPerVf);
    __VEC_SCOPE__
    {
        for (uint16_t tIdx = 0; tIdx < (uint16_t)currentChunkSize; tIdx++)
        {
             uint32_t lenND = copySizeND;
              MicroAPI::RegTensor<P> gammaReg, xRsGradMmReg, resultReg;
              for (uint16_t vfBlockIdx = 0; vfBlockIdx < vfLoopCnt; vfBlockIdx++) {
                    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(lenND);
                    uint32_t offset = tIdx * copySizeND + vfBlockIdx * eleNumPerVf;
                    MicroAPI::LoadAlign(xRsGradMmReg, xRsGradMmAddr + offset);

                    if constexpr(hasGamma) {
                        MicroAPI::LoadAlign(gammaReg, gammaBroadAddr + vfBlockIdx * eleNumPerVf);
                        
                        MicroAPI::Mul(resultReg, gammaReg, xRsGradMmReg, mask); // 逐元素乘：gamma * xRsGradMm
                        MicroAPI::StoreAlign(xRsGradUbAddr + offset, resultReg, mask);
                    } else { // 没有gamma，原始逻辑是 xFp32Reg * 1.0，等价于不处理，这里直接store
                        MicroAPI::StoreAlign(xRsGradUbAddr + offset, xRsGradMmReg, mask);
                    }
              }
        }
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV2(uint32_t offsetND, uint32_t copySizeND, uint32_t bsStart,
                                                             uint32_t bsEnd, LocalTensor<P> &sumBuf, uint32_t buffId)
{
    // ==========================================================
    // 初始化
    // ==========================================================
    uint32_t currentN = offsetND / D_;
    uint32_t chunkSize = PROCESS_V2_CHUNK_SIZE;
    uint32_t currentBsSize = bsEnd - bsStart;
    buffId = buffId % 2;

    // ==========================================================
    // Buffer分配 - 按最大CHUNK大小分配
    // ==========================================================
    uint32_t ubOffset = globalUbOffset_; // sumBuf offset
    uint32_t maxChunkNDSize = chunkSize * copySizeND;
    LocalTensor<P> gammaBroadCast;
    if (withGamma_) {
        gammaBroadCast = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset); // 16K
        ubOffset = CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(32));
    }

    // 按最大chunkSize分配buffer: [chunkSize, copySizeND]

    LocalTensor<P> xRsFp32Buf = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset); // 64*64*4=16k
    auto hInGradFp32Buf = xRsFp32Buf;
    ubOffset = CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(32));

    LocalTensor<P> xGradVec3Buf = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset); // 64*64*4=16k
    ubOffset = CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(32));

    LocalTensor<P> xRsGradMmUb = fp32TBuf_.GetWithOffset<P>(maxChunkNDSize, ubOffset); // 64*64*4=16k (复用fp32inqueue)
    auto xRsGradInvUb = xRsGradMmUb;
    ubOffset = CeilAlign(uint32_t(ubOffset + maxChunkNDSize * sizeof(P)), uint32_t(32));

    LocalTensor<P> invRmsUb = fp32TBuf_.GetWithOffset<P>(currentBsSize, ubOffset); // 64B
    ubOffset = CeilAlign(uint32_t(ubOffset + currentBsSize * sizeof(P)), uint32_t(32));

    LocalTensor<P> hPreUb = fp32TBuf_.GetWithOffset<P>(currentBsSize * N_, ubOffset); // 1024 * 4 = 16k
    ubOffset = CeilAlign(uint32_t(ubOffset + currentBsSize * N_ * sizeof(P)), uint32_t(32));
    LocalTensor<uint8_t> sharedTmpBuf = fp32TBuf_.GetWithOffset<uint8_t>(5 * 1024, ubOffset);

    // ==========================================================
    // 参数初始化
    // ==========================================================
    DataCopyPadParams copyPadParams;
    copyPadParams.isPad = false;

    // TODO
    // 19.h_pre = load(h_pre) 加载整个bsSize的Hpre
    DataCopyParams bsCopyParams;
    bsCopyParams.blockCount = 1;
    bsCopyParams.blockLen = currentBsSize * N_ * sizeof(P);
    bsCopyParams.srcStride = 0; // 每个BS元素之间的stride
    bsCopyParams.dstStride = 0; // 目标buffer连续存储
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    DataCopyPad(hPreUb, hPreGm_[bsStart * N_], bsCopyParams, copyPadParams);
    // ==========================================================
    // 主循环: 按chunk=64切分BS方向
    // ==========================================================
    LocalTensor<P> invRmsInBuf;
    LocalTensor<T> hInGradInBuf;
    LocalTensor<P> gammaUb;
    LocalTensor<P> xRsGradMmInBuf;
    for (uint32_t bsChunkStart = bsStart; bsChunkStart < bsEnd; bsChunkStart += chunkSize) {
        uint32_t bsChunkEnd = Min(bsChunkStart + chunkSize, bsEnd);
        uint32_t currentChunkSize = bsChunkEnd - bsChunkStart;
        uint32_t chunkNDSize = currentChunkSize * copySizeND;

        // ==========================================================
        // 预加载当前chunk的常量数据
        // ==========================================================
        DataCopyParams bsCopyParams;
        bsCopyParams.blockCount = 1;
        bsCopyParams.blockLen = currentChunkSize * sizeof(P);
        bsCopyParams.srcStride = 0;
        bsCopyParams.dstStride = 0;

        // 11.加载inv_rms = load(inv_rms) : [B, S] (当前chunk)
        bf16InQueue_.AllocTensor<P>(invRmsInBuf);
        DataCopyPad(invRmsInBuf, invRmsGm_[bsChunkStart], bsCopyParams, copyPadParams);
        bf16InQueue_.EnQue(invRmsInBuf);
        bf16InQueue_.DeQue<P>(invRmsInBuf);

        // 13-15 计算inv_rms = (-1/nD) * pow(inv_rms, 3) * inv_rms_grad (当前chunk)
        Mul(invRmsUb, invRmsInBuf, invRmsInBuf, currentChunkSize); // inv_rms^2
        PipeBarrier<PIPE_V>();

        Mul(invRmsUb, invRmsUb, invRmsInBuf, currentChunkSize); // inv_rms^3
        PipeBarrier<PIPE_V>();
        bf16InQueue_.FreeTensor(invRmsInBuf);

        // 12.inv_rms_grad = load(inv_rms_grad) : [B, S] (当前chunk)
        PipeBarrier<PIPE_MTE2>();
        LocalTensor<P> invRmsGradBuf = fp32InQueue_.AllocTensor<P>();
        DataCopyPad(invRmsGradBuf, workSpaceGm_[workspaceBuf_.GetInvRmsGradOffset(bsChunkStart)], bsCopyParams,
                    copyPadParams);
        fp32InQueue_.EnQue(invRmsGradBuf);
        LocalTensor<P> invRmsGradUb = fp32InQueue_.DeQue<P>();

        Mul(invRmsUb, invRmsUb, invRmsGradUb, currentChunkSize);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(invRmsGradUb);

        Muls(invRmsUb, invRmsUb, (-scaleMean_), currentChunkSize);
        PipeBarrier<PIPE_V>();

        // // 2. 批量加载 x_rs 并bf16转换为fp32: [chunkSize, copySizeND]
        // blockCopyParams.srcStride = D_ * sizeof(T);  // 每个BS元素之间的stride
        DataCopyParams blockCopyParams;
        blockCopyParams.blockCount = currentChunkSize;
        blockCopyParams.blockLen = copySizeND * sizeof(T);
        blockCopyParams.srcStride = (D_ - copySizeND) * sizeof(T);
        blockCopyParams.dstStride = 0;
        bf16InQueue_.AllocTensor<T>(hInGradInBuf);
        DataCopyPad(hInGradInBuf, hInGradGm_[bsChunkStart * D_ + (offsetND % D_)], blockCopyParams, copyPadParams);
        bf16InQueue_.EnQue(hInGradInBuf);
        bf16InQueue_.DeQue<T>(hInGradInBuf);
        Cast(hInGradFp32Buf, hInGradInBuf, RoundMode::CAST_NONE, copySizeND * currentChunkSize);
        PipeBarrier<PIPE_V>();
        bf16InQueue_.FreeTensor(hInGradInBuf);

        // ==========================================================
        // 向量化处理整个chunk
        // ==========================================================
        // 20.计算外积 x_grad_vec3 = h_in_grad_fp32 * h_pre (广播h_pre到每个ND维度)
        // hPreValBuf需要广播到每个ND维度: [chunkSize] -> [chunkSize, copySizeND]
        auto hPreBroadCast = xRsGradMmUb;
        const uint32_t hPreDst[2] = {currentChunkSize, copySizeND};
        const uint32_t hPreSrc[2] = {currentChunkSize, 1};

        Gather(hPreUb, hPreUb, hPreGatherOffsetBuf_, uint32_t(((bsChunkStart - bsStart) * N_ + currentN) * sizeof(P)),
               currentChunkSize);
        PipeBarrier<PIPE_V>();

        BroadCast<P, 2, 1>(hPreBroadCast, hPreUb, hPreDst, hPreSrc, sharedTmpBuf);
        PipeBarrier<PIPE_V>();

        Mul(xGradVec3Buf, hInGradFp32Buf, hPreBroadCast, copySizeND * currentChunkSize);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams xCopyParams;
        DataCopyPadExtParams<T> xCopyPadParams;

        xCopyParams.blockCount = currentChunkSize;
        xCopyParams.blockLen = copySizeND * sizeof(T);
        xCopyParams.srcStride = (nD_ - copySizeND) * sizeof(T); // 每个BS元素之间的stride
        xCopyParams.dstStride = 0;

        LocalTensor<T> bf16InputBuf = fp32InQueue_.AllocTensor<T>();
        DataCopyPad(bf16InputBuf, xGm_[bsChunkStart * nD_ + offsetND], xCopyParams, xCopyPadParams);
        fp32InQueue_.EnQue(bf16InputBuf);
        bf16InputBuf = fp32InQueue_.DeQue<T>();
        // Cast(xRsFp32Buf, bf16InputBuf, RoundMode::CAST_NONE, chunkNDSize);
        // fp32InQueue_.FreeTensor(bf16InputBuf);
        // PipeBarrier<PIPE_V>();

        // 读取Gamma
        if (withGamma_ && bsChunkStart == bsStart) {
            bf16InQueue_.AllocTensor<P>(gammaUb);
            DataCopyParams fp32BlockCopyParams;
            fp32BlockCopyParams.blockCount = 1;
            fp32BlockCopyParams.blockLen = copySizeND * sizeof(P);
            fp32BlockCopyParams.srcStride = 0;
            fp32BlockCopyParams.dstStride = 0;
            DataCopyPad(gammaUb, gammaGm_[offsetND], fp32BlockCopyParams, copyPadParams);
            bf16InQueue_.EnQue(gammaUb);
            bf16InQueue_.DeQue<P>(gammaUb);
            // const uint32_t gammaBroadCastDst[2] = {currentChunkSize, copySizeND};
            // const uint32_t gammaBroadCastSrc[2] = {1, copySizeND};

            // BroadCast<P, 2, 0>(gammaBroadCast, gammaUb, gammaBroadCastDst, gammaBroadCastSrc, sharedTmpBuf);
            // PipeBarrier<PIPE_V>();
            // bf16InQueue_.FreeTensor(gammaUb);
        }

        // LocalTensor<P> gammaOutUb = bf16OutQueue_.AllocTensor<P>();
        // if (withGamma_) {
        //     Mul(gammaOutUb, gammaBroadCast, xRsFp32Buf, copySizeND * currentChunkSize);
        // } else {
        //     Muls(gammaOutUb, xRsFp32Buf, 1.0f, copySizeND * currentChunkSize);
        // }
        // PipeBarrier<PIPE_V>();

        // VF
        __ubuf__ T* bf16InputAddr = (__ubuf__ T*)bf16InputBuf.GetPhyAddr();
        __ubuf__ P* xRsFp32Addr = (__ubuf__ P*)xRsFp32Buf.GetPhyAddr();
        LocalTensor<P> gammaOutUb = bf16OutQueue_.AllocTensor<P>();
        __ubuf__ P* gammaOutAddr = (__ubuf__ P*)gammaOutUb.GetPhyAddr();
        if (withGamma_) {
            __ubuf__ P* gammaBroadAddr = (__ubuf__ P*)gammaBroadCast.GetPhyAddr();
            if (bsChunkStart == bsStart) { // 第一个chunk，gamma存到gammaBroadCast
                __ubuf__ P* gammaSrcAddr = (__ubuf__ P*)gammaUb.GetPhyAddr();
                VFDoV2XCastAndMulGamma<true, true>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, gammaSrcAddr, gammaBroadAddr, currentChunkSize, copySizeND);
                bf16InQueue_.FreeTensor(gammaUb);
            } else { // 后续chunk，从gammaBroadCast取gamma
                VFDoV2XCastAndMulGamma<true, false>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, gammaBroadAddr, gammaBroadAddr, currentChunkSize, copySizeND);
            }
        } else {
             VFDoV2XCastAndMulGamma<false, false>(xRsFp32Addr, gammaOutAddr, bf16InputAddr, nullptr, nullptr, currentChunkSize, copySizeND);
        }
        fp32InQueue_.FreeTensor(bf16InputBuf);
        bf16OutQueue_.EnQue(gammaOutUb);
        gammaOutUb = bf16OutQueue_.DeQue<P>();

        DataCopyExtParams fp32ExtCopyParams;
        fp32ExtCopyParams.blockCount = currentChunkSize;
        fp32ExtCopyParams.blockLen = copySizeND * sizeof(P);
        fp32ExtCopyParams.srcStride = 0;
        fp32ExtCopyParams.dstStride = (ND_BLOCK_SIZE - copySizeND) * sizeof(P);

        uint64_t offset = workspaceBuf_.GetXRsOffset(GetBlockIdx() / 2, buffId); // 在workspace的偏移
        uint32_t vecIdx = blockIdx_ % 2;                                         // 0 或 1
        offset += (bsChunkStart - bsStart) * ND_BLOCK_SIZE;
        offset += vecIdx * (ND_BLOCK_SIZE / 2); // 在 1024 * 128 内的偏移

        DataCopyPad(workSpaceGm_[offset], gammaOutUb, fp32ExtCopyParams);
        bf16OutQueue_.FreeTensor(gammaOutUb);

        const uint32_t xRsGradInvBroadCastDst[2] = {currentChunkSize, copySizeND};
        const uint32_t xRsGradInvBroadCastSrc[2] = {currentChunkSize, 1};
        BroadCast<P, 2, 1>(xRsGradInvUb, invRmsUb, xRsGradInvBroadCastDst, xRsGradInvBroadCastSrc, sharedTmpBuf);
        PipeBarrier<PIPE_V>();

        Mul(xRsGradInvUb, xRsGradInvUb, xRsFp32Buf, copySizeND * currentChunkSize);
        PipeBarrier<PIPE_V>();

        // add
        // 21. <- 16
        Add(xGradVec3Buf, xGradVec3Buf, xRsGradInvUb, chunkNDSize);
        PipeBarrier<PIPE_V>();

        // datacopy x rs grad mm

        uint64_t offsetXRsGradMm = workspaceBuf_.GetXRsGradOffset(GetBlockIdx() / 2, buffId);
        offsetXRsGradMm += (bsChunkStart - bsStart) * ND_BLOCK_SIZE + vecIdx * (ND_BLOCK_SIZE / 2);

        bf16InQueue_.AllocTensor<P>(xRsGradMmInBuf);
        blockCopyParams.blockCount = currentChunkSize;
        blockCopyParams.blockLen = copySizeND * sizeof(P);
        blockCopyParams.srcStride = (128 - copySizeND) * sizeof(P); // 每个BS元素之间的stride
        blockCopyParams.dstStride = 0;

        DataCopyPad(xRsGradMmInBuf, workSpaceGm_[offsetXRsGradMm], blockCopyParams, copyPadParams);

        bf16InQueue_.EnQue(xRsGradMmInBuf);
        bf16InQueue_.DeQue<P>(xRsGradMmInBuf);
        // if (withGamma_) {
        //     Mul(xRsGradMmUb, gammaBroadCast, xRsGradMmInBuf, copySizeND * currentChunkSize);
        // } else {
        //     Muls(xRsGradMmUb, xRsGradMmInBuf, 1.0f, copySizeND * currentChunkSize);
        // }
        // PipeBarrier<PIPE_V>();
        __ubuf__ P* xRsGradMmInBufAddr = (__ubuf__ P*)xRsGradMmInBuf.GetPhyAddr();
        __ubuf__ P* xRsGradMmUbAddr = (__ubuf__ P*)xRsGradMmUb.GetPhyAddr();
        if (withGamma_) {
            __ubuf__ P* gammaBroadCastAddr = (__ubuf__ P*)gammaBroadCast.GetPhyAddr();
            VFDoV2GammaMulXRsGradMm<true>(xRsGradMmUbAddr, xRsGradMmInBufAddr, gammaBroadCastAddr, currentChunkSize, copySizeND);
        } else {
            VFDoV2GammaMulXRsGradMm<false>(xRsGradMmUbAddr, xRsGradMmInBufAddr, nullptr, currentChunkSize, copySizeND);
        }

        // uint32_t srcReduceShape[] = {currentChunkSize, copySizeND}; // 64，64
        if (withGamma_) {
            __ubuf__ P *xRsGradMmInBufAddr = (__ubuf__ P *)xRsGradMmInBuf.GetPhyAddr();
            __ubuf__ P *xRsFp32BufAddr = (__ubuf__ P *)xRsFp32Buf.GetPhyAddr();
            __ubuf__ P *sumBufAddr = (__ubuf__ P *)sumBuf.GetPhyAddr();
            uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(P);
            uint32_t ndComputeLen = copySizeND;
            uint16_t ndRepeatTimes = Ceil(copySizeND, oneRepeatSize);
            uint16_t bsRepeatTimes = currentChunkSize;
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<P> xRsGradMmInReg, xRsFp32Reg, mulReg, sumReg;
                for (uint16_t ndRepeatIdx = 0; ndRepeatIdx < ndRepeatTimes; ndRepeatIdx++) {
                    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(ndComputeLen);
                    if (bsChunkStart == 0) {
                        MicroAPI::Duplicate(sumReg, 0.0f, mask);
                    } else {
                        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                        MicroAPI::LoadAlign(sumReg, sumBufAddr + ndRepeatIdx * oneRepeatSize);
                    }
                    for (uint16_t bsRepeatIdx = 0; bsRepeatIdx < bsRepeatTimes; bsRepeatIdx++) {
                        uint32_t Offset = bsRepeatIdx * copySizeND + ndRepeatIdx * oneRepeatSize;
                        MicroAPI::LoadAlign(xRsGradMmInReg, xRsGradMmInBufAddr + Offset);
                        MicroAPI::LoadAlign(xRsFp32Reg, xRsFp32BufAddr + Offset);
                        MicroAPI::Mul(mulReg, xRsGradMmInReg, xRsFp32Reg, mask);
                        MicroAPI::Add(sumReg, sumReg, mulReg, mask);
                    }
                    MicroAPI::StoreAlign(sumBufAddr + ndRepeatIdx * oneRepeatSize, sumReg, mask);
                }
            }

            // Mul(xRsGradMmInBuf, xRsFp32Buf, xRsGradMmInBuf, copySizeND * currentChunkSize);
            // LocalTensor<P> reduceSumBuf = xRsFp32Buf;
            // PipeBarrier<PIPE_V>();
            // if (bsChunkStart == 0) {
            //     ReduceSum<P, Pattern::Reduce::RA, true>(sumBuf, xRsGradMmInBuf, srcReduceShape, true);
            // } else {
            //     ReduceSum<P, Pattern::Reduce::RA, true>(reduceSumBuf, xRsGradMmInBuf, srcReduceShape, true);
            //     PipeBarrier<PIPE_V>();
            //     Add(sumBuf, sumBuf, reduceSumBuf, copySizeND);
            // }
            // PipeBarrier<PIPE_V>();
        }
        bf16InQueue_.FreeTensor(xRsGradMmInBuf);
        // 22. <- 10
        Add(xGradVec3Buf, xGradVec3Buf, xRsGradMmUb, chunkNDSize);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> bf16OutputBuf = bf16OutQueue_.AllocTensor<T>();
        Cast(bf16OutputBuf, xGradVec3Buf, RoundMode::CAST_RINT, chunkNDSize);
        PipeBarrier<PIPE_V>();
        bf16OutQueue_.EnQue(bf16OutputBuf);
        bf16OutputBuf = bf16OutQueue_.DeQue<T>();

        DataCopyExtParams bf16ExtCopyParams;
        bf16ExtCopyParams.blockCount = currentChunkSize;
        bf16ExtCopyParams.blockLen = copySizeND * sizeof(T);
        bf16ExtCopyParams.srcStride = 0;
        bf16ExtCopyParams.dstStride = (nD_ - copySizeND) * sizeof(T);
        DataCopyPad(xGradGm_[bsChunkStart * nD_ + offsetND], bf16OutputBuf, bf16ExtCopyParams);
        bf16OutQueue_.FreeTensor(bf16OutputBuf);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreBackwardKernel<T, P>::ProcessV3()
{
    // alphagrad

    LocalTensor<uint8_t> tmpLocal = fp32TBuf_.GetWithOffset<uint8_t>(hFusionBufLen_ / 4, 0);
    constexpr bool isReuse = false;

    LocalTensor<P> alphaGradInLocal = fp32InQueue_.AllocTensor<P>();
    LocalTensor<P> alphaGradOutLocal = fp32OutQueue_.AllocTensor<P>();

    dataCopyParams_.blockCount = 1;
    dataCopyParams_.blockLen = usedVecCoreNum_ * ALPHA_GRAD_PADDING * sizeof(P);
    dataCopyParams_.srcStride = 0;
    dataCopyParams_.dstStride = 0;
    DataCopyPad(alphaGradInLocal, workSpaceGm_[workspaceBuf_.GetAlphaGradOffset(0)], dataCopyParams_,
                dataCopyPadParams_);
    fp32InQueue_.EnQue(alphaGradInLocal);
    alphaGradInLocal = fp32InQueue_.DeQue<P>();

    uint32_t alphaGradShapeSrc[] = {usedVecCoreNum_, ALPHA_GRAD_PADDING};
    ReduceSum<P, AscendC::Pattern::Reduce::RA, isReuse>(alphaGradOutLocal, alphaGradInLocal, tmpLocal,
                                                        alphaGradShapeSrc, true);

    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    alphaGradOutLocal.SetValue(1, alphaGradOutLocal.GetValue(8));
    alphaGradOutLocal.SetValue(2, alphaGradOutLocal.GetValue(16));
    fp32OutQueue_.EnQue(alphaGradOutLocal);
    alphaGradOutLocal = fp32OutQueue_.DeQue<P>();
    PipeBarrier<PIPE_V>();

    dataCopyParams_.blockCount = 1;
    dataCopyParams_.blockLen = 3 * sizeof(P);
    dataCopyParams_.srcStride = 0;
    dataCopyParams_.dstStride = 0;
    DataCopyPad(alphaGradGm_, alphaGradOutLocal, dataCopyParams_);

    fp32InQueue_.FreeTensor(alphaGradInLocal);
    fp32OutQueue_.FreeTensor(alphaGradOutLocal);

    // biasgrad

    LocalTensor<P> biasGradInLocal = fp32InQueue_.AllocTensor<P>();
    LocalTensor<P> biasGradOutLocal = fp32OutQueue_.AllocTensor<P>();

    dataCopyParams_.blockCount = 1;
    dataCopyParams_.blockLen = fusionSize_ * usedVecCoreNum_ * sizeof(P);
    dataCopyParams_.srcStride = 0;
    dataCopyParams_.dstStride = 0;
    DataCopyPad(biasGradInLocal, workSpaceGm_[workspaceBuf_.GetBiasGradOffset(0)], dataCopyParams_, dataCopyPadParams_);

    fp32InQueue_.EnQue(biasGradInLocal);
    biasGradInLocal = fp32InQueue_.DeQue<P>();

    uint32_t biasGradShapeSrc[] = {usedVecCoreNum_, fusionSize_};
    ReduceSum<P, AscendC::Pattern::Reduce::RA, isReuse>(biasGradOutLocal, biasGradInLocal, tmpLocal, biasGradShapeSrc,
                                                        true);

    fp32OutQueue_.EnQue(biasGradOutLocal);
    biasGradOutLocal = fp32OutQueue_.DeQue<P>();

    dataCopyParams_.blockCount = 1;
    dataCopyParams_.blockLen = fusionSize_ * sizeof(P);
    dataCopyParams_.srcStride = 0;
    dataCopyParams_.dstStride = 0;
    DataCopyPad(biasPostGradGm_[0], biasGradOutLocal, dataCopyParams_);

    fp32InQueue_.FreeTensor(biasGradInLocal);
    fp32OutQueue_.FreeTensor(biasGradOutLocal);
}

} // namespace MhcPreBackward

#endif // __mhc_pre_backward_KERNEL_H_
