/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ========================================================================
// mhc_sandwich_norm_post_preonly_kernel.h — 类定义和共享计算模块
//
// 文件结构:
//   1. 类声明 + Init / Process
//   2. 共享计算模块
//   3. 成员变量
//   4. 引入各执行路径的实现头文件
// ========================================================================
#ifndef MHC_SANDWICH_NORM_POST_PREONLY_KERNEL_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_KERNEL_H_

#include "mhc_sandwich_norm_post_preonly_common.h"

template<typename T>
class KernelMhcSandwichNormPostPreonly {
public:
    __aicore__ inline KernelMhcSandwichNormPostPreonly() {}

    // =====================================================================
    //  Init: 解析 tiling, 设置 GM buffer, 分配 UB
    // =====================================================================
    __aicore__ inline void Init(
        GM_ADDR h_out_gm, GM_ADDR residual_gm,
        GM_ADDR h_post_gm, GM_ADDR h_res_gm,
        GM_ADDR phi_gm, GM_ADDR alpha_0_gm, GM_ADDR bias_gm,
        GM_ADDR gamma_0_gm, GM_ADDR gamma_1_gm, GM_ADDR gamma_2_gm,
        GM_ADDR h_in_prime_gm, GM_ADDR x_2_out_gm, GM_ADDR h_in_f32_gm,
        GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(td, tiling);

        totalTokens_    = td.totalTokens;
        N_              = td.headCount;
        D_              = td.embedDim;
        usedCoreNum_    = td.usedCoreNum;
        rowsPerCore_    = td.rowsPerCore;
        rowsPerCoreSp_  = td.rowsPerCoreSp;
        blockPivot_     = td.blockPivot;
        dimTile_        = td.dimTile;
        dimLoop_        = td.dimLoop;
        lastDimTile_    = td.lastDimTile;
        phiRows_        = td.phiRows;
        phiCols_        = td.phiCols;
        normEps_        = td.normEps;
        mhcPreNormEps_  = td.mhcPreNormEps;
        hcEps_          = td.hcEps;
        K_              = N_ * D_;
        invD_           = td.invD;
        invK_           = td.invK;
        useRmsMid_      = (td.useRmsMid != 0);
        coresPerToken_  = td.coresPerToken;
        returnHInF32_ = (td.returnHInF32 != 0);

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum_) {
            tokenStart_ = 0;
            tokenEnd_ = 0;
            return;
        }

        // 第一阶段按token分区启动相互独立的AIV，不使用block奇偶配对、相邻block配对或N轴切分。
        if (blockIdx < blockPivot_) {
            tokenStart_ = blockIdx * rowsPerCoreSp_;
            tokenEnd_ = tokenStart_ + rowsPerCoreSp_;
        } else {
            tokenStart_ = blockPivot_ * rowsPerCoreSp_ + (blockIdx - blockPivot_) * rowsPerCore_;
            tokenEnd_ = tokenStart_ + rowsPerCore_;
        }
        if (tokenEnd_ > totalTokens_)
            tokenEnd_ = totalTokens_;

        // GM张量
        hOutGm_.SetGlobalBuffer((__gm__ T *)h_out_gm,         (uint64_t)totalTokens_ * D_);
        residualGm_.SetGlobalBuffer((__gm__ T *)residual_gm,  (uint64_t)totalTokens_ * N_ * D_);
        hPostGm_.SetGlobalBuffer((__gm__ float *)h_post_gm, (uint64_t)totalTokens_ * N_);
        hResGm_.SetGlobalBuffer((__gm__ float *)h_res_gm, (uint64_t)totalTokens_ * N_ * N_);
        phiGm_.SetGlobalBuffer((__gm__ float *)phi_gm,        (uint64_t)phiRows_ * phiCols_);
        alpha0Gm_.SetGlobalBuffer((__gm__ float *)alpha_0_gm, ALIGN8);
        biasGm_.SetGlobalBuffer((__gm__ float *)bias_gm,      ALIGN8);
        gamma0Gm_.SetGlobalBuffer((__gm__ float *)gamma_0_gm, D_);
        gamma1Gm_.SetGlobalBuffer((__gm__ float *)gamma_1_gm, D_);
        if (useRmsMid_) {
            gamma2Gm_.SetGlobalBuffer((__gm__ float *)gamma_2_gm, K_);
        }
        hInPrimeGm_.SetGlobalBuffer((__gm__ T *)h_in_prime_gm, (uint64_t)totalTokens_ * D_);
        x2OutGm_.SetGlobalBuffer((__gm__ T *)x_2_out_gm,      (uint64_t)totalTokens_ * N_ * D_);
        if (h_in_f32_gm != nullptr) {
            hInF32Gm_.SetGlobalBuffer((__gm__ float *)h_in_f32_gm, (uint64_t)totalTokens_ * D_);
        }

        // UB空间分配
        pipe_.InitBuffer(bufHOutBf16_,       1, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufHOutFp32_,          dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma0Fp32_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma1Fp32_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufX2Fp32_,            N_ * dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufOutFp32_,           dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufResidualBf16_,   1, N_ * dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufResFp32_,           N_ * dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufSqFp32_,            dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufPostComb_,          POST_COMB_SIZE * sizeof(float));
        pipe_.InitBuffer(bufPhiFp32_,  BUFFER_NUM_TWO, dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufHInPrimeBf16_,   1, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufScalar_,            SCALAR_BUF_SIZE);
        pipe_.InitBuffer(bufReduceWork_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma2Fp32_,        dimTile_ * sizeof(float));

        evtMte2toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        evtVtoMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
        evtMte3toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        evtVtoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        evtMte3toMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_MTE2>());
    }

    // =====================================================================
    //  Process: gamma hoisting + token loop dispatch
    // =====================================================================
    __aicore__ inline void Process()
    {
        if (tokenStart_ >= tokenEnd_) {
            return;
        }
        // 缓存标量
        alpha0_ = alpha0Gm_.GetValue(0);
        for (uint32_t i = 0; i < phiRows_ && i < N_MAX; ++i) {
            biasCache_[i] = biasGm_.GetValue(i);
        }

        // 仅单分块路径提前加载gamma
        if (dimLoop_ == 1) {
            DataCopyExtParams gP = {1, 0, 0, 0, 0};
            gP.blockLen = dimTile_ * sizeof(float);
            DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
            DataCopyPad(bufGamma0Fp32_.Get<float>(), gamma0Gm_[0], gP, gPad);
            DataCopyPad(bufGamma1Fp32_.Get<float>(), gamma1Gm_[0], gP, gPad);
            // gamma_2的形状为[N*D]，在RMSNorm_mid中按head加载
        }
        
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

        for (uint32_t t = tokenStart_; t < tokenEnd_; ++t) {
            ProcessTokenSingleAiv(t);
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(evtMte2toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(evtVtoMte2_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(evtMte3toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(evtVtoMte3_);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
    }

    // =================================================================
    //  各执行路径的方法，实现在独立头文件中
    // =================================================================
private:
    __aicore__ inline void ProcessTokenSingleAiv(uint32_t token);
    __aicore__ inline void ComputeX2PostTile(uint32_t token, uint32_t tOff, uint32_t tD,
        float invRms0, const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX]);
    __aicore__ inline void LoadX2OutputTile(uint32_t token, uint32_t tOff, uint32_t tD);
    __aicore__ inline void AccumulateHalfPartials(uint32_t tOff, uint32_t tD,
        float (&sumSqHalf)[2], float (&dotHalf)[2][N_MAX]);

    // =================================================================
    //  共享计算模块
    // =================================================================
#include "mhc_sandwich_norm_post_preonly_kernel_compute.h"
#include "mhc_sandwich_norm_post_preonly_kernel_io.h"

    // =================================================================
    //  成员变量
    // =================================================================
    TPipe pipe_;

    // UB缓存
    TQue<TPosition::VECIN, 1>  bufHOutBf16_;
    TBuf<TPosition::VECCALC>   bufHOutFp32_;
    TBuf<TPosition::VECCALC>   bufGamma0Fp32_;
    TBuf<TPosition::VECCALC>   bufGamma1Fp32_;
    TBuf<TPosition::VECCALC>   bufGamma2Fp32_;
    TBuf<TPosition::VECCALC>   bufX2Fp32_;
    TBuf<TPosition::VECCALC>   bufOutFp32_;
    TQue<TPosition::VECIN, 1>  bufResidualBf16_;
    TBuf<TPosition::VECCALC>   bufResFp32_;
    TBuf<TPosition::VECCALC>   bufSqFp32_;
    TBuf<TPosition::VECCALC>   bufPostComb_;
    TQue<TPosition::VECIN, BUFFER_NUM_TWO>  bufPhiFp32_;
    TQue<TPosition::VECOUT, 1> bufHInPrimeBf16_;
    TBuf<TPosition::VECCALC>   bufScalar_;
    TBuf<TPosition::VECCALC>   bufReduceWork_;

    // GM张量
    GlobalTensor<T>     hOutGm_, residualGm_, hInPrimeGm_, x2OutGm_;
    GlobalTensor<float> hPostGm_, hResGm_, phiGm_;
    GlobalTensor<float> alpha0Gm_, biasGm_;
    GlobalTensor<float> gamma0Gm_, gamma1Gm_, gamma2Gm_;
    GlobalTensor<float> hInF32Gm_;

    // Tiling参数
    uint32_t totalTokens_ = 0, N_ = 0, D_ = 0, K_ = 0;
    uint32_t usedCoreNum_ = 0, rowsPerCore_ = 0, rowsPerCoreSp_ = 0, blockPivot_ = 0;
    uint32_t dimTile_ = 0, dimLoop_ = 0, lastDimTile_ = 0;
    uint32_t phiRows_ = 0, phiCols_ = 0;
    float normEps_ = 1e-6f, mhcPreNormEps_ = 1e-6f, hcEps_ = 1e-6f, invD_ = 0.0f, invK_ = 0.0f;
    uint32_t tokenStart_ = 0, tokenEnd_ = 0;
    float alpha0_ = 0.0f;
    float biasCache_[N_MAX] = {0.0f};
    bool useRmsMid_ = false;
    bool returnHInF32_ = false;

    // A5执行方案选择值，第一阶段只使用1
    uint32_t coresPerToken_ = 1;

    int32_t evtMte2toV_ = 0;
    int32_t evtVtoMte2_ = 0;
    int32_t evtMte3toV_ = 0;
    int32_t evtVtoMte3_ = 0;
    int32_t evtMte3toMte2_ = 0;
};

// 引入各执行路径的方法实现
#include "mhc_sandwich_norm_post_preonly_single_aiv.h"

#endif // MHC_SANDWICH_NORM_POST_PREONLY_KERNEL_H_
