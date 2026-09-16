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
// mhc_sandwich_norm_post_preonly_kernel.h — 类定义 + 共享 building blocks
//
// 文件结构:
//   1. 类声明 + Init / Process
//   2. 共享 building blocks (被三条路径复用)
//   3. 成员变量
//   4. #include 三条路径的实现头文件
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
        myHeadCount_    = td.myHeadCount;
        maxTokensPerPair_ = td.maxTokensPerPair;
        syncGmOffsetFloats_ = td.syncGmOffsetFloats;
        returnHInF32_ = (td.returnHInF32 != 0);

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum_) {
            tokenStart_ = 0;
            tokenEnd_ = 0;
            return;
        }

        // ---- Core mapping ----
        if (coresPerToken_ == NUM_TWO) {
            coreRole_ = blockIdx % DUAL_CORE_COUNT;
            uint32_t pairIdx = blockIdx / DUAL_CORE_COUNT;
            headStart_ = coreRole_ * myHeadCount_;

            uint32_t numPairs = usedCoreNum_ / NUM_TWO;
            if (numPairs == 0)
                numPairs = 1;
            if (pairIdx < blockPivot_) {
                tokenStart_ = pairIdx * rowsPerCoreSp_;
                tokenEnd_ = tokenStart_ + rowsPerCoreSp_;
            } else {
                tokenStart_ = blockPivot_ * rowsPerCoreSp_ + (pairIdx - blockPivot_) * rowsPerCore_;
                tokenEnd_ = tokenStart_ + rowsPerCore_;
            }
            uint64_t pairWsFloats = WS_HIN_BASE + (uint64_t)N_ * D_;
            pairWsBase_ = (uint64_t)pairIdx * pairWsFloats;
        } else {
            coreRole_ = 0;
            headStart_ = 0;
            if (blockIdx < blockPivot_) {
                tokenStart_ = blockIdx * rowsPerCoreSp_;
                tokenEnd_ = tokenStart_ + rowsPerCoreSp_;
            } else {
                tokenStart_ = blockPivot_ * rowsPerCoreSp_ + (blockIdx - blockPivot_) * rowsPerCore_;
                tokenEnd_ = tokenStart_ + rowsPerCore_;
            }
            uint64_t coreWsFloats = (uint64_t)(N_ + 1) * D_;
            pairWsBase_ = (uint64_t)blockIdx * coreWsFloats;
        }
        if (tokenEnd_ > totalTokens_)
            tokenEnd_ = totalTokens_;

        // ---- GM tensors ----
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

        uint64_t maxPairWs = WS_HIN_BASE + (uint64_t)N_MAX * D_MAX;
        uint64_t maxSingleWs = (uint64_t)(N_MAX + 1) * D_MAX;
        uint64_t wsPerUnit = maxPairWs > maxSingleWs ? maxPairWs : maxSingleWs;
        workspaceGm_.SetGlobalBuffer((__gm__ float *)workspace, (uint64_t)usedCoreNum_ * wsPerUnit);

        // ---- UB allocation ----
        uint32_t x2HeadCount = (coresPerToken_ == NUM_TWO) ? myHeadCount_ : N_;
        uint32_t resBatchSize = (coresPerToken_ == NUM_TWO) ? N_ : 2u;
        pipe_.InitBuffer(bufHOutBf16_,       1, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufHOutFp32_,          dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma0Fp32_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma1Fp32_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufX2Fp32_,            x2HeadCount * dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufOutFp32_,           dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufResidualBf16_,   1, resBatchSize * dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufResFp32_,           resBatchSize * dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufSqFp32_,            dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufPostComb_,          POST_COMB_SIZE * sizeof(float));
        pipe_.InitBuffer(bufPhiFp32_,  BUFFER_NUM_TWO, dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufHInPrimeBf16_,   1, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufScalar_,            SCALAR_BUF_SIZE);
        pipe_.InitBuffer(bufReduceWork_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma2Fp32_,        dimTile_ * sizeof(float));

        // Sync workspace for dual-core sense-reversing barrier.
        // syncGmOffsetFloats_ is computed host-side in tiling as
        // `coreNumAiv * wsPerUnit_MAX` so kernel writes flags into exactly the
        // region host reserved (not into the middle of the data region).
        if (coresPerToken_ == NUM_TWO) {
            syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                (__gm__ float *)workspace + syncGmOffsetFloats_),
                (uint64_t)usedCoreNum_ * WS_SYNC_BLOCK);
            pipe_.InitBuffer(bufSyncUb_, 1, WS_SYNC_BLOCK * sizeof(int32_t));
        }

        evtMte2toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        evtMte3toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        evtVtoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        evtVtoMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
    }

    // =====================================================================
    //  Process: gamma hoisting + token loop dispatch
    // =====================================================================
    __aicore__ inline void Process()
    {
        if (tokenStart_ >= tokenEnd_) {
            return;
        }
        // Initialize sync flag slot to 0 (sense-reversing barrier)
        if (coresPerToken_ == NUM_TWO) {
            syncRound_ = 0;
            LocalTensor<int32_t> syncUb = bufSyncUb_.AllocTensor<int32_t>();
            for (uint32_t i = 0; i < WS_SYNC_BLOCK; ++i) {
                syncUb.SetValue(i, 0);
            }
            PipeBarrier<PIPE_V>();
            DataCopy(syncGm_[GetBlockIdx() * WS_SYNC_BLOCK], syncUb, WS_SYNC_BLOCK);
            PipeBarrier<PIPE_ALL>();
            bufSyncUb_.FreeTensor(syncUb);
        }

        // Cache scalars
        alpha0_ = alpha0Gm_.GetValue(0);
        for (uint32_t i = 0; i < phiRows_ && i < N_MAX; ++i) {
            biasCache_[i] = biasGm_.GetValue(i);
        }

        // Gamma hoisting: only for single-tile paths (dimLoop==1)
        if (dimLoop_ == 1) {
            DataCopyExtParams gP = {1, 0, 0, 0, 0};
            gP.blockLen = dimTile_ * sizeof(float);
            DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
            DataCopyPad(bufGamma0Fp32_.Get<float>(), gamma0Gm_[0], gP, gPad);
            DataCopyPad(bufGamma1Fp32_.Get<float>(), gamma1Gm_[0], gP, gPad);
            // gamma_2 is [N*D] — loaded per-head in RmsNormMid, not hoisted
        }
        
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

        // Dispatch
        for (uint32_t t = tokenStart_; t < tokenEnd_; ++t) {
            if (coresPerToken_ == NUM_TWO && dimLoop_ == 1)
                ProcessTokenDualCore(t);
            else if (coresPerToken_ == NUM_TWO && dimLoop_ > 1)
                ProcessTokenDualCoreMultiTile(t);
            else
                ProcessToken(t);
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(evtMte2toV_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(evtMte3toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(evtVtoMte3_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(evtVtoMte2_);
    }

    // =================================================================
    //  Path-specific methods (implementations in separate headers)
    // =================================================================
private:
    __aicore__ inline void ProcessTokenDualCore(uint32_t token);
    __aicore__ inline void ProcessTokenDualCoreMultiTile(uint32_t token);
    __aicore__ inline void ProcessToken(uint32_t token);

    // Single-core sub-phase helpers
    __aicore__ inline void PhaseAB_SingleTile(uint32_t token,
        const float (&postVals)[N_MAX], const float (&combVals)[N_MAX*N_MAX],
        float (&xHatVals)[N_MAX], float &sumSqPre);
    __aicore__ inline void PhaseAB_MultiTile(uint32_t token,
        const float (&postVals)[N_MAX], const float (&combVals)[N_MAX*N_MAX],
        float (&xHatVals)[N_MAX], float &sumSqPre);
    __aicore__ inline void PhaseC_HInAndX2Output(uint32_t token,
        const float (&gate)[N_MAX], float &sumSq1);
    __aicore__ inline void PhaseD_RmsNorm1Output(uint32_t token, float invRms1);
    __aicore__ __attribute__((noinline)) void Apply2BatchResidualAxpyTail(
        uint64_t rBaseTile, uint32_t tD,
        const float (&combVals)[N_MAX*N_MAX]);

    // =================================================================
    //  Shared building blocks (被三条路径复用)
    // =================================================================
#include "mhc_sandwich_norm_post_preonly_kernel_compute.h"
#include "mhc_sandwich_norm_post_preonly_kernel_io.h"

    // =================================================================
    //  Members
    // =================================================================
    TPipe pipe_;

    // UB buffers
    TQue<TPosition::VECIN, 1>  bufHOutBf16_;
    TBuf<TPosition::VECCALC>   bufHOutFp32_;
    TBuf<TPosition::VECCALC>   bufGamma0Fp32_;
    TBuf<TPosition::VECCALC>   bufGamma1Fp32_;  // [R9] also x2[head0,tile0] store
    TBuf<TPosition::VECCALC>   bufGamma2Fp32_;  // [R9] also x2[head1,tile0] store
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
    GlobalTensor<int32_t>      syncGm_;
    TQue<TPosition::VECIN, 1>  bufSyncUb_;

    // GM tensors
    GlobalTensor<T>     hOutGm_, residualGm_, hInPrimeGm_, x2OutGm_;
    GlobalTensor<float> hPostGm_, hResGm_, phiGm_;
    GlobalTensor<float> alpha0Gm_, biasGm_;
    GlobalTensor<float> gamma0Gm_, gamma1Gm_, gamma2Gm_;
    GlobalTensor<float> hInF32Gm_;
    GlobalTensor<float> workspaceGm_;

    // Tiling parameters
    uint32_t totalTokens_ = 0, N_ = 0, D_ = 0, K_ = 0;
    uint32_t usedCoreNum_ = 0, rowsPerCore_ = 0, rowsPerCoreSp_ = 0, blockPivot_ = 0;
    uint32_t dimTile_ = 0, dimLoop_ = 0, lastDimTile_ = 0;
    uint32_t phiRows_ = 0, phiCols_ = 0;
    float normEps_ = 1e-6f, mhcPreNormEps_ = 1e-6f, hcEps_ = 1e-6f, invD_ = 0.0f, invK_ = 0.0f;
    uint32_t tokenStart_ = 0, tokenEnd_ = 0;
    float alpha0_ = 0.0f;
    float biasCache_[N_MAX] = {0.0f};
    uint64_t pairWsBase_ = 0;
    bool useRmsMid_ = false;
    bool returnHInF32_ = false;

    // N-axis split
    uint32_t coresPerToken_ = 1;
    uint32_t myHeadCount_ = 0;
    uint32_t headStart_ = 0, coreRole_ = 0, maxTokensPerPair_ = 0;

    // Sync region offset from host tiling
    uint32_t syncGmOffsetFloats_ = 0;
    uint32_t syncRound_ = 0;  // sense-reversing barrier round counter

    int32_t evtMte2toV_ = 0;
    int32_t evtMte3toV_ = 0;
    int32_t evtVtoMte3_ = 0;
    int32_t evtVtoMte2_ = 0;
};

// Include path implementations (methods defined inline within the class body above
// are declared private; these headers provide the remaining method bodies)
#include "mhc_sandwich_norm_post_preonly_dualcore.h"
#include "mhc_sandwich_norm_post_preonly_dualcore_mt.h"
#include "mhc_sandwich_norm_post_preonly_singlecore.h"

#endif  // MHC_SANDWICH_NORM_POST_PREONLY_KERNEL_H_
