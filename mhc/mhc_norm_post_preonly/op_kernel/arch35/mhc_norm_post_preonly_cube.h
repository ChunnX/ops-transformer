/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#ifndef MHC_NORM_POST_PREONLY_CUBE_H_
#define MHC_NORM_POST_PREONLY_CUBE_H_

#include "mhc_norm_post_preonly_common.h"

// Decode目标shape专用的FP32 Phi投影。AIV把R3后的低精度x2恢复为FP32并写入workspace，
// AIC执行[1, 4D] x [4D, 4]，从AIV关键路径移除320KB Phi读取和四路Vector点积。
class MhcNormPostPreonlyCube {
public:
    __aicore__ inline void Init(GM_ADDR phi, GM_ADDR workspace, uint32_t totalTokens, uint32_t d)
    {
        totalTokens_ = totalTokens;
        d_ = d;
        k_ = N_MAX * d_;
        blockIdx_ = GetBlockIdx();

        const uint64_t partialElems = (uint64_t)totalTokens_ * 2U * PARTIAL_SLOT_ELEMS;
        const uint64_t x2Elems = (uint64_t)totalTokens_ * k_;
        GlobalTensor<float> workspaceFp32;
        workspaceFp32.SetGlobalBuffer((__gm__ float *)workspace,
                                      partialElems + x2Elems + (uint64_t)totalTokens_ * PARTIAL_SLOT_ELEMS);
        x2Gm_ = workspaceFp32[partialElems];
        dotGm_ = workspaceFp32[partialElems + x2Elems];
        phiGm_.SetGlobalBuffer((__gm__ float *)phi, (uint64_t)N_MAX * k_);

        pipe_.InitBuffer(l1Buf_, CUBE_L1_TOTAL_ELEMS * sizeof(float));
        aL1_ = l1Buf_.Get<float>();
        bL1_ = aL1_[CUBE_L1_SINGLE_ELEMS];
        aL0_ = LocalTensor<float>(TPosition::A2, 0, CUBE_L0_AB_ELEMS);
        bL0_ = LocalTensor<float>(TPosition::B2, 0, CUBE_L0_AB_ELEMS);
        cL0_ = LocalTensor<float>(TPosition::CO1, 0, CUBE_L0_C_ELEMS);

        SetFlag<HardEvent::MTE1_MTE2>(0);
        SetFlag<HardEvent::M_MTE1>(CUBE_MTE1_EVENT);
        SetFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= totalTokens_) {
            return;
        }

        // 两个sibling分别写连续的两个head；AIC等两份FP32 x2都对MTE2可见。
        CrossCoreWaitFlag<0x4, PIPE_MTE2>(CUBE_INPUT_READY_FLAG);
        CrossCoreWaitFlag<0x4, PIPE_MTE2>(CUBE_INPUT_READY_FLAG + AIV1_FLAG_OFFSET);

        const int64_t oldHf32Mode = GetCtrlSpr<HF32_MODE_BIT, HF32_MODE_BIT>();
        const int64_t oldHf32TransMode = GetCtrlSpr<HF32_TRANS_MODE_BIT, HF32_TRANS_MODE_BIT>();
        SetHF32Mode(0);
        SetHF32TransMode(0);

        const uint64_t x2Base = (uint64_t)blockIdx_ * k_;
        for (uint32_t kOffset = 0; kOffset < k_; kOffset += CUBE_K_TILE) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
            ClearL1();
            CopyNdToNz(aL1_, x2Gm_[x2Base + kOffset], 1U, CUBE_K_TILE, k_);
            CopyNdToNz(bL1_, phiGm_[kOffset], N_MAX, CUBE_K_TILE, k_);
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);

            WaitFlag<HardEvent::M_MTE1>(CUBE_MTE1_EVENT);
            CopyA1ToA2(aL0_, aL1_, 1U, CUBE_K_TILE);
            CopyA1TransposeToB2(bL0_, bL1_, N_MAX, CUBE_K_TILE);
            SetFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::MTE1_M>(0);

            MmadParams params;
            params.m = 1U;
            params.n = N_MAX;
            params.k = CUBE_K_TILE;
            params.disableGemv = true;
            params.cmatrixInitVal = (kOffset == 0U);
            params.cmatrixSource = false;
            params.unitFlag = 0;
            Mmad(cL0_, aL0_, bL0_, params);
            SetFlag<HardEvent::M_MTE1>(CUBE_MTE1_EVENT);
            SetFlag<HardEvent::MTE1_MTE2>(0);
        }

        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        CopyResult(dotGm_[(uint64_t)blockIdx_ * PARTIAL_SLOT_ELEMS]);
        CrossCoreSetFlag<0x4, PIPE_FIX>(CUBE_RESULT_READY_FLAG);

        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::M_MTE1>(CUBE_MTE1_EVENT);
        WaitFlag<HardEvent::FIX_M>(0);
        SetHF32TransMode(oldHf32TransMode);
        SetHF32Mode(oldHf32Mode);
    }

private:
    static constexpr uint32_t CUBE_BLOCK = 16U;
    static constexpr uint32_t CUBE_FP32_C0 = 8U;
    static constexpr uint32_t CUBE_K_TILE = 512U;
    static constexpr uint32_t CUBE_L1_SINGLE_ELEMS = CUBE_BLOCK * CUBE_K_TILE;
    static constexpr uint32_t CUBE_L1_TOTAL_ELEMS = 2U * CUBE_L1_SINGLE_ELEMS;
    static constexpr uint32_t CUBE_L0_AB_ELEMS = CUBE_BLOCK * CUBE_K_TILE;
    static constexpr uint32_t CUBE_L0_C_ELEMS = CUBE_BLOCK * CUBE_BLOCK;
    static constexpr uint32_t CUBE_MTE1_EVENT = 3U;
    static constexpr uint32_t CUBE_BLOCK_BYTES = 32U;

    __aicore__ inline void ClearL1()
    {
        constexpr uint16_t blocks =
            static_cast<uint16_t>(CUBE_L1_SINGLE_ELEMS * sizeof(float) / CUBE_BLOCK_BYTES);
        InitConstValueParams<float> zeroParams(1, blocks, 0, 0.0f);
        InitConstValue(aL1_, zeroParams);
        InitConstValue(bL1_, zeroParams);
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void CopyNdToNz(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t m,
                                      uint32_t k, uint32_t gmStride)
    {
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = m;
        params.dValue = k;
        params.srcNdMatrixStride = 1;
        params.srcDValue = gmStride;
        params.dstNzC0Stride = CUBE_BLOCK;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        DataCopy(dst, src, params);
    }

    __aicore__ inline void CopyA1ToA2(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                      uint32_t k)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = (m + CUBE_BLOCK - 1U) / CUBE_BLOCK;
        params.kStep = k / CUBE_FP32_C0;
        params.srcStride = params.mStep;
        params.dstStride = params.mStep;
        params.ifTranspose = false;
        LoadData<float>(dst, src, params);
    }

    __aicore__ inline void CopyA1TransposeToB2(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                               uint32_t n, uint32_t k)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = k / CUBE_BLOCK;
        // A5 FP32 B2每个C0含8个元素；N=4只需一个C0。旧候选误用2导致转置布局错误。
        params.kStep = (n + CUBE_FP32_C0 - 1U) / CUBE_FP32_C0;
        params.srcStride = k / CUBE_BLOCK;
        params.dstStride = 1U;
        params.ifTranspose = true;
        LoadData<float>(dst, src, params);
    }

    __aicore__ inline void CopyResult(const GlobalTensor<float> &dst)
    {
        DataCopyCO12DstParams params;
        params.nSize = N_MAX;
        params.mSize = 1U;
        params.dstStride = PARTIAL_SLOT_ELEMS;
        params.srcStride = CUBE_BLOCK;
        params.quantPre = QuantMode_t::NoQuant;
        params.nz2ndEn = true;
        params.unitFlag = 0;
        SetFixpipeNz2ndFlag(1, 1, 1);
        DataCopy(dst, cL0_, params);
    }

    TPipe pipe_;
    TBuf<TPosition::A1> l1Buf_;
    LocalTensor<float> aL1_, bL1_, aL0_, bL0_, cL0_;
    GlobalTensor<float> phiGm_, x2Gm_, dotGm_;
    uint32_t totalTokens_ = 0U;
    uint32_t d_ = 0U;
    uint32_t k_ = 0U;
    uint32_t blockIdx_ = 0U;
};

#endif // MHC_NORM_POST_PREONLY_CUBE_H_
