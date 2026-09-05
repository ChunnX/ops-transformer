/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#ifndef MHC_PRE_BACKWARD_CUBE_COMPUTE_H
#define MHC_PRE_BACKWARD_CUBE_COMPUTE_H
#include "kernel_operator.h"
#include "mhc_pre_backward_utils.h"

namespace MhcPreBackward {
using namespace AscendC;
constexpr uint32_t C0_BASIC_M = 128U;
constexpr uint32_t C0_BASIC_N_STRIDE = 128U;
constexpr uint32_t C0_BASIC_L1_BUFFER_SIZE = 512U * 1024U;
constexpr uint32_t C0_BASIC_L1_AB_OFFSET = 2U * 128U * 256U;
constexpr uint32_t C0_BASIC_L0_BUFFER_SIZE = 64U * 1024U;
constexpr uint32_t C0_BASIC_FINAL = 3U;
constexpr uint32_t C0_BASIC_NON_FINAL = 2U;

__aicore__ inline uint64_t C0BasicCeilDiv(uint64_t x, uint64_t n)
{
    return (x + n - 1U) / n;
}
__aicore__ inline uint64_t C0BasicAlign(uint64_t x, uint64_t n)
{
    return C0BasicCeilDiv(x, n) * n;
}

class MhcPreBackwardC0Basic {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(l1Buffer_, C0_BASIC_L1_BUFFER_SIZE);
        aL1_ = l1Buffer_.Get<float>();
        bL1_ = aL1_[C0_BASIC_L1_AB_OFFSET];
        aL0_ = LocalTensor<float>(TPosition::A2, 0, C0_BASIC_L0_BUFFER_SIZE / sizeof(float));
        bL0_ = LocalTensor<float>(TPosition::B2, 0, C0_BASIC_L0_BUFFER_SIZE / sizeof(float));
        cL0_ = LocalTensor<float>(TPosition::CO1, 0, C0_BASIC_L0_BUFFER_SIZE / sizeof(float));
        SetFlag<HardEvent::MTE1_MTE2>(1);
        SetFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void Process(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                   const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                   uint32_t bGmStride)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        CopyInB1(b, n, k, bGmStride);
        for (uint32_t offsetM = 0; offsetM < m; offsetM += C0_BASIC_M) {
            if (offsetM != 0U) {
                WaitFlag<HardEvent::MTE1_MTE2>(1);
            }
            uint32_t remainM = m - offsetM;
            uint32_t curM = remainM < C0_BASIC_M ? remainM : C0_BASIC_M;
            CopyInA1(a[offsetM * k], curM, k);
            SetFlag<HardEvent::MTE2_MTE1>(2);
            WaitFlag<HardEvent::MTE2_MTE1>(2);
            WaitFlag<HardEvent::M_MTE1>(3);
            CopyInA2(curM, k);
            CopyInB2(n, k);
            SetFlag<HardEvent::MTE1_MTE2>(1);
            SetFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::FIX_M>(0);
            Compute(curM, n, k);
            SetFlag<HardEvent::M_MTE1>(3);
            SetFlag<HardEvent::M_FIX>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            CopyOut(c[offsetM * C0_BASIC_N_STRIDE], curM, n);
            CrossCoreSetFlag<MhcPreBackwardUtils::CROSS_CORE_FLAG_INDEX, PIPE_FIX>(
                MhcPreBackwardUtils::CROSS_CORE_WAIT_FLAG_C0);
            SetFlag<HardEvent::FIX_M>(0);
        }
    }

    __aicore__ inline void ProcessC1(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                     const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd)
    {
        WaitFlag<HardEvent::FIX_M>(0);
        for (uint32_t offsetK = 0; offsetK < k; offsetK += C0_BASIC_M) {
            WaitFlag<HardEvent::MTE1_MTE2>(1);
            uint32_t remainK = k - offsetK;
            uint32_t curK = remainK < C0_BASIC_M ? remainK : C0_BASIC_M;
            CopyInA1Trans(a[offsetK * aGmStride], m, curK, aGmStride);
            CopyInB1(b[offsetK * bGmStride], n, curK, bGmStride);
            SetFlag<HardEvent::MTE2_MTE1>(2);
            WaitFlag<HardEvent::MTE2_MTE1>(2);
            WaitFlag<HardEvent::M_MTE1>(3);
            CopyInA2Trans(m, curK);
            CopyInB2(n, curK);
            SetFlag<HardEvent::MTE1_MTE2>(1);
            SetFlag<HardEvent::MTE1_M>(0);
            WaitFlag<HardEvent::MTE1_M>(0);
            Compute(m, n, curK, offsetK == 0U, offsetK + curK >= k);
            SetFlag<HardEvent::M_MTE1>(3);
        }
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        if (atomicAdd) {
            SetAtomicAdd<float>();
        }
        CopyOut(c, m, n, cGmStride);
        if (atomicAdd) {
            DisableDmaAtomic();
        }
        SetFlag<HardEvent::FIX_M>(0);
    }

    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::M_MTE1>(3);
        WaitFlag<HardEvent::FIX_M>(0);
    }

private:
    __aicore__ inline void CopyInA1(const GlobalTensor<float> &src, uint32_t m, uint32_t k)
    {
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = m;
        p.dValue = k;
        p.srcNdMatrixStride = 1;
        p.srcDValue = k;
        p.dstNzC0Stride = C0BasicAlign(m, BLOCK_CUBE);
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 1;
        DataCopy(aL1_, src, p);
    }

    __aicore__ inline void CopyInB1(const GlobalTensor<float> &src, uint32_t n, uint32_t k, uint32_t gmStride)
    {
        // B在GM中为[K,N]，按非转置右矩阵的NZ格式搬入L1。
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = k;
        p.dValue = n;
        p.srcNdMatrixStride = 1;
        p.srcDValue = gmStride;
        p.dstNzC0Stride = C0BasicAlign(k, BLOCK_CUBE);
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 1;
        DataCopy(bL1_, src, p);
    }

    __aicore__ inline void CopyInA1Trans(const GlobalTensor<float> &src, uint32_t m, uint32_t k, uint32_t gmStride)
    {
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = k;
        p.dValue = m;
        p.srcNdMatrixStride = 1;
        p.srcDValue = gmStride;
        p.dstNzC0Stride = C0BasicAlign(k, BLOCK_CUBE);
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 1;
        DataCopy(aL1_, src, p);
    }

    __aicore__ inline void CopyInA2(uint32_t m, uint32_t k)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = C0BasicCeilDiv(m, BLOCK_CUBE);
        p.kStep = C0BasicCeilDiv(k, AuxGetC0Size<float>());
        p.srcStride = C0BasicCeilDiv(m, BLOCK_CUBE);
        p.dstStride = p.mStep;
        p.ifTranspose = false;
        LoadData<float>(aL0_, aL1_, p);
    }

    __aicore__ inline void CopyInA2Trans(uint32_t m, uint32_t k)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = C0BasicCeilDiv(k, BLOCK_CUBE);
        p.kStep = C0BasicCeilDiv(m, BLOCK_CUBE) * 2U;
        p.dstStride = p.kStep >> 1U;
        p.srcStride = C0BasicCeilDiv(k, BLOCK_CUBE);
        p.ifTranspose = true;
        LoadData<float>(aL0_, aL1_, p);
    }

    __aicore__ inline void CopyInB2(uint32_t n, uint32_t k)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = C0BasicCeilDiv(k, BLOCK_CUBE);
        p.kStep = C0BasicCeilDiv(n, BLOCK_CUBE) * 2U;
        p.dstStride = p.kStep >> 1U;
        p.srcStride = C0BasicCeilDiv(k, BLOCK_CUBE);
        p.ifTranspose = true;
        LoadData<float>(bL0_, bL1_, p);
    }

    __aicore__ inline void Compute(uint32_t m, uint32_t n, uint32_t k, bool initC = true, bool final = true)
    {
        MmadParams p;
        p.m = m;
        p.n = n;
        p.k = k;
        p.disableGemv = true;
        p.cmatrixInitVal = initC;
        p.cmatrixSource = false;
        p.unitFlag = final ? C0_BASIC_FINAL : C0_BASIC_NON_FINAL;
        Mmad(cL0_, aL0_, bL0_, p);
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void CopyOut(const GlobalTensor<float> &dst, uint32_t m, uint32_t n,
                                   uint32_t dstStride = C0_BASIC_N_STRIDE)
    {
        DataCopyCO12DstParams p;
        p.nSize = n;
        p.mSize = m;
        p.dstStride = dstStride;
        p.srcStride = C0BasicAlign(m, BLOCK_CUBE);
        p.quantPre = QuantMode_t::NoQuant;
        p.nz2ndEn = true;
        p.unitFlag = C0_BASIC_FINAL;
        SetFixpipeNz2ndFlag(1, 1, 1);
        DataCopy(dst, cL0_, p);
    }

    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
    LocalTensor<float> aL0_;
    LocalTensor<float> bL0_;
    LocalTensor<float> cL0_;
};
} // namespace MhcPreBackward
#endif
