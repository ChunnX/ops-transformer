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

constexpr uint32_t BASIC_CUBE_M = 128U;
constexpr uint32_t BASIC_CUBE_N_STRIDE = 128U;
constexpr uint32_t BASIC_CUBE_K_L1 = 64U;
constexpr uint32_t BASIC_CUBE_K_L0 = 32U;
constexpr uint32_t BASIC_CUBE_BUFFER_NUM = 2U;
constexpr uint32_t BASIC_CUBE_L0_EVENT_BASE = 3U;
constexpr uint32_t BASIC_CUBE_L1_BUFFER_SIZE = 512U * 1024U;
// L1均分为A0/A1/B0/B1四个128KB区域，便于MTE2和MTE1跨块重叠。
constexpr uint32_t BASIC_CUBE_L1_SLOT_ELEMS = 128U * 256U;
constexpr uint32_t BASIC_CUBE_B_L1_OFFSET = BASIC_CUBE_BUFFER_NUM * BASIC_CUBE_L1_SLOT_ELEMS;
constexpr uint32_t BASIC_CUBE_L0_BUFFER_SIZE = 64U * 1024U;
constexpr uint32_t BASIC_CUBE_L0_SLOT_ELEMS = BASIC_CUBE_L0_BUFFER_SIZE / BASIC_CUBE_BUFFER_NUM / sizeof(float);
constexpr uint32_t BASIC_CUBE_L0C_BUFFER_SIZE = 256U * 1024U;
constexpr uint32_t BASIC_CUBE_L0C_SLOT_ELEMS =
    BASIC_CUBE_L0C_BUFFER_SIZE / BASIC_CUBE_BUFFER_NUM / sizeof(float);

__aicore__ inline uint64_t BasicCubeCeilDiv(uint64_t x, uint64_t n)
{
    return (x + n - 1U) / n;
}

__aicore__ inline uint64_t BasicCubeAlign(uint64_t x, uint64_t n)
{
    return BasicCubeCeilDiv(x, n) * n;
}

class MhcPreBackwardC0Basic {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(l1Buffer_, BASIC_CUBE_L1_BUFFER_SIZE);
        aL1_ = l1Buffer_.Get<float>();
        bL1_ = aL1_[BASIC_CUBE_B_L1_OFFSET];
        aL0_ = LocalTensor<float>(TPosition::A2, 0, BASIC_CUBE_L0_BUFFER_SIZE / sizeof(float));
        bL0_ = LocalTensor<float>(TPosition::B2, 0, BASIC_CUBE_L0_BUFFER_SIZE / sizeof(float));
        cL0_ = LocalTensor<float>(TPosition::CO1, 0, BASIC_CUBE_L0C_BUFFER_SIZE / sizeof(float));

        for (uint32_t slot = 0; slot < BASIC_CUBE_BUFFER_NUM; ++slot) {
            // 三组初始令牌分别表示L1、L0A/B和L0C对应半区可写。
            SetFlag<HardEvent::MTE1_MTE2>(slot);
            SetFlag<HardEvent::M_MTE1>(BASIC_CUBE_L0_EVENT_BASE + slot);
            SetFlag<HardEvent::FIX_M>(slot);
        }
    }

    __aicore__ inline void Process(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                   const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                   uint32_t bGmStride, bool enableB1Reuse)
    {
        // MhcPreBackward的fusionSize最大为80，最多占两个K_L1分块。
        // C0的所有M tile共享同一块phi，先装入两个B-L1槽，避免每128行重复GM到L1搬运。
        bool reuseB1 = enableB1Reuse && k <= BASIC_CUBE_BUFFER_NUM * BASIC_CUBE_K_L1;
        if (reuseB1) {
            PreloadB1(b, n, k, bGmStride);
        }
        for (uint32_t offsetM = 0; offsetM < m; offsetM += BASIC_CUBE_M) {
            uint32_t remainM = m - offsetM;
            uint32_t currentM = remainM < BASIC_CUBE_M ? remainM : BASIC_CUBE_M;
            ComputeTile(a[offsetM * k], b, c[offsetM * BASIC_CUBE_N_STRIDE], currentM, n, k, k,
                        bGmStride, BASIC_CUBE_N_STRIDE, false, false, true, reuseB1);
        }
    }

    __aicore__ inline void ProcessC1(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                     const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd)
    {
        ComputeTile(a, b, c, m, n, k, aGmStride, bGmStride, cGmStride, true, atomicAdd, false, false);
    }

    __aicore__ inline void End()
    {
        for (uint32_t slot = 0; slot < BASIC_CUBE_BUFFER_NUM; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
            WaitFlag<HardEvent::M_MTE1>(BASIC_CUBE_L0_EVENT_BASE + slot);
            WaitFlag<HardEvent::FIX_M>(slot);
        }
    }

private:
    __aicore__ inline void ComputeTile(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                       const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                       uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool transA,
                                       bool atomicAdd, bool publishC0, bool reuseB1)
    {
        // 最优组合：L0A/L0B保留ping-pong，L0C固定使用第0槽。
        uint32_t cSlot = 0U;
        LocalTensor<float> currentC = cL0_[cSlot * BASIC_CUBE_L0C_SLOT_ELEMS];
        WaitFlag<HardEvent::FIX_M>(cSlot);

        for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += BASIC_CUBE_K_L1) {
            uint32_t remainKL1 = k - offsetKL1;
            uint32_t currentKL1 = remainKL1 < BASIC_CUBE_K_L1 ? remainKL1 : BASIC_CUBE_K_L1;
            uint32_t l1Slot =
                reuseB1 ? ((offsetKL1 / BASIC_CUBE_K_L1) & 1U) : (l1LoopCount_ & 1U);
            LocalTensor<float> currentA1 = aL1_[l1Slot * BASIC_CUBE_L1_SLOT_ELEMS];
            LocalTensor<float> currentB1 = bL1_[l1Slot * BASIC_CUBE_L1_SLOT_ELEMS];

            WaitFlag<HardEvent::MTE1_MTE2>(l1Slot);
            if (transA) {
                CopyInA1Trans(currentA1, a[offsetKL1 * aGmStride], m, currentKL1, aGmStride);
            } else {
                CopyInA1(currentA1, a[offsetKL1], m, currentKL1, aGmStride);
            }
            if (!reuseB1) {
                CopyInB1(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1Slot);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Slot);

            for (uint32_t offsetKL0 = 0; offsetKL0 < currentKL1; offsetKL0 += BASIC_CUBE_K_L0) {
                uint32_t remainKL0 = currentKL1 - offsetKL0;
                uint32_t currentKL0 = remainKL0 < BASIC_CUBE_K_L0 ? remainKL0 : BASIC_CUBE_K_L0;
                uint32_t l0Slot = l0LoopCount_ & 1U;
                uint32_t l0Offset = l0Slot * BASIC_CUBE_L0_SLOT_ELEMS;
                LocalTensor<float> currentA2 = aL0_[l0Offset];
                LocalTensor<float> currentB2 = bL0_[l0Offset];

                WaitFlag<HardEvent::M_MTE1>(BASIC_CUBE_L0_EVENT_BASE + l0Slot);
                if (transA) {
                    CopyInA2Trans(currentA2, currentA1[offsetKL0 * AuxGetC0Size<float>()], m, currentKL0,
                                  currentKL1);
                } else {
                    CopyInA2(currentA2, currentA1[offsetKL0 * BasicCubeAlign(m, BLOCK_CUBE)], m, currentKL0);
                }
                CopyInB2(currentB2, currentB1[offsetKL0 * AuxGetC0Size<float>()], n, currentKL0, currentKL1);
                SetFlag<HardEvent::MTE1_M>(l0Slot);
                WaitFlag<HardEvent::MTE1_M>(l0Slot);

                bool initC = (offsetKL1 == 0U && offsetKL0 == 0U);
                Compute(currentC, currentA2, currentB2, m, n, currentKL0, initC);
                SetFlag<HardEvent::M_MTE1>(BASIC_CUBE_L0_EVENT_BASE + l0Slot);
                ++l0LoopCount_;
            }

            // MTE1完成该L1半区的最后一次读取后，允许MTE2复用此半区。
            SetFlag<HardEvent::MTE1_MTE2>(l1Slot);
            ++l1LoopCount_;
        }

        SetFlag<HardEvent::M_FIX>(cSlot);
        WaitFlag<HardEvent::M_FIX>(cSlot);
        if (atomicAdd) {
            SetAtomicAdd<float>();
        }
        CopyOut(c, currentC, m, n, cGmStride);
        if (atomicAdd) {
            DisableDmaAtomic();
        }
        if (publishC0) {
            // FIX流水完成当前最多128行C0写回后，按原粒度唤醒对应Vector核。
            CrossCoreSetFlag<MhcPreBackwardUtils::CROSS_CORE_FLAG_INDEX, PIPE_FIX>(
                MhcPreBackwardUtils::CROSS_CORE_WAIT_FLAG_C0);
        }
        SetFlag<HardEvent::FIX_M>(cSlot);
        ++l0cLoopCount_;
    }

    __aicore__ inline void PreloadB1(const GlobalTensor<float> &b, uint32_t n, uint32_t k,
                                     uint32_t bGmStride)
    {
        for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += BASIC_CUBE_K_L1) {
            uint32_t remainKL1 = k - offsetKL1;
            uint32_t currentKL1 = remainKL1 < BASIC_CUBE_K_L1 ? remainKL1 : BASIC_CUBE_K_L1;
            uint32_t l1Slot = (offsetKL1 / BASIC_CUBE_K_L1) & 1U;
            LocalTensor<float> currentB1 = bL1_[l1Slot * BASIC_CUBE_L1_SLOT_ELEMS];

            WaitFlag<HardEvent::MTE1_MTE2>(l1Slot);
            CopyInB1(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
            SetFlag<HardEvent::MTE2_MTE1>(l1Slot);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Slot);
            SetFlag<HardEvent::MTE1_MTE2>(l1Slot);
        }
    }

    __aicore__ inline void CopyInA1(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t m,
                                    uint32_t k, uint32_t gmStride)
    {
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = m;
        p.dValue = k;
        p.srcNdMatrixStride = 1;
        p.srcDValue = gmStride;
        p.dstNzC0Stride = BasicCubeAlign(m, BLOCK_CUBE);
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 1;
        DataCopy(dst, src, p);
    }

    __aicore__ inline void CopyInA1Trans(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t m,
                                         uint32_t k, uint32_t gmStride)
    {
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = k;
        p.dValue = m;
        p.srcNdMatrixStride = 1;
        p.srcDValue = gmStride;
        p.dstNzC0Stride = BasicCubeAlign(k, BLOCK_CUBE);
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 1;
        DataCopy(dst, src, p);
    }

    __aicore__ inline void CopyInB1(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t n,
                                    uint32_t k, uint32_t gmStride)
    {
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = k;
        p.dValue = n;
        p.srcNdMatrixStride = 1;
        p.srcDValue = gmStride;
        p.dstNzC0Stride = BasicCubeAlign(k, BLOCK_CUBE);
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 1;
        DataCopy(dst, src, p);
    }

    __aicore__ inline void CopyInA2(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                    uint32_t k)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = BasicCubeCeilDiv(m, BLOCK_CUBE);
        p.kStep = BasicCubeCeilDiv(k, AuxGetC0Size<float>());
        p.srcStride = BasicCubeCeilDiv(m, BLOCK_CUBE);
        p.dstStride = p.mStep;
        p.ifTranspose = false;
        LoadData<float>(dst, src, p);
    }

    __aicore__ inline void CopyInA2Trans(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                         uint32_t k, uint32_t kL1)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = BasicCubeCeilDiv(k, BLOCK_CUBE);
        p.kStep = BasicCubeCeilDiv(m, BLOCK_CUBE) * 2U;
        p.dstStride = p.kStep >> 1U;
        p.srcStride = BasicCubeCeilDiv(kL1, BLOCK_CUBE);
        p.ifTranspose = true;
        LoadData<float>(dst, src, p);
    }

    __aicore__ inline void CopyInB2(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t n,
                                    uint32_t k, uint32_t kL1)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = BasicCubeCeilDiv(k, BLOCK_CUBE);
        p.kStep = BasicCubeCeilDiv(n, BLOCK_CUBE) * 2U;
        p.dstStride = p.kStep >> 1U;
        p.srcStride = BasicCubeCeilDiv(kL1, BLOCK_CUBE);
        p.ifTranspose = true;
        LoadData<float>(dst, src, p);
    }

    __aicore__ inline void Compute(const LocalTensor<float> &c, const LocalTensor<float> &a,
                                   const LocalTensor<float> &b, uint32_t m, uint32_t n, uint32_t k, bool initC)
    {
        MmadParams p;
        p.m = m;
        p.n = n;
        p.k = k;
        p.disableGemv = true;
        p.cmatrixInitVal = initC;
        p.cmatrixSource = false;
        // unitFlag=0，L0C的生产/消费完全由M_FIX与FIX_M事件管理。
        p.unitFlag = 0;
        Mmad(c, a, b, p);
    }

    __aicore__ inline void CopyOut(const GlobalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                   uint32_t n, uint32_t dstStride)
    {
        DataCopyCO12DstParams p;
        p.nSize = n;
        p.mSize = m;
        p.dstStride = dstStride;
        p.srcStride = BasicCubeAlign(m, BLOCK_CUBE);
        p.quantPre = QuantMode_t::NoQuant;
        p.nz2ndEn = true;
        p.unitFlag = 0;
        SetFixpipeNz2ndFlag(1, 1, 1);
        DataCopy(dst, src, p);
    }

    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
    LocalTensor<float> aL0_;
    LocalTensor<float> bL0_;
    LocalTensor<float> cL0_;
    uint32_t l1LoopCount_{0};
    uint32_t l0LoopCount_{0};
    uint32_t l0cLoopCount_{0};
};
} // namespace MhcPreBackward
#endif
