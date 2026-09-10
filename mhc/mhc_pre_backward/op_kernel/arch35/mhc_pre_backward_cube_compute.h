/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN
 * Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not
 * use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
#ifndef MHC_PRE_BACKWARD_CUBE_COMPUTE_H
#define MHC_PRE_BACKWARD_CUBE_COMPUTE_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_backward_utils.h"

namespace MhcPreBackward {
using namespace AscendC;

// [RegBase-native][DAV_3510] 以下片上容量取自Ascend 950 AIC硬件规格：
// L1为512KB，L0A/L0B各64KB，L0C为256KB；双槽大小不得超过对应物理容量。
// 低阶Cube分块：baseM/baseN/baseK=128/128/32，L1每次组织64个K元素。
constexpr uint32_t CUBE_BASE_M = 128U;
constexpr uint32_t CUBE_BASE_N = 128U;
constexpr uint32_t CUBE_L1_K = 64U;
constexpr uint32_t CUBE_BASE_K = 32U;
constexpr uint32_t CUBE_BUFFER_DEPTH = MhcPreBackwardUtils::NUMBER_TWO;
constexpr uint32_t CUBE_L0_EVENT_BASE = 3U;
constexpr uint32_t CUBE_L1_BUFFER_BYTES = 512U * 1024U;
// L1均分为A0/A1/B0/B1四个128KB区域，便于MTE2和MTE1跨块重叠。
constexpr uint32_t CUBE_L1_SLOT_ELEMS = 128U * 256U;
constexpr uint32_t CUBE_B_L1_OFFSET = CUBE_BUFFER_DEPTH * CUBE_L1_SLOT_ELEMS;
constexpr uint32_t CUBE_L0_BUFFER_BYTES = 64U * 1024U;
constexpr uint32_t CUBE_L0_SLOT_ELEMS = CUBE_L0_BUFFER_BYTES / CUBE_BUFFER_DEPTH / sizeof(float);
constexpr uint32_t CUBE_L0C_BUFFER_BYTES = 256U * 1024U;
constexpr uint32_t CUBE_L0C_SLOT_ELEMS = CUBE_L0C_BUFFER_BYTES / CUBE_BUFFER_DEPTH / sizeof(float);

__aicore__ inline uint64_t BasicCubeCeilDiv(uint64_t x, uint64_t n)
{
    return (x + n - 1U) / n;
}

__aicore__ inline uint64_t BasicCubeAlign(uint64_t x, uint64_t n)
{
    return BasicCubeCeilDiv(x, n) * n;
}

class MhcPreBackwardCubeCompute {
public:
    __aicore__ inline void Init(TPipe *pipe);

    __aicore__ inline void Process(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                   const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k, uint32_t bGmStride,
                                   bool enableB1Reuse);

    __aicore__ inline void ProcessC1(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                     const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd);

    __aicore__ inline void End();

private:
    __aicore__ inline void ComputeTile(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                       const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                       uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool transA,
                                       bool atomicAdd, bool publishC0, bool reuseB1);

    __aicore__ inline void ProcessKRange(const LocalTensor<float> &currentC, const GlobalTensor<float> &a,
                                         const GlobalTensor<float> &b, uint32_t m, uint32_t n, uint32_t k,
                                         uint32_t aGmStride, uint32_t bGmStride, bool transA, bool initFirst,
                                         bool reuseB1 = false);

    __aicore__ inline void PreloadB1(const GlobalTensor<float> &b, uint32_t n, uint32_t k, uint32_t bGmStride);

    __aicore__ inline void CopyInA1(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t m,
                                    uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInA1Trans(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t m,
                                         uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInB1(const LocalTensor<float> &dst, const GlobalTensor<float> &src, uint32_t n,
                                    uint32_t k, uint32_t gmStride);

    __aicore__ inline void CopyInA2(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                    uint32_t k);

    __aicore__ inline void CopyInA2Trans(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                         uint32_t k, uint32_t kL1);

    __aicore__ inline void CopyInB2(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t n,
                                    uint32_t k, uint32_t kL1);

    __aicore__ inline void Compute(const LocalTensor<float> &c, const LocalTensor<float> &a,
                                   const LocalTensor<float> &b, uint32_t m, uint32_t n, uint32_t k, bool initC);

    __aicore__ inline void CopyOut(const GlobalTensor<float> &dst, const LocalTensor<float> &src, uint32_t m,
                                   uint32_t n, uint32_t dstStride);

    TBuf<TPosition::A1> l1Buf_;
    LocalTensor<float> aL1Local_;
    LocalTensor<float> bL1Local_;
    LocalTensor<float> aL0Local_;
    LocalTensor<float> bL0Local_;
    LocalTensor<float> cL0Local_;
    uint32_t l1PingPongID_{0};
    uint32_t l0PingPongID_{0};
    uint32_t cl0PingPongID_{0};
};

__aicore__ inline void MhcPreBackwardCubeCompute::Init(TPipe *pipe)
{
    pipe->InitBuffer(l1Buf_, CUBE_L1_BUFFER_BYTES);
    aL1Local_ = l1Buf_.Get<float>();
    bL1Local_ = aL1Local_[CUBE_B_L1_OFFSET];
    aL0Local_ = LocalTensor<float>(TPosition::A2, 0, CUBE_L0_BUFFER_BYTES / sizeof(float));
    bL0Local_ = LocalTensor<float>(TPosition::B2, 0, CUBE_L0_BUFFER_BYTES / sizeof(float));
    cL0Local_ = LocalTensor<float>(TPosition::CO1, 0, CUBE_L0C_BUFFER_BYTES / sizeof(float));

    for (uint32_t slot = 0; slot < CUBE_BUFFER_DEPTH; ++slot) {
        // 三组初始令牌分别表示L1、L0A/B和L0C对应半区可写。
        SetFlag<HardEvent::MTE1_MTE2>(slot);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + slot);
        SetFlag<HardEvent::FIX_M>(slot);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::Process(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                                          const GlobalTensor<float> &c, uint32_t m, uint32_t n,
                                                          uint32_t k, uint32_t bGmStride, bool enableB1Reuse)
{
    // MhcPreBackward的fusionSize最大为80，最多占两个K_L1分块。
    // C0的所有M tile共享同一块phi，先装入两个B-L1槽，避免每128行重复GM到L1搬运。
    bool reuseB1 = enableB1Reuse && k <= CUBE_BUFFER_DEPTH * CUBE_L1_K;
    if (reuseB1) {
        PreloadB1(b, n, k, bGmStride);
    }
    for (uint32_t offsetM = 0; offsetM < m; offsetM += CUBE_BASE_M) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < CUBE_BASE_M ? remainM : CUBE_BASE_M;
        ComputeTile(a[offsetM * k], b, c[offsetM * CUBE_BASE_N], currentM, n, k, k, bGmStride, CUBE_BASE_N, false,
                    false, true, reuseB1);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessC1(const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                                            const GlobalTensor<float> &c, uint32_t m, uint32_t n,
                                                            uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                                            uint32_t cGmStride, bool atomicAdd)
{
    ComputeTile(a, b, c, m, n, k, aGmStride, bGmStride, cGmStride, true, atomicAdd, false, false);
}

__aicore__ inline void MhcPreBackwardCubeCompute::End()
{
    for (uint32_t slot = 0; slot < CUBE_BUFFER_DEPTH; ++slot) {
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
        WaitFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + slot);
        WaitFlag<HardEvent::FIX_M>(slot);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ComputeTile(const GlobalTensor<float> &a,
                                                              const GlobalTensor<float> &b,
                                                              const GlobalTensor<float> &c, uint32_t m, uint32_t n,
                                                              uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                                              uint32_t cGmStride, bool transA, bool atomicAdd,
                                                              bool publishC0, bool reuseB1)
{
    // L0C双槽轮换，使当前tile的MMAD可与前一tile的Fixpipe写回重叠。
    uint32_t cSlot = cl0PingPongID_ & 1U;
    LocalTensor<float> currentC = cL0Local_[cSlot * CUBE_L0C_SLOT_ELEMS];
    WaitFlag<HardEvent::FIX_M>(cSlot);

    ProcessKRange(currentC, a, b, m, n, k, aGmStride, bGmStride, transA, true, reuseB1);

    SetFlag<HardEvent::M_FIX>(cSlot);
    WaitFlag<HardEvent::M_FIX>(cSlot);
    if (atomicAdd) {
        // 后续C1分段写同一gradPhi前仅等待上一条Fixpipe完成，固定AtomicAdd顺序；
        // 等待发生在当前MMAD结束后，继续保留L0C双槽和MMAD/Fixpipe重叠。
        PipeBarrier<PIPE_FIX>();
        SetAtomicAdd<float>();
    }
    CopyOut(c, currentC, m, n, cGmStride);
    if (atomicAdd) {
        SetAtomicNone();
    }
    if (publishC0) {
        // FIX流水完成当前最多128行C0写回后，按原粒度唤醒对应Vector核。
        CrossCoreSetFlag<MhcPreBackwardUtils::CROSS_CORE_FLAG_INDEX, PIPE_FIX>(
            MhcPreBackwardUtils::CROSS_CORE_WAIT_FLAG_C0);
    }
    SetFlag<HardEvent::FIX_M>(cSlot);
    ++cl0PingPongID_;
}

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessKRange(const LocalTensor<float> &currentC,
                                                                const GlobalTensor<float> &a,
                                                                const GlobalTensor<float> &b, uint32_t m, uint32_t n,
                                                                uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                                                bool transA, bool initFirst, bool reuseB1)
{
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += CUBE_L1_K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < CUBE_L1_K ? remainKL1 : CUBE_L1_K;
        uint32_t l1Slot = reuseB1 ? ((offsetKL1 / CUBE_L1_K) & 1U) : (l1PingPongID_ & 1U);
        LocalTensor<float> currentA1 = aL1Local_[l1Slot * CUBE_L1_SLOT_ELEMS];
        LocalTensor<float> currentB1 = bL1Local_[l1Slot * CUBE_L1_SLOT_ELEMS];

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

        for (uint32_t offsetKL0 = 0; offsetKL0 < currentKL1; offsetKL0 += CUBE_BASE_K) {
            uint32_t remainKL0 = currentKL1 - offsetKL0;
            uint32_t currentKL0 = remainKL0 < CUBE_BASE_K ? remainKL0 : CUBE_BASE_K;
            uint32_t l0Slot = l0PingPongID_ & 1U;
            uint32_t l0Offset = l0Slot * CUBE_L0_SLOT_ELEMS;
            LocalTensor<float> currentA2 = aL0Local_[l0Offset];
            LocalTensor<float> currentB2 = bL0Local_[l0Offset];

            WaitFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + l0Slot);
            if (transA) {
                CopyInA2Trans(currentA2, currentA1[offsetKL0 * AuxGetC0Size<float>()], m, currentKL0, currentKL1);
            } else {
                CopyInA2(currentA2, currentA1[offsetKL0 * BasicCubeAlign(m, BLOCK_CUBE)], m, currentKL0);
            }
            CopyInB2(currentB2, currentB1[offsetKL0 * AuxGetC0Size<float>()], n, currentKL0, currentKL1);
            SetFlag<HardEvent::MTE1_M>(l0Slot);
            WaitFlag<HardEvent::MTE1_M>(l0Slot);

            bool initC = initFirst && offsetKL1 == 0U && offsetKL0 == 0U;
            Compute(currentC, currentA2, currentB2, m, n, currentKL0, initC);
            SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + l0Slot);
            ++l0PingPongID_;
        }

        // MTE1完成该L1半区的最后一次读取后，允许MTE2复用此半区。
        SetFlag<HardEvent::MTE1_MTE2>(l1Slot);
        ++l1PingPongID_;
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::PreloadB1(const GlobalTensor<float> &b, uint32_t n, uint32_t k,
                                                            uint32_t bGmStride)
{
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += CUBE_L1_K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < CUBE_L1_K ? remainKL1 : CUBE_L1_K;
        uint32_t l1Slot = (offsetKL1 / CUBE_L1_K) & 1U;
        LocalTensor<float> currentB1 = bL1Local_[l1Slot * CUBE_L1_SLOT_ELEMS];

        WaitFlag<HardEvent::MTE1_MTE2>(l1Slot);
        CopyInB1(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
        SetFlag<HardEvent::MTE2_MTE1>(l1Slot);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Slot);
        SetFlag<HardEvent::MTE1_MTE2>(l1Slot);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA1(const LocalTensor<float> &dst,
                                                           const GlobalTensor<float> &src, uint32_t m, uint32_t k,
                                                           uint32_t gmStride)
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA1Trans(const LocalTensor<float> &dst,
                                                                const GlobalTensor<float> &src, uint32_t m, uint32_t k,
                                                                uint32_t gmStride)
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInB1(const LocalTensor<float> &dst,
                                                           const GlobalTensor<float> &src, uint32_t n, uint32_t k,
                                                           uint32_t gmStride)
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA2(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                           uint32_t m, uint32_t k)
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInA2Trans(const LocalTensor<float> &dst,
                                                                const LocalTensor<float> &src, uint32_t m, uint32_t k,
                                                                uint32_t kL1)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = BasicCubeCeilDiv(k, BLOCK_CUBE);
    p.kStep = BasicCubeCeilDiv(m, BLOCK_CUBE) * MhcPreBackwardUtils::NUMBER_TWO;
    p.dstStride = p.kStep >> 1U;
    p.srcStride = BasicCubeCeilDiv(kL1, BLOCK_CUBE);
    p.ifTranspose = true;
    LoadData<float>(dst, src, p);
}

__aicore__ inline void MhcPreBackwardCubeCompute::CopyInB2(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                           uint32_t n, uint32_t k, uint32_t kL1)
{
    LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = BasicCubeCeilDiv(k, BLOCK_CUBE);
    p.kStep = BasicCubeCeilDiv(n, BLOCK_CUBE) * MhcPreBackwardUtils::NUMBER_TWO;
    p.dstStride = p.kStep >> 1U;
    p.srcStride = BasicCubeCeilDiv(kL1, BLOCK_CUBE);
    p.ifTranspose = true;
    LoadData<float>(dst, src, p);
}

__aicore__ inline void MhcPreBackwardCubeCompute::Compute(const LocalTensor<float> &c, const LocalTensor<float> &a,
                                                          const LocalTensor<float> &b, uint32_t m, uint32_t n,
                                                          uint32_t k, bool initC)
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

__aicore__ inline void MhcPreBackwardCubeCompute::CopyOut(const GlobalTensor<float> &dst, const LocalTensor<float> &src,
                                                          uint32_t m, uint32_t n, uint32_t dstStride)
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
} // namespace MhcPreBackward
#endif
