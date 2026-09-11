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

constexpr uint32_t CUBE_L0_EVENT_BASE = 3U;

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
    __aicore__ inline void Init(TPipe *pipe, const MhcPreBackwardTilingData *tiling);

    __aicore__ inline void Process(const MMConfig &config, const GlobalTensor<float> &a, const GlobalTensor<float> &b,
                                   const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                   uint32_t bGmStride);

    __aicore__ inline void ProcessC1(const MMConfig &config, const GlobalTensor<float> &a,
                                     const GlobalTensor<float> &b,
                                     const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                     uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool atomicAdd);

    __aicore__ inline void End();

private:
    template <bool REUSE_B1>
    __aicore__ inline void ComputeTile(const MMConfig &config, const GlobalTensor<float> &a,
                                       const GlobalTensor<float> &b,
                                       const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k,
                                       uint32_t aGmStride, uint32_t bGmStride, uint32_t cGmStride, bool transA,
                                       bool atomicAdd);

    template <bool REUSE_B1>
    __aicore__ inline void ProcessKRange(const MMConfig &config, const LocalTensor<float> &currentC,
                                         const GlobalTensor<float> &a,
                                         const GlobalTensor<float> &b, uint32_t m, uint32_t n, uint32_t k,
                                         uint32_t aGmStride, uint32_t bGmStride, bool transA, bool initFirst);

    __aicore__ inline void PreloadB1(const MMConfig &config, const GlobalTensor<float> &b, uint32_t n, uint32_t k,
                                     uint32_t bGmStride);

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
    uint32_t cubeBufferDepth_{0};
    uint32_t cubeL1BufferElems_{0};
    uint32_t cubeL0BufferElems_{0};
    uint32_t cubeL0CBufferElems_{0};
};

__aicore__ inline void MhcPreBackwardCubeCompute::Init(TPipe *pipe, const MhcPreBackwardTilingData *tiling)
{
    cubeBufferDepth_ = tiling->cubeBufferDepth;
    cubeL1BufferElems_ = tiling->cubeL1BufferElems;
    cubeL0BufferElems_ = tiling->cubeL0BufferElems;
    cubeL0CBufferElems_ = tiling->cubeL0CBufferElems;
    pipe->InitBuffer(l1Buf_, tiling->cubeL1BufferBytes);
    aL1Local_ = l1Buf_.Get<float>();
    bL1Local_ = aL1Local_[tiling->cubeL1BOffset];
    aL0Local_ = LocalTensor<float>(TPosition::A2, 0, tiling->cubeL0BufferBytes / sizeof(float));
    bL0Local_ = LocalTensor<float>(TPosition::B2, 0, tiling->cubeL0BufferBytes / sizeof(float));
    cL0Local_ = LocalTensor<float>(TPosition::CO1, 0, tiling->cubeL0CBufferBytes / sizeof(float));

    for (uint32_t slot = 0; slot < cubeBufferDepth_; ++slot) {
        // 三组初始令牌分别表示L1、L0A/B和L0C对应半区可写。
        SetFlag<HardEvent::MTE1_MTE2>(slot);
        SetFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + slot);
        SetFlag<HardEvent::FIX_M>(slot);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::Process(const MMConfig &config, const GlobalTensor<float> &a,
                                                          const GlobalTensor<float> &b,
                                                          const GlobalTensor<float> &c, uint32_t m, uint32_t n,
                                                          uint32_t k, uint32_t bGmStride)
{
    // C0为[M=BS块,K=fusionSize]×[K=fusionSize,N=ND块]；头数4/6/8对应K=24/48/80。
    // B1两槽共256KB（单槽128KB），N最多128时B矩阵对齐后分别占16/24/40KB，均可完整驻留。
    // 因此所有合法C0规格固定预载B1，不再保留运行时开关；C1仍使用自身的分段搬运路径。
    PreloadB1(config, b, n, k, bGmStride);
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.get_baseM()) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.get_baseM() ? remainM : config.get_baseM();
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.get_baseN()) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.get_baseN() ? remainN : config.get_baseN();
            ComputeTile<true>(config, a[offsetM * k], b[offsetN],
                              c[offsetM * MhcPreBackwardUtils::ND_BLOCK_SIZE + offsetN], currentM, currentN, k, k,
                              bGmStride, MhcPreBackwardUtils::ND_BLOCK_SIZE, false, false);
        }
        CrossCoreSetFlag<MhcPreBackwardUtils::CROSS_CORE_FLAG_INDEX, PIPE_FIX>(
            MhcPreBackwardUtils::CROSS_CORE_WAIT_FLAG_C0);
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::ProcessC1(const MMConfig &config, const GlobalTensor<float> &a,
                                                            const GlobalTensor<float> &b,
                                                            const GlobalTensor<float> &c, uint32_t m, uint32_t n,
                                                            uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
                                                            uint32_t cGmStride, bool atomicAdd)
{
    for (uint32_t offsetM = 0; offsetM < m; offsetM += config.get_baseM()) {
        uint32_t remainM = m - offsetM;
        uint32_t currentM = remainM < config.get_baseM() ? remainM : config.get_baseM();
        for (uint32_t offsetN = 0; offsetN < n; offsetN += config.get_baseN()) {
            uint32_t remainN = n - offsetN;
            uint32_t currentN = remainN < config.get_baseN() ? remainN : config.get_baseN();
            ComputeTile<false>(config, a[offsetM], b[offsetN], c[offsetM * cGmStride + offsetN], currentM,
                               currentN, k, aGmStride, bGmStride, cGmStride, true, atomicAdd);
        }
    }
}

__aicore__ inline void MhcPreBackwardCubeCompute::End()
{
    for (uint32_t slot = 0; slot < cubeBufferDepth_; ++slot) {
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
        WaitFlag<HardEvent::M_MTE1>(CUBE_L0_EVENT_BASE + slot);
        WaitFlag<HardEvent::FIX_M>(slot);
    }
}

template <bool REUSE_B1>
__aicore__ inline void MhcPreBackwardCubeCompute::ComputeTile(
    const MMConfig &config, const GlobalTensor<float> &a, const GlobalTensor<float> &b,
    const GlobalTensor<float> &c, uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
    uint32_t cGmStride, bool transA, bool atomicAdd)
{
    // L0C双槽轮换，使当前tile的MMAD可与前一tile的Fixpipe写回重叠。
    uint32_t cDepth = config.get_l0CDepth();
    uint32_t cSlot = cl0PingPongID_ % cDepth;
    LocalTensor<float> currentC = cL0Local_[cSlot * cubeL0CBufferElems_];
    WaitFlag<HardEvent::FIX_M>(cSlot);

    ProcessKRange<REUSE_B1>(config, currentC, a, b, m, n, k, aGmStride, bGmStride, transA, true);

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
    SetFlag<HardEvent::FIX_M>(cSlot);
    ++cl0PingPongID_;
}

template <bool REUSE_B1>
__aicore__ inline void MhcPreBackwardCubeCompute::ProcessKRange(
    const MMConfig &config, const LocalTensor<float> &currentC, const GlobalTensor<float> &a,
    const GlobalTensor<float> &b, uint32_t m, uint32_t n, uint32_t k, uint32_t aGmStride, uint32_t bGmStride,
    bool transA, bool initFirst)
{
    uint32_t l1K = config.get_l1K();
    uint32_t baseK = config.get_baseK();
    uint32_t l1Depth = config.get_l1ADepth();
    uint32_t l0Depth = config.get_l0ADepth();
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += l1K) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < l1K ? remainKL1 : l1K;
        uint32_t l1Slot = REUSE_B1 ? ((offsetKL1 / l1K) % l1Depth) : (l1PingPongID_ % l1Depth);
        LocalTensor<float> currentA1 = aL1Local_[l1Slot * cubeL1BufferElems_];
        LocalTensor<float> currentB1 = bL1Local_[l1Slot * cubeL1BufferElems_];

        WaitFlag<HardEvent::MTE1_MTE2>(l1Slot);
        if (transA) {
            CopyInA1Trans(currentA1, a[offsetKL1 * aGmStride], m, currentKL1, aGmStride);
        } else {
            CopyInA1(currentA1, a[offsetKL1], m, currentKL1, aGmStride);
        }
        if constexpr (!REUSE_B1) {
            CopyInB1(currentB1, b[offsetKL1 * bGmStride], n, currentKL1, bGmStride);
        }
        SetFlag<HardEvent::MTE2_MTE1>(l1Slot);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Slot);

        for (uint32_t offsetKL0 = 0; offsetKL0 < currentKL1; offsetKL0 += baseK) {
            uint32_t remainKL0 = currentKL1 - offsetKL0;
            uint32_t currentKL0 = remainKL0 < baseK ? remainKL0 : baseK;
            uint32_t l0Slot = l0PingPongID_ % l0Depth;
            uint32_t l0Offset = l0Slot * cubeL0BufferElems_;
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

__aicore__ inline void MhcPreBackwardCubeCompute::PreloadB1(const MMConfig &config, const GlobalTensor<float> &b,
                                                            uint32_t n, uint32_t k, uint32_t bGmStride)
{
    for (uint32_t offsetKL1 = 0; offsetKL1 < k; offsetKL1 += config.get_l1K()) {
        uint32_t remainKL1 = k - offsetKL1;
        uint32_t currentKL1 = remainKL1 < config.get_l1K() ? remainKL1 : config.get_l1K();
        uint32_t l1Slot = (offsetKL1 / config.get_l1K()) % config.get_l1BDepth();
        LocalTensor<float> currentB1 = bL1Local_[l1Slot * cubeL1BufferElems_];

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
