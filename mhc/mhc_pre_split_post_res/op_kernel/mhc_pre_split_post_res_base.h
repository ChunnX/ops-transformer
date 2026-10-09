/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res_base.h
 * \brief KernelMhcBase<T>：SingleTile / MultiTile 共享的 GM 绑定、UB 分配和计算辅助函数
 */

#ifndef MHC_PRE_SPLIT_POST_RES_BASE_H_
#define MHC_PRE_SPLIT_POST_RES_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
// MhcPreSplitPostResTilingData 采用与 mhc_pre 相同的单份模式：op_host 宏版
// （mhc_pre_split_post_res_tiling.h 的 BEGIN_TILING_DATA_DEF）为唯一事实源，
// kernel 侧定义由编译链自动注入（入口 GET_TILING_DATA 取数）。

namespace MhcPreSplitPostRes {
using namespace AscendC;

constexpr int64_t FP32_BYTES = 4;           // float 单元素字节数
constexpr int64_t FP32_ALIGN_ELEM = 8;      // 向量操作最小对齐单位：32B / 4B = 8 个 float

// ascend950（arch35）Regbase：一条向量指令的 fp32 lane 数（寄存器宽度 256B）
constexpr uint32_t REG_VL_FP32 = AscendC::VECTOR_REG_WIDTH / static_cast<uint32_t>(sizeof(float));

constexpr float FLOAT_ZERO = 0.0f;
// RMSNorm 默认 epsilon：与 op_host/mhc_pre_split_post_res_const.h 的 MHC_DEFAULT_NORM_EPS 同源互锚
// （kernel 侧编译环境隔离无法 include host 头，改动必须两侧同步）
constexpr float NORM_EPS_DEFAULT = 1e-6f;
constexpr float FLOAT_ONE = 1.0f;
constexpr float FLOAT_TWO = 2.0f;

constexpr int32_t NUM_ZERO = 0;
constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;

// 申请 event，SetFlag 后立即 WaitFlag，形成流水线屏障
template <HardEvent event>
__aicore__ inline void SetWaitFlag(TPipe *pipe)
{
    event_t eventId = static_cast<event_t>(pipe->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
class KernelMhcBase {
protected:
    // 上取整到 FP32_ALIGN_ELEM 的整数倍（用于 UB float 数组对齐）
    __aicore__ inline int64_t AlignUp(int64_t n) const
    {
        return ((n + FP32_ALIGN_ELEM - 1) / FP32_ALIGN_ELEM) * FP32_ALIGN_ELEM;
    }

    __aicore__ inline void InitGM(
        GM_ADDR xAddr, GM_ADDR phiAddr, GM_ADDR alphaAddr, GM_ADDR biasAddr, GM_ADDR hPostAddr, GM_ADDR hResAddr)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)xAddr, static_cast<uint64_t>(totalLength_) * n_ * d_);
        phiGm_.SetGlobalBuffer((__gm__ float *)phiAddr, static_cast<uint64_t>(phiRows_) * n_ * d_);
        alphaGm_.SetGlobalBuffer((__gm__ float *)alphaAddr, NUM_TWO);  // [alpha0, alpha1]
        biasGm_.SetGlobalBuffer((__gm__ float *)biasAddr, static_cast<uint64_t>(phiRows_));
        hPostGm_.SetGlobalBuffer((__gm__ float *)hPostAddr, static_cast<uint64_t>(totalLength_) * n_);
        hResGm_.SetGlobalBuffer((__gm__ float *)hResAddr, static_cast<uint64_t>(totalLength_) * n_ * n_);
    }

    // doubleBufferX=false（默认，SINGLE_TILE）：xInQueue_ depth=1。
    // doubleBufferX=true（MULTI_TILE）：xInQueue_ depth=2，MTE2 CopyIn 与 Cast/AccumSumSq 重叠。
    __aicore__ inline void InitCommonBuffers(TPipe *pipe, int64_t dimTile, bool doubleBufferX = false)
    {
        int64_t xTileElems = n_ * dimTile;
        // xInQueue_ depth：SINGLE_TILE=1，MULTI_TILE=2（双缓冲）
        pipe->InitBuffer(xInQueue_, doubleBufferX ? NUM_TWO : NUM_ONE, static_cast<uint32_t>(xTileElems * sizeof(T)));
        // xFp32Buf_：整 token 全量 n*d 并按 tile 拼接布局（tile d 存于偏移 d*xTileElems）——
        // Regbase 归约对 xFp32 只读不覆盖，MULTI 的第二轮遍历直接复用持久化数据，
        // 免除 x 的 GM 重读（PERF-6）。
        int64_t xFp32Elems = n_ * d_;
        pipe->InitBuffer(xFp32Buf_, static_cast<uint32_t>(xFp32Elems * FP32_BYTES));
        // phiRowQue_ 深度 2：一个 tile 在 MTE2 传输，同时另一个 tile 在向量计算
        pipe->InitBuffer(phiRowQue_, NUM_TWO, static_cast<uint32_t>(xTileElems * FP32_BYTES));

        // xHatBuf_ / biasBuf_：AlignUp(phiRows_) + FP32_ALIGN_ELEM 额外 padding, （防卫padding）
        int64_t xHatAligned = AlignUp(phiRows_) + FP32_ALIGN_ELEM;
        pipe->InitBuffer(xHatBuf_, static_cast<uint32_t>(xHatAligned * FP32_BYTES));
        pipe->InitBuffer(biasBuf_, static_cast<uint32_t>(xHatAligned * FP32_BYTES));

        // sumSqBuf_：存放 ReduceSum(x^2) 的累加结果，大小 FP32_ALIGN_ELEM 个 float。
        // sumSqDotBuf_ / dotBuf_：仅结果槽（FP32_ALIGN_ELEM float）——Regbase 归约在寄存器内
        // 完成，只经 DIST_FIRST_ELEMENT_B32 落结果槽 [0]，无 work 段需求。
        int64_t reduceBufElems = FP32_ALIGN_ELEM;
        pipe->InitBuffer(sumSqBuf_, static_cast<uint32_t>(FP32_ALIGN_ELEM * FP32_BYTES));
        pipe->InitBuffer(dotBuf_, static_cast<uint32_t>(reduceBufElems * FP32_BYTES));
        pipe->InitBuffer(sumSqDotBuf_, static_cast<uint32_t>(reduceBufElems * FP32_BYTES));

        pipe->InitBuffer(hPostOutQueue_, NUM_ONE, static_cast<uint32_t>(AlignUp(n_) * FP32_BYTES));
        pipe->InitBuffer(hResOutQueue_, NUM_ONE, static_cast<uint32_t>(AlignUp(n_ * n_) * FP32_BYTES));

        // alphaBuf_：alpha[2] 标量经 GM→UB 搬运后从 UB 标量读（GlobalTensor::GetValue 属黑名单 API）。
        // MTE2_S：等 MTE2 搬运完成后再由标量单元读取。
        pipe->InitBuffer(alphaBuf_, static_cast<uint32_t>(FP32_ALIGN_ELEM * FP32_BYTES));
        LocalTensor<float> alphaLocal = alphaBuf_.Get<float>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.srcStride = 0;
        params.dstStride = 0;
        params.blockLen = static_cast<uint32_t>(NUM_TWO * FP32_BYTES);
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(alphaLocal, alphaGm_[NUM_ZERO], params, pad);
        SetWaitFlag<HardEvent::MTE2_S>(pipe);
        alpha0_ = alphaLocal.GetValue(0);
        alpha1_ = alphaLocal.GetValue(1);
    }

    // 从 xGm_[xTokenOff + dimOff] 搬入当前 token 的一个 dimTile（N 行 × copyD 列）到 xInQueue_
    __aicore__ inline void CopyInXTile(int64_t xTokenOff, int64_t dimOff, int64_t copyD)
    {
        LocalTensor<T> xLocal = xInQueue_.AllocTensor<T>();
        DataCopyExtParams cp;
        cp.blockCount = static_cast<uint16_t>(n_);
        cp.blockLen = static_cast<uint32_t>(copyD * sizeof(T));
        cp.srcStride = static_cast<uint32_t>((d_ - copyD) * sizeof(T));
        cp.dstStride = 0;
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[xTokenOff + dimOff], cp, pad);
        xInQueue_.EnQue(xLocal);
    }

    // 计算 xFp32 的元素平方和并累加到 sumSq[0]（处理当前 dimTile，N×copyD 个元素）。
    // Regbase 实现：Reg 内 Mul(x,x) + ReduceSum 段折叠 + 段间 Add 链——段内二叉、
    // 段间固定顺序的确定性归约，xFp32 只读不覆盖。N×copyD 恒为 VL 的整数倍
    // （N=4 × 白名单 D 均 128 对齐），无尾块。
    __aicore__ inline void AccumSumSq(const LocalTensor<float> &xFp32, LocalTensor<float> &sumSq, int64_t copyD,
    TPipe *pipe)
    {
        LocalTensor<float> sqDot = sumSqDotBuf_.Get<float>();
        auto srcAddr = (__ubuf__ float *)xFp32.GetPhyAddr();
        auto dstAddr = (__ubuf__ float *)sqDot.GetPhyAddr();
        const uint32_t elemCount = static_cast<uint32_t>(n_ * copyD);
        const uint16_t repeatTimes = static_cast<uint16_t>((elemCount + REG_VL_FP32 - 1U) / REG_VL_FP32);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> x;
            AscendC::Reg::RegTensor<float> prod;
            AscendC::Reg::RegTensor<float> part;
            AscendC::Reg::RegTensor<float> acc;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            // 段间累加与落盘只取 lane0（ReduceSum 结果有效 lane）
            AscendC::Reg::MaskReg maskFirst = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            AscendC::Reg::Duplicate(acc, FLOAT_ZERO, maskAll);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                // N×copyD 恒为 VL 整数倍无尾块，段掩码恒为全 1，直接用 maskAll
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x, srcAddr + i * REG_VL_FP32);
                AscendC::Reg::Mul(prod, x, x, maskAll);
                AscendC::Reg::ReduceSum(part, prod, maskAll);
                AscendC::Reg::Add(acc, acc, part, maskFirst);
            }
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, acc, maskFirst);
        }
        SetWaitFlag<HardEvent::V_S>(pipe);
        // GetValue/SetValue 是标量读写，不发 V 指令，无需 PipeBarrier
        sumSq.SetValue(0, sumSq.GetValue(0) + sqDot.GetValue(0));
    }

    // 发起 phi[row] 第 dimOff 列起 copyD 列的 MTE2 搬入，结果入队 phiRowQue_（不等待）
    __aicore__ inline void PrefetchPhiRow(int64_t row, int64_t dimOff, int64_t copyD)
    {
        LocalTensor<float> phiRow = phiRowQue_.AllocTensor<float>();
        DataCopyExtParams cp;
        cp.blockCount = static_cast<uint16_t>(n_);
        cp.blockLen = static_cast<uint32_t>(copyD * FP32_BYTES);
        cp.srcStride = static_cast<uint32_t>((d_ - copyD) * FP32_BYTES);
        cp.dstStride = 0;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(phiRow, phiGm_[row * n_ * d_ + dimOff], cp, pad);
        phiRowQue_.EnQue(phiRow);
    }

    // 从 phiRowQue_ 取出 phi 行，与 xFp32 逐元素乘后 ReduceSum，结果累加到 xHat[localIdx]。
    // Regbase：Reg 内 Mul + ReduceSum 段折叠 + 段间 Add 链（段内二叉、段间固定顺序的
    // 确定性归约）；phi 行 UB 数据只读，不原地覆盖。
    __aicore__ inline void ComputePhiDot(
        const LocalTensor<float> &xFp32, LocalTensor<float> &xHat, int64_t localIdx, int64_t copyD, TPipe *pipe)
    {
        LocalTensor<float> phiRow = phiRowQue_.DeQue<float>();
        LocalTensor<float> dot = dotBuf_.Get<float>();
        auto xAddr = (__ubuf__ float *)xFp32.GetPhyAddr();
        auto phiAddr = (__ubuf__ float *)phiRow.GetPhyAddr();
        auto dstAddr = (__ubuf__ float *)dot.GetPhyAddr();
        const uint32_t elemCount = static_cast<uint32_t>(n_ * copyD);
        const uint16_t repeatTimes = static_cast<uint16_t>((elemCount + REG_VL_FP32 - 1U) / REG_VL_FP32);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> x;
            AscendC::Reg::RegTensor<float> phi;
            AscendC::Reg::RegTensor<float> prod;
            AscendC::Reg::RegTensor<float> part;
            AscendC::Reg::RegTensor<float> acc;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            // 段间累加与落盘只取 lane0（ReduceSum 结果有效 lane）
            AscendC::Reg::MaskReg maskFirst = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            AscendC::Reg::Duplicate(acc, FLOAT_ZERO, maskAll);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + i * REG_VL_FP32);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(phi, phiAddr + i * REG_VL_FP32);
                AscendC::Reg::Mul(prod, phi, x, maskAll);
                AscendC::Reg::ReduceSum(part, prod, maskAll);
                AscendC::Reg::Add(acc, acc, part, maskFirst);
            }
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, acc, maskFirst);
        }
        SetWaitFlag<HardEvent::V_S>(pipe);
        // GetValue/SetValue 是标量读写，不发 V 指令；无需 PipeBarrier
        xHat.SetValue(localIdx, xHat.GetValue(localIdx) + dot.GetValue(0));
        phiRowQue_.FreeTensor(phiRow);
    }

    // 搬入 count bias 到 biasLocal，
    __aicore__ inline void CopyInBias(LocalTensor<float> biasLocal, int64_t biasStart, int64_t count, TPipe *pipe)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.srcStride = 0;
        params.dstStride = 0;
        params.blockLen = static_cast<uint32_t>(count * FP32_BYTES);
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(biasLocal, biasGm_[biasStart], params, pad);

        SetWaitFlag<HardEvent::MTE2_V>(pipe);
    }

    // 计算 hPostOut = 2*sigmoid(alpha0*xHat[0:postCount] + biasLocal[0:postCount])，
    // 写出到 hPostGm_[tokenIdx*N + gmRowStart : gmRowStart+postCount)
    //
    // 输出队列同步契约（标准 EnQue/DeQue 流程，与 xInQueue_/phiRowQue_ 范式一致）：
    //   EnQue       ：V 侧写完发 V→MTE3 就绪事件（替代手动 SetWaitFlag<V_MTE3>）；
    //   DeQue       ：等 V 写完成后才下发 DataCopyPad（MTE3 读）；
    //   FreeTensor  ：slot 归还 TQue，由队列内建 MTE3_V 屏障保证该 slot 下次
    //   AllocTensor 不早于在途 MTE3 读完成——跨 token 复用同一 UB buffer 无竞态。
    __aicore__ inline void ComputeAndWriteHPost(int64_t tokenIdx, const LocalTensor<float> &xHat,
        const LocalTensor<float> &biasLocal, int64_t gmRowStart, int64_t postCount, TPipe *pipe)
    {
        LocalTensor<float> hPostOut = hPostOutQueue_.AllocTensor<float>();
        Muls(hPostOut, xHat, alpha0_, static_cast<uint32_t>(postCount));
        PipeBarrier<PIPE_V>();
        Add(hPostOut, hPostOut, biasLocal, static_cast<uint32_t>(postCount));
        PipeBarrier<PIPE_V>();
        Sigmoid<float, true>(hPostOut, hPostOut, static_cast<uint32_t>(postCount));
        PipeBarrier<PIPE_V>();
        Muls(hPostOut, hPostOut, FLOAT_TWO, static_cast<uint32_t>(postCount));
        hPostOutQueue_.EnQue(hPostOut);
        LocalTensor<float> hPostCopy = hPostOutQueue_.DeQue<float>();
        DataCopyExtParams op;
        op.blockCount = 1;
        op.blockLen = static_cast<uint32_t>(postCount * FP32_BYTES);
        op.srcStride = 0;
        op.dstStride = 0;
        DataCopyPad(hPostGm_[tokenIdx * n_ + gmRowStart], hPostCopy, op);
        hPostOutQueue_.FreeTensor(hPostCopy);
    }

    // 计算 hResOut = alpha1*xHat[localOff:localOff+resCount] + biasLocal[localOff:localOff+resCount]，
    // 写出到 hResGm_[tokenIdx*N*N + (gmRowStart-N) : (gmRowStart-N)+resCount)
    //
    // 输出队列同步契约：与 ComputeAndWriteHPost 相同（EnQue/DeQue 标准流程，
    // FreeTensor 后由队列内建 MTE3_V 屏障保证跨 token 复用无竞态）。
    __aicore__ inline void ComputeAndWriteHRes(int64_t tokenIdx, const LocalTensor<float> &xHat,
        const LocalTensor<float> &biasLocal, int64_t gmRowStart, int64_t localOff, int64_t resCount, TPipe *pipe)
    {
        LocalTensor<float> hResOut = hResOutQueue_.AllocTensor<float>();
        Muls(hResOut, xHat[localOff], alpha1_, static_cast<uint32_t>(resCount));
        PipeBarrier<PIPE_V>();
        Add(hResOut, hResOut, biasLocal[localOff], static_cast<uint32_t>(resCount));
        PipeBarrier<PIPE_V>();
        hResOutQueue_.EnQue(hResOut);
        LocalTensor<float> hResCopy = hResOutQueue_.DeQue<float>();
        DataCopyExtParams op;
        op.blockCount = 1;
        op.blockLen = static_cast<uint32_t>(resCount * FP32_BYTES);
        op.srcStride = 0;
        op.dstStride = 0;
        DataCopyPad(hResGm_[tokenIdx * n_ * n_ + (gmRowStart - n_)], hResCopy, op);
        hResOutQueue_.FreeTensor(hResCopy);
    }

protected:
    GlobalTensor<T> xGm_;
    GlobalTensor<float> phiGm_;
    GlobalTensor<float> alphaGm_;
    GlobalTensor<float> biasGm_;
    GlobalTensor<float> hPostGm_;
    GlobalTensor<float> hResGm_;

    // depth=2：SINGLE_TILE 路径 InitBuffer 传 depth=1（仅用1个slot），
    // MULTI_TILE 路径传 depth=2，启用 MTE2/Vector 双缓冲流水。
    TQue<TPosition::VECIN, NUM_TWO> xInQueue_;
    TQue<TPosition::VECOUT, NUM_ONE> hPostOutQueue_;
    TQue<TPosition::VECOUT, NUM_ONE> hResOutQueue_;
    TBuf<TPosition::VECCALC> xFp32Buf_;
    TQue<TPosition::VECIN, NUM_TWO> phiRowQue_;
    TBuf<TPosition::VECCALC> xHatBuf_;
    TBuf<TPosition::VECCALC> sumSqBuf_;
    TBuf<TPosition::VECCALC> dotBuf_;
    TBuf<TPosition::VECCALC> sumSqDotBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> alphaBuf_;

    int64_t totalLength_ = 0;
    int64_t n_ = 0;
    int64_t d_ = 0;
    int64_t phiRows_ = 0;
    float normEps_ = NORM_EPS_DEFAULT;
    float alpha0_ = 1.0f;
    float alpha1_ = 1.0f;
    float invK_ = 0.0f;  // 1/(N*D)，由 tiling 传入
};

}  // namespace MhcPreSplitPostRes

#endif  // MHC_PRE_SPLIT_POST_RES_BASE_H_