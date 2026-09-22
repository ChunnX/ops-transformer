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
// ascend950（arch35）APT 编译链路不为 op_host 侧 BEGIN_TILING_DATA_DEF 宏生成 kernel 侧
// MhcPreSplitPostResTilingData 定义，此处条件引入 arch35 手写版（namespace 内声明，
// 使下方 kernel 类的 tiling 形参解析到手写结构）；910b 链路仍用宏生成版。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/mhc_pre_split_post_res_tiling_data.h"
#endif

namespace MhcPreSplitPostRes {
using namespace AscendC;

constexpr int64_t FP32_BYTES = 4;           // float 单元素字节数
constexpr int64_t FP32_ALIGN_ELEM = 8;      // 向量操作最小对齐单位：32B / 4B = 8 个 float
constexpr int64_t DATACOPY_MIN_BYTES = 32;  // DataCopy 硬件要求最小块大小

// 二叉树累加需要
// float 下每 repeat 最多处理 256B/4B = 64 个元素
constexpr uint32_t ONE_REP_FLOAT = 64u;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
// ascend950（arch35）Regbase：一条向量指令的 fp32 lane 数（寄存器宽度 256B）
constexpr uint32_t REG_VL_FP32 = AscendC::VECTOR_REG_WIDTH / static_cast<uint32_t>(sizeof(float));
#endif
constexpr uint32_t DEFAULT_BLK_STRIDE = 1u;
constexpr uint32_t DEFAULT_REP_STRIDE = 8u;
constexpr uint32_t SIXTEEN_U = 16u;
constexpr uint32_t EIGHT_U = 8u;

constexpr float FLOAT_ZERO = 0.0f;
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

    // 功能：将输入 tensor 进行二叉树折叠，直到元素数量 ≤ ONE_REP_FLOAT
    // 输出：折叠后的总元素数 totalNum
    __aicore__ inline uint32_t BinaryTreeFold(LocalTensor<float> &srcTmp, uint32_t totalNum)
    {
        while (totalNum > ONE_REP_FLOAT) {
            uint32_t halfNum = DivCeil(totalNum, SIXTEEN_U) * EIGHT_U;
            SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum - halfNum);
            BinaryRepeatParams bp;
            Add<float, false>(srcTmp, srcTmp, srcTmp[halfNum], MASK_PLACEHOLDER, 1, bp);
            PipeBarrier<PIPE_V>();
            totalNum = halfNum;
        }
        return totalNum;
    }

    __aicore__ inline void InitGM(
        GM_ADDR xAddr, GM_ADDR phiAddr, GM_ADDR alphaAddr, GM_ADDR biasAddr, GM_ADDR hPostAddr, GM_ADDR hResAddr)
    {
        xGm_.SetGlobalBuffer((__gm__ T *)xAddr, static_cast<uint64_t>(totalLength_) * N_ * D_);
        phiGm_.SetGlobalBuffer((__gm__ float *)phiAddr, static_cast<uint64_t>(phiRows_) * N_ * D_);
        alphaGm_.SetGlobalBuffer((__gm__ float *)alphaAddr, NUM_TWO);  // [alpha0, alpha1]
        biasGm_.SetGlobalBuffer((__gm__ float *)biasAddr, static_cast<uint64_t>(phiRows_));
        hPostGm_.SetGlobalBuffer((__gm__ float *)hPostAddr, static_cast<uint64_t>(totalLength_) * N_);
        hResGm_.SetGlobalBuffer((__gm__ float *)hResAddr, static_cast<uint64_t>(totalLength_) * N_ * N_);
        alpha0_ = alphaGm_.GetValue(0);
        alpha1_ = alphaGm_.GetValue(1);
    }

    // doubleBufferX=false（默认，SINGLE_TILE）：xInQueue_ depth=1，squareBuf_ 正常初始化。
    // doubleBufferX=true（MULTI_TILE）：xInQueue_ depth=2，MTE2 CopyIn 与 Cast/AccumSumSq 重叠；
    //   squareBuf_ 跳过 InitBuffer（不占 UB），AccumSumSq 改为原地 Mul 到 xFp32，净省 UB。
    __aicore__ inline void InitCommonBuffers(
        TPipe *pipe, int64_t dimTile, int64_t reduceWorkSize, bool doubleBufferX = false)
    {
        int64_t xTileElems = N_ * dimTile;
        // xInQueue_ depth：SINGLE_TILE=1，MULTI_TILE=2（双缓冲）
        pipe->InitBuffer(xInQueue_, doubleBufferX ? 2 : 1, static_cast<uint32_t>(xTileElems * sizeof(T)));
        pipe->InitBuffer(xFp32Buf_, static_cast<uint32_t>(xTileElems * FP32_BYTES));
        // phiRowQue_ 深度 2：一个 tile 在 MTE2 传输，同时另一个 tile 在向量计算
        pipe->InitBuffer(phiRowQue_, NUM_TWO, static_cast<uint32_t>(xTileElems * FP32_BYTES));

        // xHatBuf_ / biasBuf_：AlignUp(phiRows_) + FP32_ALIGN_ELEM 额外 padding, （防卫padding）
        int64_t xHatAligned = AlignUp(phiRows_) + FP32_ALIGN_ELEM;
        pipe->InitBuffer(xHatBuf_, static_cast<uint32_t>(xHatAligned * FP32_BYTES));
        pipe->InitBuffer(biasBuf_, static_cast<uint32_t>(xHatAligned * FP32_BYTES));

        // sumSqBuf_：存放 ReduceSum(x^2) 的累加结果，大小 FP32_ALIGN_ELEM 个 float
        // squareBuf_：存放 x^2 的中间结果（MULTI_TILE 跳过，改用原地 Mul 节省 UB）
        // sumSqDotBuf_ / dotBuf_：布局为 [result(FP32_ALIGN_ELEM float) | work(reduceWorkSize float)]
        // reduceBufElems : ReduceSum api所需空间，ADD+WholeReduceSum实现后未使用，暂时保存。
        int64_t reduceBufElems = FP32_ALIGN_ELEM + reduceWorkSize;
        pipe->InitBuffer(sumSqBuf_, static_cast<uint32_t>(FP32_ALIGN_ELEM * FP32_BYTES));
        if (!doubleBufferX) {
            // SINGLE_TILE：AccumSumSq 需要 squareBuf_ 暂存 x^2（xFp32 在后续 ComputePhiDot 中仍需使用）
            pipe->InitBuffer(squareBuf_, static_cast<uint32_t>(xTileElems * FP32_BYTES));
        }
        pipe->InitBuffer(dotBuf_, static_cast<uint32_t>(reduceBufElems * FP32_BYTES));
        pipe->InitBuffer(sumSqDotBuf_, static_cast<uint32_t>(reduceBufElems * FP32_BYTES));

        pipe->InitBuffer(hPostOutQueue_, 1, static_cast<uint32_t>(AlignUp(N_) * FP32_BYTES));
        pipe->InitBuffer(hResOutQueue_, 1, static_cast<uint32_t>(AlignUp(N_ * N_) * FP32_BYTES));
    }

    // 从 xGm_[xTokenOff + dimOff] 搬入当前 token 的一个 dimTile（N 行 × copyD 列）到 xInQueue_
    __aicore__ inline void CopyInXTile(int64_t xTokenOff, int64_t dimOff, int64_t copyD)
    {
        LocalTensor<T> xLocal = xInQueue_.AllocTensor<T>();
        DataCopyExtParams cp;
        cp.blockCount = static_cast<uint16_t>(N_);
        cp.blockLen = static_cast<uint32_t>(copyD * sizeof(T));
        cp.srcStride = static_cast<uint32_t>((D_ - copyD) * sizeof(T));
        cp.dstStride = 0;
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[xTokenOff + dimOff], cp, pad);
        xInQueue_.EnQue(xLocal);
    }

    // 计算 xFp32 的元素平方和并累加到 sumSq[0]（处理当前 dimTile，N×copyD 个元素）
    //
    // inPlace=false（默认，SINGLE_TILE）：将 x^2 写入 squareBuf_，xFp32 内容保持不变，
    //   供后续 ComputePhiDot 复用。
    // inPlace=true（MULTI_TILE Phase A）：将 x^2 原地写回 xFp32（squareBuf_ 未初始化，不可用）。
    //   调用方在此之后立即 FreeTensor(xLocal) 并不再读 xFp32，原地覆盖安全。
    //
    // ascend950（arch35）走 Regbase 实现：Reg 内 Mul(x,x) + ReduceSum 段折叠 + 段间 Add 链，
    // 段内二叉、段间固定顺序的确定性归约；xFp32 只读不覆盖，
    // INPLACE 模板参数仅为兼容调用点保留。N×copyD 恒为 VL 的整数倍（N=4 × 白名单 D 均 128 对齐），无尾块。
    template <bool INPLACE>
    __aicore__ inline void AccumSumSq(LocalTensor<float> &xFp32, LocalTensor<float> &sumSq, int64_t copyD, TPipe *pipe)
    {
        LocalTensor<float> sqDot = sumSqDotBuf_.Get<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        auto srcAddr = (__ubuf__ float *)xFp32.GetPhyAddr();
        auto dstAddr = (__ubuf__ float *)sqDot.GetPhyAddr();
        const uint32_t elemCount = static_cast<uint32_t>(N_ * copyD);
        uint32_t sreg = elemCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((elemCount + REG_VL_FP32 - 1U) / REG_VL_FP32);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> x, prod, part, acc;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            // 段间累加与落盘只取 lane0（ReduceSum 结果有效 lane），对齐 fkle ReduceLoop 写法
            AscendC::Reg::MaskReg maskFirst = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            AscendC::Reg::Duplicate(acc, FLOAT_ZERO, maskAll);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = AscendC::Reg::UpdateMask<float>(sreg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x, srcAddr + i * REG_VL_FP32);
                AscendC::Reg::Mul(prod, x, x, mask);
                AscendC::Reg::ReduceSum(part, prod, mask);
                AscendC::Reg::Add(acc, acc, part, maskFirst);
            }
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, acc, maskFirst);
        }
        SetWaitFlag<HardEvent::V_S>(pipe);
        // GetValue/SetValue 是标量读写，不发 V 指令，无需 PipeBarrier
        sumSq.SetValue(0, sumSq.GetValue(0) + sqDot.GetValue(0));
#else
        uint32_t totalNum = static_cast<uint32_t>(N_ * copyD);
        LocalTensor<float> srcTmp;
        if constexpr (!INPLACE) {
            // SINGLE_TILE：用 squareBuf_ 暂存 x^2，保留 xFp32 原始值
            LocalTensor<float> square = squareBuf_.Get<float>();
            Mul(square, xFp32, xFp32, static_cast<uint32_t>(N_ * copyD));
            PipeBarrier<PIPE_V>();
            // 二叉树折叠 + WholeReduceSum，与 ComputePhiDot 保持一致
            SetMaskCount();
            srcTmp = square;
            // 调用通用二叉树折叠
            totalNum = BinaryTreeFold(srcTmp, totalNum);
        } else {
            // MULTI_TILE Phase A：原地 Mul，xFp32 被覆盖为 x^2；squareBuf_ 不访问
            Mul(xFp32, xFp32, xFp32, static_cast<uint32_t>(N_ * copyD));
            PipeBarrier<PIPE_V>();
            // 二叉树折叠 + WholeReduceSum，与 ComputePhiDot 保持一致
            SetMaskCount();
            srcTmp = xFp32;
            // 调用通用二叉树折叠
            totalNum = BinaryTreeFold(srcTmp, totalNum);
        }
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum);
        WholeReduceSum<float, false>(
            sqDot, srcTmp, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        SetWaitFlag<HardEvent::V_S>(pipe);
        ResetMask();
        SetMaskNorm();
        // GetValue/SetValue 是标量读写，不发 V 指令，无需 PipeBarrier
        sumSq.SetValue(0, sumSq.GetValue(0) + sqDot.GetValue(0));
#endif
    }

    // 发起 phi[row] 第 dimOff 列起 copyD 列的 MTE2 搬入，结果入队 phiRowQue_（不等待）
    __aicore__ inline void PrefetchPhiRow(int64_t row, int64_t dimOff, int64_t copyD)
    {
        LocalTensor<float> phiRow = phiRowQue_.AllocTensor<float>();
        DataCopyExtParams cp;
        cp.blockCount = static_cast<uint16_t>(N_);
        cp.blockLen = static_cast<uint32_t>(copyD * FP32_BYTES);
        cp.srcStride = static_cast<uint32_t>((D_ - copyD) * FP32_BYTES);
        cp.dstStride = 0;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(phiRow, phiGm_[row * N_ * D_ + dimOff], cp, pad);
        phiRowQue_.EnQue(phiRow);
    }

    // 从 phiRowQue_ 取出 phi 行，与 xFp32 逐元素乘后 ReduceSum，结果累加到 xHat[localIdx]
    // 910b：二叉树折叠 + WholeReduceSum，全程二叉树累加，规避顺序累加的精度误差。
    // ascend950（arch35）Regbase：Reg 内 Mul + ReduceSum 段折叠 + 段间 Add 链（段内二叉、
    // 段间固定顺序的确定性归约）；phi 行 UB 数据只读，不再原地覆盖。
    __aicore__ inline void ComputePhiDot(
        LocalTensor<float> &xFp32, LocalTensor<float> &xHat, int64_t localIdx, int64_t copyD, TPipe *pipe)
    {
        LocalTensor<float> phiRow = phiRowQue_.DeQue<float>();
        LocalTensor<float> dot = dotBuf_.Get<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        auto xAddr = (__ubuf__ float *)xFp32.GetPhyAddr();
        auto phiAddr = (__ubuf__ float *)phiRow.GetPhyAddr();
        auto dstAddr = (__ubuf__ float *)dot.GetPhyAddr();
        const uint32_t elemCount = static_cast<uint32_t>(N_ * copyD);
        uint32_t sreg = elemCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((elemCount + REG_VL_FP32 - 1U) / REG_VL_FP32);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> x, phi, prod, part, acc;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            // 段间累加与落盘只取 lane0（ReduceSum 结果有效 lane），对齐 fkle ReduceLoop 写法
            AscendC::Reg::MaskReg maskFirst = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            AscendC::Reg::Duplicate(acc, FLOAT_ZERO, maskAll);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = AscendC::Reg::UpdateMask<float>(sreg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x, xAddr + i * REG_VL_FP32);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(phi, phiAddr + i * REG_VL_FP32);
                AscendC::Reg::Mul(prod, phi, x, mask);
                AscendC::Reg::ReduceSum(part, prod, mask);
                AscendC::Reg::Add(acc, acc, part, maskFirst);
            }
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, acc, maskFirst);
        }
        SetWaitFlag<HardEvent::V_S>(pipe);
        // GetValue/SetValue 是标量读写，不发 V 指令；无需 PipeBarrier
        xHat.SetValue(localIdx, xHat.GetValue(localIdx) + dot.GetValue(0));
        phiRowQue_.FreeTensor(phiRow);
#else
        Mul(phiRow, phiRow, xFp32, static_cast<uint32_t>(N_ * copyD));
        PipeBarrier<PIPE_V>();

        // 二叉树折叠：每轮将后半段加到前半段，直至剩余 ≤ ONE_REP_FLOAT 个元素
        // phiRow 空间 = N_*D floats（由 phiRowQue_ 保证），原地操作安全
        SetMaskCount();
        uint32_t totalNum = static_cast<uint32_t>(N_ * copyD);
        LocalTensor<float> srcTmp = phiRow;
        // 调用通用二叉树折叠
        totalNum = BinaryTreeFold(srcTmp, totalNum);

        // 最后一轮：≤ 64 个元素，WholeReduceSum 一次收尾（全二叉树）
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum);
        WholeReduceSum<float, false>(
            dot, srcTmp, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        SetWaitFlag<HardEvent::V_S>(pipe);
        ResetMask();
        SetMaskNorm();

        // GetValue/SetValue 是标量读写，不发 V 指令；无需 PipeBarrier
        xHat.SetValue(localIdx, xHat.GetValue(localIdx) + dot.GetValue(0));
        phiRowQue_.FreeTensor(phiRow);
#endif
    }

    // 搬入 count bias 到 biasLocal，
    __aicore__ inline void CopyInBias(LocalTensor<float> biasLocal, int64_t biasStart, int64_t count, TPipe *pipe)
    {
        int64_t available = phiRows_ - biasStart;
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
    __aicore__ inline void ComputeAndWriteHPost(int64_t tokenIdx, LocalTensor<float> &xHat,
        LocalTensor<float> &biasLocal, int64_t gmRowStart, int64_t postCount, TPipe *pipe)
    {
        LocalTensor<float> hPostOut = hPostOutQueue_.AllocTensor<float>();
        Muls(hPostOut, xHat, alpha0_, static_cast<uint32_t>(postCount));
        PipeBarrier<PIPE_V>();
        Add(hPostOut, hPostOut, biasLocal, static_cast<uint32_t>(postCount));
        PipeBarrier<PIPE_V>();
        Sigmoid<float, true>(hPostOut, hPostOut, static_cast<uint32_t>(postCount));
        PipeBarrier<PIPE_V>();
        Muls(hPostOut, hPostOut, FLOAT_TWO, static_cast<uint32_t>(postCount));
        SetWaitFlag<HardEvent::V_MTE3>(pipe);
        DataCopyExtParams op;
        op.blockCount = 1;
        op.blockLen = static_cast<uint32_t>(postCount * FP32_BYTES);
        op.srcStride = 0;
        op.dstStride = 0;
        DataCopyPad(hPostGm_[tokenIdx * N_ + gmRowStart], hPostOut, op);
        hPostOutQueue_.FreeTensor(hPostOut);
    }

    // 计算 hResOut = alpha1*xHat[localOff:localOff+resCount] + biasLocal[localOff:localOff+resCount]，
    // 写出到 hResGm_[tokenIdx*N*N + (gmRowStart-N) : (gmRowStart-N)+resCount)
    __aicore__ inline void ComputeAndWriteHRes(int64_t tokenIdx, LocalTensor<float> &xHat,
        LocalTensor<float> &biasLocal, int64_t gmRowStart, int64_t localOff, int64_t resCount, TPipe *pipe)
    {
        LocalTensor<float> hResOut = hResOutQueue_.AllocTensor<float>();
        Muls(hResOut, xHat[localOff], alpha1_, static_cast<uint32_t>(resCount));
        PipeBarrier<PIPE_V>();
        Add(hResOut, hResOut, biasLocal[localOff], static_cast<uint32_t>(resCount));
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE3>(pipe);
        DataCopyExtParams op;
        op.blockCount = 1;
        op.blockLen = static_cast<uint32_t>(resCount * FP32_BYTES);
        op.srcStride = 0;
        op.dstStride = 0;
        DataCopyPad(hResGm_[tokenIdx * N_ * N_ + (gmRowStart - N_)], hResOut, op);
        hResOutQueue_.FreeTensor(hResOut);
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
    TBuf<TPosition::VECCALC> squareBuf_;
    TBuf<TPosition::VECCALC> dotBuf_;
    TBuf<TPosition::VECCALC> sumSqDotBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;

    int64_t totalLength_ = 0;
    int64_t N_ = 0;
    int64_t D_ = 0;
    int64_t phiRows_ = 0;
    float normEps_ = 1e-6f;
    float alpha0_ = 1.0f;
    float alpha1_ = 1.0f;
    float invK_ = 0.0f;  // 1/(N*D)，由 tiling 传入
};

}  // namespace MhcPreSplitPostRes

#endif  // MHC_PRE_SPLIT_POST_RES_BASE_H_