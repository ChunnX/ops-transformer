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
 * \file mhc_pre_split_post_res_multi_tile.h
 * \brief KernelMhcMultiTile<T>：MULTI_TILE 路径（dimLoop>1，D 分多个 dimTile 处理）
 *
 * 每个 token 由 coresPerToken（1 或 2）个核并行处理（T < coreNum/2 时双核），
 * phi 行按 phiRowMid 对半分：core0 处理 [0, phiRowMid)，core1 处理 [phiRowMid, phiRows)。
 * 每个 token 分两轮遍历 dimTile：
 *   Phase A：逐 tile CopyIn x 并 Cast，累加平方和，循环结束后计算 invRms（双核各自全量，冗余可接受）
 *   Phase B：再次逐 tile CopyIn x 并 Cast，对本核负责的 phi 行做双缓冲流水 dot 计算
 *            （Phase A 的 xFp32 未持久化，Phase B 需从 xGm_ 重读；双核下每核只算一半 phi 行）
 *
 * xHat 本地下标规则（phiRowStart 由调用方传入）：
 *   row < N_  → row - phiRowStart          （core0 的 h_post 段，phiRowStart==0）
 *   row >= N_ → AlignUp(postCount) + (row - resStart)  （h_res 段）
 */

#ifndef MHC_PRE_SPLIT_POST_RES_KERNEL_MULTI_TILE_H_
#define MHC_PRE_SPLIT_POST_RES_KERNEL_MULTI_TILE_H_

#include "mhc_pre_split_post_res_single_tile.h"

namespace MhcPreSplitPostRes {
using namespace AscendC;

template <typename T>
class KernelMhcMultiTile : public KernelMhcBase<T> {
    using Base = KernelMhcBase<T>;
    using Base::AccumSumSq;
    using Base::AlignUp;
    using Base::alpha0_;
    using Base::alpha1_;
    using Base::biasBuf_;
    using Base::ComputeAndWriteHPost;
    using Base::ComputeAndWriteHRes;
    using Base::ComputePhiDot;
    using Base::CopyInBias;
    using Base::CopyInXTile;
    using Base::D_;
    using Base::InitCommonBuffers;
    using Base::InitGM;
    using Base::invK_;
    using Base::N_;
    using Base::normEps_;
    using Base::phiRows_;
    using Base::PrefetchPhiRow;
    using Base::sumSqBuf_;
    using Base::totalLength_;
    using Base::xFp32Buf_;
    using Base::xGm_;
    using Base::xHatBuf_;
    using Base::xInQueue_;

public:
    __aicore__ inline KernelMhcMultiTile()
    {}

    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR phiAddr, GM_ADDR alphaAddr, GM_ADDR biasAddr, GM_ADDR hPostAddr,
        GM_ADDR hResAddr, GM_ADDR workspaceAddr, TPipe *pipe, const MhcPreSplitPostResTilingData *tiling)
    {
        pipe_ = pipe;
        totalLength_ = tiling->totalLength;
        N_ = tiling->N;
        D_ = tiling->D;
        phiRows_ = tiling->phiRows;
        dimTile_ = tiling->dimTile;
        dimLoop_ = tiling->dimLoop;
        normEps_ = tiling->normEps;
        invK_ = tiling->invK;
        phiRowMid_ = tiling->phiRowMid;
        coresPerToken_ = tiling->coresPerToken;
        baseT_ = tiling->baseT;
        tailT_ = tiling->tailT;
        coreTAct_ = tiling->coreTAct;

        InitGM(xAddr, phiAddr, alphaAddr, biasAddr, hPostAddr, hResAddr);
        // doubleBufferX=true：xInQueue_ depth=2，MTE2 CopyIn(tile d+1) 与
        // Cast+AccumSumSq(tile d) 重叠；squareBuf_ 跳过 InitBuffer（原地 Mul 替代）
        InitCommonBuffers(pipe, dimTile_, tiling->reduceWorkSize, true);
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t groupIdx = blockIdx / coresPerToken_;  // token 组索引
        int64_t localIdx = blockIdx % coresPerToken_;  // 组内核索引（0 或 1）

        if (groupIdx >= coreTAct_) {
            return;
        }

        int64_t tStart = groupIdx * baseT_;
        int64_t tCount = (groupIdx == coreTAct_ - 1) ? tailT_ : baseT_;

        // localIdx==0: 处理 phi 行 [0, phiRowMid)；localIdx==1: 处理 [phiRowMid, phiRows)
        int64_t phiRowStart = (localIdx == NUM_ONE) ? phiRowMid_ : NUM_ZERO;
        int64_t phiRowEnd = (localIdx == NUM_ZERO) ? phiRowMid_ : phiRows_;

        for (int64_t t = 0; t < tCount; t++) {
            ProcessOneToken(tStart + t, phiRowStart, phiRowEnd);
        }
    }

private:
    // 将 phi 行号映射到 xHat 本地下标：
    //   row < N_  → row - phiRowStart（core0 h_post 段，phiRowStart==0）
    //   row >= N_ → AlignUp(postCount) + (row - resStart)（h_res 段）
    __aicore__ inline int64_t LocalIdxOf(int64_t row, int64_t phiRowStart) const
    {
        if (row < N_) {
            return row - phiRowStart;
        }
        int64_t postCount = (N_ > phiRowStart) ? (N_ - phiRowStart) : NUM_ZERO;
        int64_t resStart = (phiRowStart >= N_) ? phiRowStart : N_;
        return AlignUp(postCount) + (row - resStart);
    }

    // 计算 tile d 的 copyD：最后一个 tile 取尾部实际列数，其余取 dimTile_
    __aicore__ inline int64_t TileCopyD(int64_t d) const
    {
        int64_t dimOff = d * dimTile_;
        int64_t copyDRaw = (d == dimLoop_ - 1) ? (D_ - dimOff) : dimTile_;
        return copyDRaw;
    }

    __aicore__ inline void ProcessOneToken(int64_t tokenIdx, int64_t phiRowStart, int64_t phiRowEnd)
    {
        int64_t xTokenOff = tokenIdx * N_ * D_;

        // ── Phase A：双缓冲流水累加 sumSq ────────────────────────────
        //
        // 流水时序（depth=2 队列，MTE2 与 Vector 重叠）：
        //
        //   prefetch d=0                          （MTE2[0] 发出）
        //   loop d = 0 .. dimLoop_-2:
        //     prefetch d+1                        （MTE2[d+1] 发出，与下面的 V 重叠）
        //     dequeue d → Cast → AccumSumSq(d)    （V[d]，与 MTE2[d+1] 并行）
        //   dequeue dimLoop_-1 → Cast → AccumSumSq （收尾：无下一个 MTE2）
        //
        LocalTensor<float> sumSq = this->sumSqBuf_.template Get<float>();
        Duplicate(sumSq, FLOAT_ZERO, static_cast<uint32_t>(FP32_ALIGN_ELEM));
        PipeBarrier<PIPE_V>();

        // 预取 tile 0
        CopyInXTile(xTokenOff, 0, TileCopyD(0));

        for (int64_t d = 0; d < dimLoop_ - 1; d++) {
            int64_t copyD = TileCopyD(d);
            int64_t copyDNext = TileCopyD(d + 1);

            // 先发下一个 tile 的 MTE2（与当前 tile 的向量计算重叠）
            CopyInXTile(xTokenOff, (d + 1) * dimTile_, copyDNext);

            // 等待当前 tile 就绪，Cast → AccumSumSq（原地 Mul）
            LocalTensor<T> xLocal = this->xInQueue_.template DeQue<T>();
            LocalTensor<float> xFp32 = this->xFp32Buf_.template Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(N_ * copyD));
            PipeBarrier<PIPE_V>();
            this->xInQueue_.FreeTensor(xLocal);
            // AccumSumSq 末尾已无 PipeBarrier，sumSq.SetValue 是标量写，
            // 下一次循环的 CopyInXTile 不依赖 V 流水，无需额外屏障
            this->template AccumSumSq<true>(xFp32, sumSq, copyD, pipe_);
        }

        // 收尾：处理最后一个 tile（已由上一轮或首次预取发出）
        {
            int64_t copyD = TileCopyD(dimLoop_ - 1);
            LocalTensor<T> xLocal = this->xInQueue_.template DeQue<T>();
            LocalTensor<float> xFp32 = this->xFp32Buf_.template Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(N_ * copyD));
            PipeBarrier<PIPE_V>();
            this->xInQueue_.FreeTensor(xLocal);
            // sumSq.GetValue(0) 是标量读，不依赖 V 流水，无需 PipeBarrier
            this->template AccumSumSq<true>(xFp32, sumSq, copyD, pipe_);
        }

        float invRms = FLOAT_ONE / sqrt(sumSq.GetValue(NUM_ZERO) * invK_ + normEps_);

        // ── Phase B：双缓冲流水计算 xHat ─────────────────────────────
        //
        // 流水时序同 Phase A：先 prefetch d+1，再处理 d。
        // 每个 tile 内部 phiRowQue_（depth=2）已对 phi 行做双缓冲，
        // 此处外层对 x tile 的 MTE2 再加一层重叠。
        //
        LocalTensor<float> xHat = this->xHatBuf_.template Get<float>();
        Duplicate(xHat, FLOAT_ZERO, static_cast<uint32_t>(AlignUp(phiRows_)));
        PipeBarrier<PIPE_V>();

        // 预取 tile 0
        CopyInXTile(xTokenOff, 0, TileCopyD(0));

        for (int64_t d = 0; d < dimLoop_ - 1; d++) {
            int64_t dimOff = d * dimTile_;
            int64_t copyD = TileCopyD(d);
            int64_t copyDNext = TileCopyD(d + 1);

            // 先发下一个 tile 的 MTE2
            CopyInXTile(xTokenOff, (d + 1) * dimTile_, copyDNext);

            // 等待当前 tile 就绪，Cast → phi 行双缓冲流水
            LocalTensor<T> xLocal = this->xInQueue_.template DeQue<T>();
            LocalTensor<float> xFp32 = this->xFp32Buf_.template Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(N_ * copyD));
            PipeBarrier<PIPE_V>();
            this->xInQueue_.FreeTensor(xLocal);

            PrefetchPhiRow(phiRowStart, dimOff, copyD);
            for (int64_t row = phiRowStart; row < phiRowEnd - 1; row++) {
                PrefetchPhiRow(row + 1, dimOff, copyD);
                ComputePhiDot(xFp32, xHat, LocalIdxOf(row, phiRowStart), copyD, pipe_);
            }
            ComputePhiDot(xFp32, xHat, LocalIdxOf(phiRowEnd - 1, phiRowStart), copyD, pipe_);
        }

        // 收尾：处理最后一个 tile
        {
            int64_t dimOff = (dimLoop_ - 1) * dimTile_;
            int64_t copyD = TileCopyD(dimLoop_ - 1);
            LocalTensor<T> xLocal = this->xInQueue_.template DeQue<T>();
            LocalTensor<float> xFp32 = this->xFp32Buf_.template Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(N_ * copyD));
            PipeBarrier<PIPE_V>();
            this->xInQueue_.FreeTensor(xLocal);

            PrefetchPhiRow(phiRowStart, dimOff, copyD);
            for (int64_t row = phiRowStart; row < phiRowEnd - 1; row++) {
                PrefetchPhiRow(row + 1, dimOff, copyD);
                ComputePhiDot(xFp32, xHat, LocalIdxOf(row, phiRowStart), copyD, pipe_);
            }
            ComputePhiDot(xFp32, xHat, LocalIdxOf(phiRowEnd - 1, phiRowStart), copyD, pipe_);
        }
        // ComputePhiDot 末尾是 SetValue（标量写）+ FreeTensor，不发 V 指令，
        // Muls 不需要等待 V 流水 drain，无需 PipeBarrier

        // xHat 乘以 invRms，完成 RMS norm
        int64_t rowCount = phiRowEnd - phiRowStart;
        int64_t xHatUsed = (phiRowEnd <= N_) ? rowCount : LocalIdxOf(phiRowEnd - 1, phiRowStart) + 1;
        Muls(xHat, xHat, invRms, static_cast<uint32_t>(xHatUsed));
        PipeBarrier<PIPE_V>();

        LocalTensor<float> biasLocal = this->biasBuf_.template Get<float>();

        // 本核有 h_post 段（仅 core0，phiRowStart < N_）：行 [0, N_)
        if (phiRowStart < N_) {
            int64_t postCount = N_ - phiRowStart;
            CopyInBias(biasLocal, phiRowStart, postCount, pipe_);
            ComputeAndWriteHPost(tokenIdx, xHat, biasLocal, phiRowStart, postCount, pipe_);
        }

        // 本核有 h_res 段（core0: [N_, phiRowMid)；core1: [phiRowMid, phiRows)）
        int64_t resStart = (phiRowStart >= N_) ? phiRowStart : N_;
        if (resStart < phiRowEnd) {
            int64_t resCount = phiRowEnd - resStart;
            int64_t resLocalOff = LocalIdxOf(resStart, phiRowStart);
            CopyInBias(biasLocal[resLocalOff], resStart, resCount, pipe_);
            ComputeAndWriteHRes(tokenIdx, xHat, biasLocal, resStart, resLocalOff, resCount, pipe_);
        }
    }

private:
    TPipe *pipe_ = nullptr;
    int64_t dimTile_ = 0;
    int64_t dimLoop_ = 0;
    int64_t phiRowMid_ = 0;
    int64_t coresPerToken_ = 0;
    int64_t baseT_ = 0;
    int64_t tailT_ = 0;
    int64_t coreTAct_ = 0;
};

}  // namespace MhcPreSplitPostRes

#endif  // MHC_PRE_SPLIT_POST_RES_KERNEL_MULTI_TILE_H_
