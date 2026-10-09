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
 * 每个 token 由 coresPerToken（1 或 2）个核并行处理（T×2 ≤ coreNum 时双核），
 * phi 行按 phiRowMid 对半分：core0 处理 [0, phiRowMid)，core1 处理 [phiRowMid, phiRows)。
 * 每个 token 分两轮遍历 dimTile（第一轮累加平方和求 invRms → 第二轮对本核负责的
 * phi 行做双缓冲流水 dot；双核下每核只算一半 phi 行，第一轮双核各自全量冗余可接受）。
 * 第一轮的 Cast 按 tile 偏移拼接持久化整 token xFp32（Regbase 归约只读不覆盖），
 * 第二轮直接切片复用、无 GM 重读（PERF-6）。
 *
 * xHat 本地下标规则（phiRowStart 由调用方传入）：
 *   row < n_  → row - phiRowStart          （core0 的 h_post 段，phiRowStart==0）
 *   row >= n_ → AlignUp(postCount) + (row - resStart)  （h_res 段）
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
    using Base::d_;
    using Base::InitCommonBuffers;
    using Base::InitGM;
    using Base::invK_;
    using Base::n_;
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
        n_ = tiling->n;
        d_ = tiling->d;
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
        // doubleBufferX=true：xInQueue_ depth=2，MTE2 CopyIn(tile d+1) 与 Cast+AccumSumSq(tile d) 重叠
        InitCommonBuffers(pipe, dimTile_, true);
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
    //   row < n_  → row - phiRowStart（core0 h_post 段，phiRowStart==0）
    //   row >= n_ → AlignUp(postCount) + (row - resStart)（h_res 段）
    __aicore__ inline int64_t LocalIdxOf(int64_t row, int64_t phiRowStart) const
    {
        if (row < n_) {
            return row - phiRowStart;
        }
        int64_t postCount = (n_ > phiRowStart) ? (n_ - phiRowStart) : NUM_ZERO;
        int64_t resStart = (phiRowStart >= n_) ? phiRowStart : n_;
        return AlignUp(postCount) + (row - resStart);
    }

    // 计算 tile d 的 copyD：最后一个 tile 取尾部实际列数，其余取 dimTile_
    __aicore__ inline int64_t TileCopyD(int64_t d) const
    {
        return (d == dimLoop_ - 1) ? (d_ - d * dimTile_) : dimTile_;
    }

    __aicore__ inline void ProcessOneToken(int64_t tokenIdx, int64_t phiRowStart, int64_t phiRowEnd)
    {
        int64_t xTokenOff = tokenIdx * n_ * d_;

        // sumSq 双缓冲流水累加（第一遍遍历 dimTile）：tile 预取与向量计算重叠
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
        LocalTensor<float> xFp32Base = this->xFp32Buf_.template Get<float>();

        for (int64_t d = 0; d < dimLoop_ - 1; d++) {
            int64_t copyD = TileCopyD(d);
            int64_t copyDNext = TileCopyD(d + 1);

            // 先发下一个 tile 的 MTE2（与当前 tile 的向量计算重叠）
            CopyInXTile(xTokenOff, (d + 1) * dimTile_, copyDNext);

            // 等待当前 tile 就绪，Cast → AccumSumSq；Cast 写入 tile 偏移 d*n*dimTile
            // （整 token 拼接持久化，供第二轮遍历复用）
            LocalTensor<float> xFp32 = xFp32Base[d * dimTile_ * n_];
            LocalTensor<T> xLocal = this->xInQueue_.template DeQue<T>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(n_ * copyD));
            PipeBarrier<PIPE_V>();
            this->xInQueue_.FreeTensor(xLocal);
            // AccumSumSq 末尾已无 PipeBarrier，sumSq.SetValue 是标量写，
            // 下一次循环的 CopyInXTile 不依赖 V 流水，无需额外屏障
            this->AccumSumSq(xFp32, sumSq, copyD, pipe_);
        }

        // 收尾：处理最后一个 tile（已由上一轮或首次预取发出）
        {
            int64_t copyD = TileCopyD(dimLoop_ - 1);
            LocalTensor<float> xFp32 = xFp32Base[(dimLoop_ - 1) * dimTile_ * n_];
            LocalTensor<T> xLocal = this->xInQueue_.template DeQue<T>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(n_ * copyD));
            PipeBarrier<PIPE_V>();
            this->xInQueue_.FreeTensor(xLocal);
            // sumSq.GetValue(0) 是标量读，不依赖 V 流水，无需 PipeBarrier
            this->AccumSumSq(xFp32, sumSq, copyD, pipe_);
        }

        float invRms = FLOAT_ONE / sqrt(sumSq.GetValue(NUM_ZERO) * invK_ + normEps_);

        // xHat 双缓冲流水计算（第二遍遍历 dimTile）：对本核负责的 phi 行逐行 dot
        //
        // 第一轮已将整 token 的 xFp32 按 tile 拼接持久化（Regbase 归约对 xFp32 只读
        // 不覆盖），此处直接按偏移切片复用，免除 x 的 GM 重读（PERF-6：每 token 省一遍
        // x 搬运）；tile 内 phiRowQue_（depth=2）对 phi 行做双缓冲。
        LocalTensor<float> xHat = this->xHatBuf_.template Get<float>();
        Duplicate(xHat, FLOAT_ZERO, static_cast<uint32_t>(AlignUp(phiRows_)));
        PipeBarrier<PIPE_V>();

        for (int64_t d = 0; d < dimLoop_; d++) {
            int64_t dimOff = d * dimTile_;
            int64_t copyD = TileCopyD(d);
            LocalTensor<float> xFp32 = xFp32Base[d * dimTile_ * n_];

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
        int64_t xHatUsed = (phiRowEnd <= n_) ? rowCount : LocalIdxOf(phiRowEnd - 1, phiRowStart) + 1;
        Muls(xHat, xHat, invRms, static_cast<uint32_t>(xHatUsed));
        PipeBarrier<PIPE_V>();

        LocalTensor<float> biasLocal = this->biasBuf_.template Get<float>();

        // 本核有 h_post 段（仅 core0，phiRowStart < n_）：行 [0, n_)
        if (phiRowStart < n_) {
            int64_t postCount = n_ - phiRowStart;
            CopyInBias(biasLocal, phiRowStart, postCount, pipe_);
            ComputeAndWriteHPost(tokenIdx, xHat, biasLocal, phiRowStart, postCount, pipe_);
        }

        // 本核有 h_res 段（core0: [n_, phiRowMid)；core1: [phiRowMid, phiRows)）
        int64_t resStart = (phiRowStart >= n_) ? phiRowStart : n_;
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
