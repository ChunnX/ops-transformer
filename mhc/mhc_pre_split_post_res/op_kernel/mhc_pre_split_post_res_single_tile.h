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
 * \file mhc_pre_split_post_res_single_tile.h
 * \brief KernelMhcSingleTile<T>：SINGLE_TILE 路径（dimLoop==1，全量 D 一次搬入）
 *
 * 每个 token 由 coresPerToken（1 或 2）个核并行处理，phi 行按 phiRowMid 对半分：
 *   core0 (phiRowStart=0)        ：处理 h_post 行 [0, N) 和 h_res 行 [N, phiRowMid)
 *   core1 (phiRowStart=phiRowMid)：处理 h_res 行 [phiRowMid, phiRows)
 *
 * xHat 本地下标规则（phiRowStart 由调用方传入）：
 *   row < n_  → row - phiRowStart          （core0 的 h_post 段，phiRowStart==0）
 *   row >= n_ → AlignUp(postCount) + (row - resStart)  （h_res 段）
 */

#ifndef MHC_PRE_SPLIT_POST_RES_SINGLE_TILE_H_
#define MHC_PRE_SPLIT_POST_RES_SINGLE_TILE_H_

#include "mhc_pre_split_post_res_base.h"

namespace MhcPreSplitPostRes {
using namespace AscendC;

template <typename T>
class KernelMhcSingleTile : public KernelMhcBase<T> {
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
    __aicore__ inline KernelMhcSingleTile()
    {}

    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR phiAddr, GM_ADDR alphaAddr, GM_ADDR biasAddr, GM_ADDR hPostAddr,
        GM_ADDR hResAddr, GM_ADDR workspace, TPipe *pipe, const MhcPreSplitPostResTilingData *tiling)
    {
        pipe_ = pipe;
        totalLength_ = tiling->totalLength;
        n_ = tiling->n;
        d_ = tiling->d;
        phiRows_ = tiling->phiRows;
        normEps_ = tiling->normEps;
        invK_ = tiling->invK;
        phiRowMid_ = tiling->phiRowMid;
        baseT_ = tiling->baseT;
        tailT_ = tiling->tailT;
        coreTAct_ = tiling->coreTAct;
        coresPerToken_ = tiling->coresPerToken;

        InitGM(xAddr, phiAddr, alphaAddr, biasAddr, hPostAddr, hResAddr);
        // dimTile 传入 d_，表示全量 D 一次搬入，无 dimLoop 循环
        InitCommonBuffers(pipe, d_);
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
        int64_t tCount = (groupIdx == coreTAct_ - NUM_ONE) ? tailT_ : baseT_;

        // localIdx==NUM_ZERO: 处理 phi 行 [NUM_ZERO, phiRowMid)；localIdx==NUM_ONE: 处理 [phiRowMid, phiRows)
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

    __aicore__ inline void ProcessOneToken(int64_t tokenIdx, int64_t phiRowStart, int64_t phiRowEnd)
    {
        int64_t xTokenOff = tokenIdx * n_ * d_;

        int64_t copyD = d_;

        CopyInXTile(xTokenOff, 0, copyD);
        LocalTensor<T> xLocal = this->xInQueue_.template DeQue<T>();
        LocalTensor<float> xFp32 = this->xFp32Buf_.template Get<float>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(n_ * copyD));
        PipeBarrier<PIPE_V>();
        this->xInQueue_.FreeTensor(xLocal);

        // 计算全量 x 的平方和，invRms = 1/sqrt(mean(x^2) + eps)
        LocalTensor<float> sumSq = this->sumSqBuf_.template Get<float>();
        Duplicate(sumSq, FLOAT_ZERO, static_cast<uint32_t>(FP32_ALIGN_ELEM));
        PipeBarrier<PIPE_V>();
        this->AccumSumSq(xFp32, sumSq, copyD, pipe_);
        float invRms = FLOAT_ONE / sqrt(sumSq.GetValue(NUM_ZERO) * invK_ + normEps_);

        // 对本核负责的 phi 行做双缓冲流水：预取 row+1 与计算 row 重叠
        LocalTensor<float> xHat = this->xHatBuf_.template Get<float>();
        Duplicate(xHat, FLOAT_ZERO, static_cast<uint32_t>(AlignUp(phiRows_)));
        PipeBarrier<PIPE_V>();

        PrefetchPhiRow(phiRowStart, 0, copyD);
        for (int64_t row = phiRowStart; row < phiRowEnd - 1; row++) {
            PrefetchPhiRow(row + 1, 0, copyD);
            ComputePhiDot(xFp32, xHat, LocalIdxOf(row, phiRowStart), copyD, pipe_);
        }
        ComputePhiDot(xFp32, xHat, LocalIdxOf(phiRowEnd - 1, phiRowStart), copyD, pipe_);

        // xHat 乘以 invRms，完成 RMS norm
        int64_t rowCount = phiRowEnd - phiRowStart;
        int64_t xHatUsed = (phiRowEnd <= n_) ? rowCount : LocalIdxOf(phiRowEnd - 1, phiRowStart) + 1;
        Muls(xHat, xHat, invRms, static_cast<uint32_t>(xHatUsed));
        PipeBarrier<PIPE_V>();

        LocalTensor<float> biasLocal = this->biasBuf_.template Get<float>();

        // 本核有 h_post 段（仅 core0，phiRowStart < n_）
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
    int64_t phiRowMid_ = 0;
    int64_t baseT_ = 0;
    int64_t tailT_ = 0;
    int64_t coreTAct_ = 0;
    int64_t coresPerToken_ = 0;
};

}  // namespace MhcPreSplitPostRes

#endif  // MHC_PRE_SPLIT_POST_RES_SINGLE_TILE_H_
