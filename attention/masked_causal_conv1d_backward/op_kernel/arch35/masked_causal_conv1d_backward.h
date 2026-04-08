/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file masked_causal_conv1d_backward.h
 * \brief Arch35 AICore kernel for masked_causal_conv1d_backward (W=3)
 */

#ifndef MASK_EDCAUSAL_CONV1D_BACKWARD_H
#define MASK_EDCAUSAL_CONV1D_BACKWARD_H

#include "kernel_operator.h"
#include "vf/compute.h"
#include "masked_causal_conv1d_backward_struct.h"

namespace MaskedCausalConv1dBackwardKernelNS {
using namespace AscendC;
using MaskedCausalConv1dBackwardArch35Tiling::MaskedCausalConv1dBackwardTilingDataV35;

template <typename DT>
class MaskedCausalConv1dBackwardKernel {
public:
    static constexpr int kW = 3;
    static constexpr int kBufferNum = 2;
    static constexpr int kAlignBytes = 32;

    __aicore__ inline MaskedCausalConv1dBackwardKernel();

    __aicore__ inline void Init(GM_ADDR grad_output,
                                GM_ADDR input,
                                GM_ADDR weight,
                                GM_ADDR mask,
                                GM_ADDR grad_input,
                                GM_ADDR grad_weight,
                                const MaskedCausalConv1dBackwardTilingDataV35 *td,
                                TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInGradOutput(int64_t bTile, int64_t sTile, int64_t hTile,
                                            int64_t sEff, int64_t hLenThis);
    __aicore__ inline void CopyInInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                       int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyInWeight(int64_t hTile, int64_t hLenThis);
    __aicore__ inline void CopyInMask(int64_t bTile, int64_t sTile, int64_t sLen);
    __aicore__ inline void ComputeGradOutputMask(int64_t sLen, int64_t hLenThis);
    __aicore__ inline void ComputeGradInputWeight(LocalTensor<DT>& goLocal,
                                                    LocalTensor<DT>& inLocal,
                                                    LocalTensor<DT>& wLocal,
                                                    int64_t sLen, int64_t hLenThis);
    __aicore__ inline void ReduceSTile(int64_t sTile, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void ReduceWsTile(int64_t bTile, int64_t hLenThis);
    __aicore__ inline void ReduceBTileAndCopyOut(int64_t hTile, int64_t hLenThis);
    __aicore__ inline void CopyOutGradInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                            int64_t sLen, int64_t hLenThis);

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

private:

    TPipe *pipe_;
    const MaskedCausalConv1dBackwardTilingDataV35 *td_;
    TQue<QuePosition::VECIN, kBufferNum> gradOutQ_;
    TQue<QuePosition::VECIN, kBufferNum> inputQ_;
    TQue<QuePosition::VECIN, kBufferNum> weightQ_;
    TQue<QuePosition::VECIN, kBufferNum> maskQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradInQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradWeightQ_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    GlobalTensor<DT> gradOutGm_;
    GlobalTensor<DT> inputGm_;
    GlobalTensor<DT> weightGm_;
    GlobalTensor<DT> gradInGm_;
    GlobalTensor<DT> gradWeightGm_;
    GlobalTensor<bool> maskGm_;

    int64_t hMainCoreCnt_{0};           // h维度主核核数
    int64_t hTailCoreCnt_{0};           // h维度尾核核数
    int64_t hMainSize_{0};              // h维度主核处理的大小
    int64_t hTailSize_{0};              // h维度尾核处理的大小
    int64_t hasMask_{0};                // 1，有mask；0，无mask
    int64_t S_{0};                      // S维度大小
    int64_t B_{0};                      // B维度大小
    int64_t H_{0};                      // H维度大小
    int64_t W_{0};                      // W维度大小

    int64_t hStart_{0}, hLen_{0};

    // whether this block is a main core (true) or tail core (false)
    bool isMainCore_{true};

    // per-core selected loop counts and UB factors
    int64_t hLoopCntCur_{0};
    int64_t sLoopCntCur_{0};
    int64_t ubMainHCur_{0};
    int64_t ubTailHCur_{0};
    int64_t ubMainSCur_{0};
    int64_t ubTailSCur_{0};

    // fp32 accumulators for grad_weight
    LocalTensor<float> gwAccF32_[kW];
};


template <typename DT>
__aicore__ inline MaskedCausalConv1dBackwardKernel<DT>::MaskedCausalConv1dBackwardKernel() {}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::Init(GM_ADDR grad_output,
                                                           GM_ADDR input,
                                                           GM_ADDR weight,
                                                           GM_ADDR mask,
                                                           GM_ADDR grad_input,
                                                           GM_ADDR grad_weight,
                                                           const MaskedCausalConv1dBackwardTilingDataV35 *td,
                                                           TPipe *pipe)
{
    pipe_ = pipe;
    td_ = td;

    // cache tiling fields
    hMainCoreCnt_ = td->hMainCoreCnt;           // h维度主核核数
    hTailCoreCnt_ = td->hTailCoreCnt;           // h维度尾核核数
    hMainSize_ = td->hMainSize;              // h维度主核处理的大小
    hTailSize_ = td->hTailSize;              // h维度尾核处理的大小

    hasMask_ = td->hasMask;                // 1，有mask；0，无mask
    S_ = td->S;                      // S维度大小
    B_ = td->B;                      // B维度大小
    H_ = td->H;                      // H维度大小
    W_ = td->W;                      // W维度大小

    // compute this core's H start/len
    uint64_t blkIdx = GetBlockIdx();
    isMainCore_ = (blkIdx < static_cast<uint64_t>(hMainCoreCnt_));
    if (isMainCore_) {
        hStart_ = blkIdx * hMainSize_;
        hLen_ = hMainSize_;
    } else {
        uint64_t tailIdx = blkIdx - static_cast<uint64_t>(hMainCoreCnt_);
        hStart_ = static_cast<uint64_t>(hMainCoreCnt_) * hMainSize_ + tailIdx * hTailSize_;
        hLen_ = hTailSize_;
    }

    // set per-core loop counts and UB factors
    hLoopCntCur_ = isMainCore_ ? td->hLoopCnt : td->tailHLoopCnt;
    sLoopCntCur_ = isMainCore_ ? td->sLoopCnt : td->tailSLoopCnt;
    ubMainHCur_ = isMainCore_ ? td->ubMainFactorH : td->tailCoreUbMainFactorH;
    ubTailHCur_ = isMainCore_ ? td->ubTailFactorH : td->tailCoreUbTailFactorH;
    ubMainSCur_ = isMainCore_ ? td->ubMainFactorS : td->tailCoreUbMainFactorS;
    ubTailSCur_ = isMainCore_ ? td->ubTailFactorS : td->tailCoreUbTailFactorS;

    // bind GM tensors
    gradOutGm_.SetGlobalBuffer((__gm__ DT *)grad_output, S_ * B_ * H_);
    inputGm_.SetGlobalBuffer((__gm__ DT *)input, S_ * B_ * H_);
    weightGm_.SetGlobalBuffer((__gm__ DT *)weight, W_ * H_);
    gradInGm_.SetGlobalBuffer((__gm__ DT *)grad_input, S_ * B_ * H_);
    gradWeightGm_.SetGlobalBuffer((__gm__ DT *)grad_weight, W_ * H_);
    if (hasMask_) {
        maskGm_.SetGlobalBuffer((__gm__ bool *)mask, B_ * S_);
    }

    // init queues and tmp buffers
    pipe_->InitBuffer(gradOutQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * (ubMainSCur_ + 2) * sizeof(DT)));
    pipe_->InitBuffer(inputQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * ubMainSCur_ * sizeof(DT)));
    pipe_->InitBuffer(weightQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * kW * sizeof(DT)));
    if (hasMask_) {
        uint32_t maskBytes = static_cast<uint32_t>(ubMainSCur_ + 2);
        uint32_t maskBufSize = (maskBytes + (kAlignBytes - 1)) / kAlignBytes * kAlignBytes;
        pipe_->InitBuffer(maskQ_, kBufferNum, maskBufSize);
    }
    pipe_->InitBuffer(gradInQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * ubMainSCur_ * sizeof(DT)));
    pipe_->InitBuffer(gradWeightQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * kW * sizeof(DT)));

    // Reuse a single buffer (tmpBuf_) for accumulators, product planes, S-partial sums (Ws) and sum buffer
    // Layout in bytes: [accBytes | p2Plane | p1Plane | p0Plane | wsBytes | sumBytes]
    uint32_t accBytes = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float));
    uint32_t planeBytesCap = static_cast<uint32_t>(kW * ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t wsBytes = static_cast<uint32_t>(kW * sLoopCntCur_ * ubMainHCur_ * sizeof(float));
    uint32_t sumBytes = static_cast<uint32_t>(ubMainHCur_ * sizeof(float));
    uint32_t totalBytes = accBytes + planeBytesCap + wsBytes + sumBytes;

    pipe_->InitBuffer(tmpBuf_, totalBytes);

    // Map gwAccF32_ to the accumulator region within tmpBuf_
    uint32_t stride = static_cast<uint32_t>(ubMainHCur_ * B_ * sizeof(float));
    gwAccF32_[0] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(ubMainHCur_ * B_), 0);
    gwAccF32_[1] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(ubMainHCur_ * B_), stride);
    gwAccF32_[2] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(ubMainHCur_ * B_), 2U * stride);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::Process()
{
    // Accumulate grad_weight in fp32 across all (b,s) tiles per h-tile, then cast+store once per h-tile
    for (int64_t hTile = 0; hTile < hLoopCntCur_; ++hTile) {
        int64_t hLenThis = (hTile == hLoopCntCur_ - 1 && ubTailHCur_ > 0) ? ubTailHCur_ : ubMainHCur_;

        // zero fp32 accumulators for this h-tile (batch-wise contiguous slices)
        Duplicate(gwAccF32_[0], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
        Duplicate(gwAccF32_[1], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
        Duplicate(gwAccF32_[2], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));

        // Load weight once per h-tile
        CopyInWeight(hTile, hLenThis);
        LocalTensor<DT> wLocal = weightQ_.DeQue<DT>();

        for (int64_t bTile = 0; bTile < B_; ++bTile) {
            for (int64_t sTile = 0; sTile < sLoopCntCur_; ++sTile) {
                int64_t sLen = (sTile == sLoopCntCur_ - 1 && ubTailSCur_ > 0) ? ubTailSCur_ : ubMainSCur_;
                int64_t sEff = (sTile == sLoopCntCur_ - 1) ? sLen : (sLen + 2);

                CopyInGradOutput(bTile, sTile, hTile, sEff, hLenThis);
                CopyInInput(bTile, sTile, hTile, sLen, hLenThis);

                LocalTensor<DT> goLocal = gradOutQ_.DeQue<DT>();
                LocalTensor<DT> inLocal = inputQ_.DeQue<DT>();
                ComputeGradInputWeight(goLocal, inLocal, wLocal, sLen, hLenThis);
                ReduceSTile(sTile, sLen, hLenThis);

                PipeBarrier<PIPE_V>();
                gradOutQ_.FreeTensor(goLocal);
                inputQ_.FreeTensor(inLocal);

                CopyOutGradInput(bTile, sTile, hTile, sLen, hLenThis);
            }

            ReduceWsTile(bTile, hLenThis);
        }

        weightQ_.FreeTensor(wLocal);
        ReduceBTileAndCopyOut(hTile, hLenThis);
    }
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::ComputeGradOutputMask(int64_t sLen, int64_t hLenThis)
{
    LocalTensor<bool> mLocal = maskQ_.DeQue<bool>();
    LocalTensor<DT> goLocal = gradOutQ_.DeQue<DT>();
    for (int64_t s = 0; s < sLen; ++s) {
        bool flag = mLocal.GetValue(static_cast<uint32_t>(s));
        if (!flag) {
            uint32_t offset = static_cast<uint32_t>((s) * hLenThis);
            LocalTensor<DT> row = goLocal[offset];
            Duplicate(row, static_cast<DT>(0), static_cast<uint32_t>(hLenThis));
        }
    }
    maskQ_.FreeTensor(mLocal);
    gradOutQ_.EnQue<DT>(goLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::ComputeGradInputWeight(
    LocalTensor<DT>& goLocal,
    LocalTensor<DT>& inLocal,
    LocalTensor<DT>& wLocal,
    int64_t sLen,
    int64_t hLenThis)
{
    LocalTensor<DT> giLocal = gradInQ_.AllocTensor<DT>();

    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t baseOff = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float)); // accBytes region size
    LocalTensor<float> p2Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff);
    LocalTensor<float> p1Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + planeBytesCap);
    LocalTensor<float> p0Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + 2U * planeBytesCap);

    AggHiddenGradVF::DoGradInputWeight<DT>(goLocal, wLocal, inLocal, giLocal,
                                           p0Ub, p1Ub, p2Ub,
                                           static_cast<uint32_t>(sLen), static_cast<uint32_t>(hLenThis));
    gradInQ_.EnQue<DT>(giLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::ReduceSTile(
    int64_t sTile, int64_t sLen, int64_t hLenThis)
{
    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t baseOff = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float)); // accBytes region size
    uint32_t wsBase  = baseOff + 3U * planeBytesCap; // start of Ws region
    uint32_t wsStrideBytes = static_cast<uint32_t>(hLenThis * sizeof(float));
    uint32_t wsBytesThisHTile = static_cast<uint32_t>(3U * sLoopCntCur_ * hLenThis * sizeof(float));
    uint32_t sumOff = wsBase + wsBytesThisHTile; // sum buffer placed after Ws region
    LocalTensor<float> sumUb = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis), sumOff);

    uint32_t srcShape[2] = {static_cast<uint32_t>(sLen), static_cast<uint32_t>(hLenThis)};

    LocalTensor<float> p2Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff);
    LocalTensor<float> p1Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + planeBytesCap);
    LocalTensor<float> p0Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + 2U * planeBytesCap);

    // k=2
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, p2Ub, srcShape, false);
    LocalTensor<float> ws2 = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis),
        static_cast<uint32_t>(wsBase + (0U * sLoopCntCur_ + static_cast<uint32_t>(sTile)) * wsStrideBytes));
    DataCopy(ws2, sumUb, static_cast<uint32_t>(hLenThis));

    // k=1
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, p1Ub, srcShape, false);
    LocalTensor<float> ws1 = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis),
        static_cast<uint32_t>(wsBase + (1U * sLoopCntCur_ + static_cast<uint32_t>(sTile)) * wsStrideBytes));
    DataCopy(ws1, sumUb, static_cast<uint32_t>(hLenThis));

    // k=0
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, p0Ub, srcShape, false);
    LocalTensor<float> ws0 = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis),
        static_cast<uint32_t>(wsBase + (2U * sLoopCntCur_ + static_cast<uint32_t>(sTile)) * wsStrideBytes));
    DataCopy(ws0, sumUb, static_cast<uint32_t>(hLenThis));
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::ReduceWsTile(
    int64_t bTile, int64_t hLenThis)
{
    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t baseOff = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float));
    uint32_t wsBase  = baseOff + 3U * planeBytesCap;
    uint32_t wsStrideBytes = static_cast<uint32_t>(hLenThis * sizeof(float));
    uint32_t wsBytesThisHTile = static_cast<uint32_t>(3U * sLoopCntCur_ * hLenThis * sizeof(float));
    uint32_t sumOff = wsBase + wsBytesThisHTile;
    LocalTensor<float> sumUb = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis), sumOff);
    uint32_t srcShape2[2] = {static_cast<uint32_t>(sLoopCntCur_), static_cast<uint32_t>(hLenThis)};

    // k=2
    LocalTensor<float> ws2Plane = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLoopCntCur_ * hLenThis),
        static_cast<uint32_t>(wsBase + 0U * sLoopCntCur_ * wsStrideBytes));
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, ws2Plane, srcShape2, false);
    LocalTensor<float> acc2Slice = gwAccF32_[2][static_cast<uint32_t>(bTile * ubMainHCur_)];
    DataCopy(acc2Slice, sumUb, static_cast<uint32_t>(hLenThis));

    // k=1
    LocalTensor<float> ws1Plane = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLoopCntCur_ * hLenThis),
        static_cast<uint32_t>(wsBase + 1U * sLoopCntCur_ * wsStrideBytes));
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, ws1Plane, srcShape2, false);
    LocalTensor<float> acc1Slice = gwAccF32_[1][static_cast<uint32_t>(bTile * ubMainHCur_)];
    DataCopy(acc1Slice, sumUb, static_cast<uint32_t>(hLenThis));

    // k=0
    LocalTensor<float> ws0Plane = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLoopCntCur_ * hLenThis),
        static_cast<uint32_t>(wsBase + 2U * sLoopCntCur_ * wsStrideBytes));
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, ws0Plane, srcShape2, false);
    LocalTensor<float> acc0Slice = gwAccF32_[0][static_cast<uint32_t>(bTile * ubMainHCur_)];
    DataCopy(acc0Slice, sumUb, static_cast<uint32_t>(hLenThis));
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInGradOutput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                       int64_t sEff, int64_t hLenThis)
{
    LocalTensor<DT> goLocal = gradOutQ_.AllocTensor<DT>();

    int64_t sStart = sTile * ubMainSCur_; // 12 * 315
    int64_t bStart = bTile;
    int64_t hOff = hTile * ubMainHCur_; // 0
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> dstBlock = goLocal[0];
    DataCopyExtParams inParams{static_cast<uint16_t>(sEff), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0, 0};

    DataCopyPad(dstBlock, gradOutGm_[base], inParams, padParams);

    // For last S-tile, zero the remaining 2 halo rows in buffer (batch clear)
    if (sTile == sLoopCntCur_ - 1) {
        uint32_t tailOff = static_cast<uint32_t>(sEff * hLenThis);
        LocalTensor<DT> tailRows = goLocal[tailOff];
        Duplicate(tailRows, static_cast<DT>(0), static_cast<uint32_t>(2 * hLenThis));
    }

    gradOutQ_.EnQue<DT>(goLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                  int64_t sLen, int64_t hLenThis)
{
    LocalTensor<DT> inLocal = inputQ_.AllocTensor<DT>();

    int64_t sStart = sTile * ubMainSCur_;
    int64_t bStart = bTile;
    int64_t hOff = hTile * ubMainHCur_;
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> dstBlock = inLocal[0];
    DataCopyExtParams inParams{static_cast<uint16_t>(sLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0, 0};
    DataCopyPad(dstBlock, inputGm_[base], inParams, padParams);

    inputQ_.EnQue<DT>(inLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInWeight(int64_t hTile, int64_t hLenThis)
{
    LocalTensor<DT> wLocal = weightQ_.AllocTensor<DT>();
    DataCopyExtParams inParams{static_cast<uint16_t>(kW), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0, 0};
    int64_t hOff = hTile * ubMainHCur_;
    int64_t base = (hStart_ + hOff);
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
    DataCopyPad(wLocal, weightGm_[base], inParams, padParams);
    weightQ_.EnQue<DT>(wLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyInMask(int64_t bTile, int64_t sTile,
                                                                 int64_t sLen)
{
    LocalTensor<bool> mLocal = maskQ_.AllocTensor<bool>();
    // bool is 1B on Ascend: mask has shape [B,S] with 1 byte per element
    DataCopyExtParams inParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sLen),
                               static_cast<uint32_t>(S_ - sLen), 0, 0};
    int64_t sStart = sTile * ubMainSCur_;
    int64_t bStart = bTile;
    int64_t base = bStart * S_ + sStart;
    DataCopyPadExtParams<bool> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, maskGm_[base], inParams, padParams);
    maskQ_.EnQue<bool>(mLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::CopyOutGradInput(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                       int64_t sLen,
                                                                       int64_t hLenThis)
{
    LocalTensor<DT> giLocal = gradInQ_.DeQue<DT>();

    int64_t sStart = sTile * ubMainSCur_;
    int64_t bStart = bTile;
    int64_t hOff = hTile * ubMainHCur_;

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> srcBlock = giLocal[0];
    DataCopyExtParams outParams{static_cast<uint16_t>(sLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               0, static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0};
    DataCopyPad(gradInGm_[base], srcBlock, outParams);

    // pipe_barrier(PIPE_ALL);
    gradInQ_.FreeTensor(giLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackwardKernel<DT>::ReduceBTileAndCopyOut(int64_t hTile, int64_t hLenThis)
{
    // Reduce across B in fp32 using ReduceSum, cast to DT, then write to GM
    int64_t hOff = hTile * ubMainHCur_;
    int64_t baseH = hStart_ + hOff;

    // Allocate UB buffer to hold casted [W,H] slices for this h-tile
    LocalTensor<DT> gwCast = gradWeightQ_.AllocTensor<DT>();
    LocalTensor<DT> out0 = gwCast;                                      // k=0 slice at offset 0
    LocalTensor<DT> out1 = gwCast[static_cast<uint32_t>(ubMainHCur_)];  // k=1 slice
    LocalTensor<DT> out2 = gwCast[static_cast<uint32_t>(2 * ubMainHCur_)]; // k=2 slice

    uint32_t accBytes = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float));
    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t sumOff = accBytes + 3U * planeBytesCap;
    LocalTensor<float> sumUb = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis), sumOff);
    uint32_t srcShape[2] = {static_cast<uint32_t>(B_), static_cast<uint32_t>(hLenThis)};

    // k=0
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, gwAccF32_[0], srcShape, false);
    AggHiddenGradVF::DoCastToDT<DT>(sumUb, static_cast<uint32_t>(hLenThis), out0);
    // k=1
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, gwAccF32_[1], srcShape, false);
    AggHiddenGradVF::DoCastToDT<DT>(sumUb, static_cast<uint32_t>(hLenThis), out1);
    // k=2
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, gwAccF32_[2], srcShape, false);
    AggHiddenGradVF::DoCastToDT<DT>(sumUb, static_cast<uint32_t>(hLenThis), out2);

    int64_t base = 0 * H_ + baseH;
    DataCopyExtParams outParams{static_cast<uint16_t>(kW), 
                                static_cast<uint32_t>(hLenThis * sizeof(DT)), 
                                static_cast<uint32_t>((ubMainHCur_ - hLenThis) * sizeof(DT)),
                                static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)),
                                0};
    DataCopyPad(gradWeightGm_[base], gwCast, outParams);

    gradWeightQ_.FreeTensor(gwCast);
}

} // namespace MaskedCausalConv1dBackwardKernelNS

#endif // MASK_EDCAUSAL_CONV1D_BACKWARD_H
