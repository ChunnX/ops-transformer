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
 * \file inplace_partial_rotary_mul_grad.h
 * \brief RegBase kernel class for InplacePartialRotaryMulGrad (arch35 / ascend950).
 *
 * Only the x == nullptr scenario is supported: dy is rotated in place into dx over the partial slice
 * [partialStart, partialStart + sliceLen) of its last dim. Only BSND layout and B/S/N broadcast of
 * cos / sin are supported. dcos / dsin are not computed (x is nullptr).
 *
 * Work split: M = B*S*N rows are split across cores (blockFactorM per core). Each core walks its rows
 * in UB tiles of ubFactorM rows. For every tile dy is read / written back with a strided partial-slice
 * DataCopy (rows are D apart, sliceLen elements per row starting at partialStart); cos / sin are loaded
 * with broadcast awareness (full broadcast loads a single row, otherwise consecutive dy rows that map to
 * consecutive cos rows are coalesced into one strided DataCopy).
 */

#ifndef INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_
#define INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_

#include "kernel_operator.h"
#include "inplace_partial_rotary_mul_grad_tiling_data.h"
#include "inplace_partial_rotary_mul_grad_common.h"

using namespace AscendC;

namespace InplacePartialRotaryMulGrad {

template <typename Tdy, typename Tcs>
class InplacePartialRotaryMulGrad {
public:
    __aicore__ inline InplacePartialRotaryMulGrad() {}
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dyOut,
        const InplacePartialRotaryMulGradTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInDy(LocalTensor<Tdy>& dyTensor, int64_t m0, uint16_t tileRows);
    __aicore__ inline void CopyOutDy(const LocalTensor<Tdy>& outTensor, int64_t m0, uint16_t tileRows);
    __aicore__ inline void CopyInCosSin(LocalTensor<Tcs>& csTensor, GlobalTensor<Tcs>& csGm, int64_t m0,
        uint16_t tileRows);
    __aicore__ inline void Compute(const LocalTensor<Tdy>& dyTensor, const LocalTensor<Tcs>& cosTensor,
        const LocalTensor<Tcs>& sinTensor, LocalTensor<Tdy>& outTensor, uint16_t tileRows);
    __aicore__ inline int64_t CosRowIdx(int64_t m) const;

private:
    TPipe* pipe_{nullptr};
    const InplacePartialRotaryMulGradTilingData* tiling_{nullptr};

    GlobalTensor<Tdy> dyGm_;
    GlobalTensor<Tdy> dyOutGm_;
    GlobalTensor<Tcs> cosGm_;
    GlobalTensor<Tcs> sinGm_;

    TQue<QuePosition::VECIN, 1> dyInQue_;
    TQue<QuePosition::VECIN, 1> cosInQue_;
    TQue<QuePosition::VECIN, 1> sinInQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    DataCopyPadExtParams<Tdy> dyPad_{false, 0, 0, static_cast<Tdy>(0)};
    DataCopyPadExtParams<Tcs> csPad_{false, 0, 0, static_cast<Tcs>(0)};

    int64_t blockIdx_{0};
    int64_t b_{0}, s_{0}, n_{0}, d_{0};
    int64_t sliceStart_{0}, sliceLen_{0};
    int64_t rotaryMode_{0};
    int64_t dSplitCoef_{1};
    int64_t dAlignDy_{0};   // dy / dx UB row stride (elements), also used as out stride except mode 3
    int64_t dAlignCs_{0};   // cos / sin UB row stride (elements)
    int64_t dAlignOut_{0};  // dx output UB row stride (elements); == dAlignDy_ except mode 3 (interleaved)
    int64_t ubFactorM_{0};
    int64_t coreRows_{0};     // rows assigned to this core
    int64_t coreRowStart_{0}; // global row index of the first row this core handles
    int64_t cosRowStride_{0}; // advance of cos/sin pointer per dy row in the tile (0 for full broadcast)
};

template <typename Tdy, typename Tcs>
__aicore__ inline int64_t InplacePartialRotaryMulGrad<Tdy, Tcs>::CosRowIdx(int64_t m) const
{
    int64_t sn = s_ * n_;
    int64_t b = m / sn;
    int64_t rem = m % sn;
    int64_t sIdx = rem / n_;
    int64_t nIdx = rem % n_;
    return b * tiling_->cosStrideB + sIdx * tiling_->cosStrideS + nIdx * tiling_->cosStrideN;
}

template <typename Tdy, typename Tcs>
__aicore__ inline void InplacePartialRotaryMulGrad<Tdy, Tcs>::Init(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR dyOut, const InplacePartialRotaryMulGradTilingData* tiling, TPipe* pipe)
{
    pipe_ = pipe;
    tiling_ = tiling;
    blockIdx_ = GetBlockIdx();
    b_ = tiling->b;
    s_ = tiling->s;
    n_ = tiling->n;
    d_ = tiling->d;
    sliceStart_ = tiling->partialStart;
    sliceLen_ = tiling->sliceLen;
    rotaryMode_ = tiling->rotaryMode;
    ubFactorM_ = tiling->ubFactorM;

    if (rotaryMode_ == 0 || rotaryMode_ == 3) { // half / interleave-half: input split in halves
        dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (rotaryMode_ == 2) { // quarter
        dSplitCoef_ = QUARTER_MODE_COEF;
    } else { // interleave
        dSplitCoef_ = 1;
    }
    uint32_t blockElemsDy = BLOCK_TYPE_SIZE / sizeof(Tdy);
    uint32_t blockElemsCs = BLOCK_TYPE_SIZE / sizeof(Tcs);
    dAlignDy_ = Ops::Base::CeilAlign(sliceLen_ / dSplitCoef_, static_cast<int64_t>(blockElemsDy)) * dSplitCoef_;
    dAlignCs_ = Ops::Base::CeilAlign(sliceLen_ / dSplitCoef_, static_cast<int64_t>(blockElemsCs)) * dSplitCoef_;
    if (rotaryMode_ == 3) {
        dAlignOut_ = Ops::Base::CeilAlign(sliceLen_, static_cast<int64_t>(blockElemsDy));
    } else {
        dAlignOut_ = dAlignDy_;
    }
    cosRowStride_ = (tiling->isFullBroadcast == 1) ? 0 : dAlignCs_;

    int64_t mTotal = b_ * s_ * n_;
    int64_t usedCoreNum = tiling->usedCoreNum;
    int64_t blockFactor = tiling->blockFactorM;
    coreRowStart_ = blockIdx_ * blockFactor;
    if (blockIdx_ == usedCoreNum - 1) {
        coreRows_ = mTotal - coreRowStart_;
    } else {
        coreRows_ = blockFactor;
    }

    dyGm_.SetGlobalBuffer((__gm__ Tdy*)dy);
    dyOutGm_.SetGlobalBuffer((__gm__ Tdy*)dyOut);
    cosGm_.SetGlobalBuffer((__gm__ Tcs*)cos);
    sinGm_.SetGlobalBuffer((__gm__ Tcs*)sin);

    constexpr int64_t bufferNum = 1;
    pipe_->InitBuffer(dyInQue_, bufferNum, ubFactorM_ * dAlignDy_ * sizeof(Tdy));
    pipe_->InitBuffer(outQue_, bufferNum, ubFactorM_ * dAlignOut_ * sizeof(Tdy));
    pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorM_ * dAlignCs_ * sizeof(Tcs));
    pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorM_ * dAlignCs_ * sizeof(Tcs));
}

template <typename Tdy, typename Tcs>
__aicore__ inline void InplacePartialRotaryMulGrad<Tdy, Tcs>::CopyInDy(
    LocalTensor<Tdy>& dyTensor, int64_t m0, uint16_t tileRows)
{
    int64_t offset = m0 * d_ + sliceStart_;
    uint32_t blockLen = static_cast<uint32_t>(sliceLen_ / dSplitCoef_ * sizeof(Tdy));
    if (rotaryMode_ == 0 || rotaryMode_ == 3) {
        CopyInHalfMode<Tdy>(dyTensor, dyGm_, offset, tileRows, blockLen, dyPad_, d_, sliceLen_, dSplitCoef_);
    } else if (rotaryMode_ == 2) {
        CopyInQuarterMode<Tdy>(dyTensor, dyGm_, offset, tileRows, blockLen, dyPad_, d_, sliceLen_, dSplitCoef_);
    } else {
        CopyInInterleaveMode<Tdy>(dyTensor, dyGm_, offset, tileRows, blockLen, dyPad_, d_, sliceLen_);
    }
}

template <typename Tdy, typename Tcs>
__aicore__ inline void InplacePartialRotaryMulGrad<Tdy, Tcs>::CopyOutDy(
    const LocalTensor<Tdy>& outTensor, int64_t m0, uint16_t tileRows)
{
    int64_t offset = m0 * d_ + sliceStart_;
    uint32_t blockLen = static_cast<uint32_t>(sliceLen_ / dSplitCoef_ * sizeof(Tdy));
    if (rotaryMode_ == 0 || rotaryMode_ == 2) {
        if (rotaryMode_ == 0) {
            CopyOutHalfMode<Tdy>(outTensor, dyOutGm_, offset, tileRows, blockLen, d_, sliceLen_, dSplitCoef_);
        } else {
            CopyOutQuarterMode<Tdy>(outTensor, dyOutGm_, offset, tileRows, blockLen, d_, sliceLen_, dSplitCoef_);
        }
    } else {
        // interleave (1) and interleave-half (3) write the slice contiguously: blockLen is the full
        // slice (not sliceLen/dSplitCoef, which only applies to the split half/quarter outputs).
        uint32_t outBlockLen = static_cast<uint32_t>(sliceLen_ * sizeof(Tdy));
        CopyOutInterleaveMode<Tdy>(outTensor, dyOutGm_, offset, tileRows, outBlockLen, d_, sliceLen_);
    }
}

template <typename Tdy, typename Tcs>
__aicore__ inline void InplacePartialRotaryMulGrad<Tdy, Tcs>::CopyInCosSin(
    LocalTensor<Tcs>& csTensor, GlobalTensor<Tcs>& csGm, int64_t m0, uint16_t tileRows)
{
    uint32_t blockLen = static_cast<uint32_t>(sliceLen_ / dSplitCoef_ * sizeof(Tcs));
    if (tiling_->isFullBroadcast == 1) {
        // cos / sin broadcast fully: a single row feeds every dy row.
        int64_t offset = 0;
        if (rotaryMode_ == 0 || rotaryMode_ == 3) {
            CopyInHalfMode<Tcs>(csTensor, csGm, offset, 1, blockLen, csPad_, sliceLen_, sliceLen_, dSplitCoef_);
        } else if (rotaryMode_ == 2) {
            CopyInQuarterMode<Tcs>(csTensor, csGm, offset, 1, blockLen, csPad_, sliceLen_, sliceLen_, dSplitCoef_);
        } else {
            CopyInInterleaveMode<Tcs>(csTensor, csGm, offset, 1, blockLen, csPad_, sliceLen_, sliceLen_);
        }
        return;
    }
    // Coalesce runs of consecutive dy rows that map to consecutive cos/sin rows into one strided copy.
    // cos/sin rows are sliceLen apart in GM, so "D" passed to the helper == sliceLen.
    uint16_t i = 0;
    while (i < tileRows) {
        int64_t c0 = CosRowIdx(m0 + i);
        uint16_t j = i + 1;
        while (j < tileRows && CosRowIdx(m0 + j) == c0 + (j - i)) {
            j++;
        }
        uint16_t runLen = j - i;
        LocalTensor<Tcs> runTensor = csTensor[i * dAlignCs_];
        int64_t offset = c0 * sliceLen_;
        if (rotaryMode_ == 0 || rotaryMode_ == 3) {
            CopyInHalfMode<Tcs>(runTensor, csGm, offset, runLen, blockLen, csPad_, sliceLen_, sliceLen_, dSplitCoef_);
        } else if (rotaryMode_ == 2) {
            CopyInQuarterMode<Tcs>(runTensor, csGm, offset, runLen, blockLen, csPad_, sliceLen_, sliceLen_, dSplitCoef_);
        } else {
            CopyInInterleaveMode<Tcs>(runTensor, csGm, offset, runLen, blockLen, csPad_, sliceLen_, sliceLen_);
        }
        i = j;
    }
}

template <typename Tdy, typename Tcs>
__aicore__ inline void InplacePartialRotaryMulGrad<Tdy, Tcs>::Compute(const LocalTensor<Tdy>& dyTensor,
    const LocalTensor<Tcs>& cosTensor, const LocalTensor<Tcs>& sinTensor, LocalTensor<Tdy>& outTensor,
    uint16_t tileRows)
{
    __local_mem__ Tcs* sinUb = (__local_mem__ Tcs*)sinTensor.GetPhyAddr();
    __local_mem__ Tcs* cosUb = (__local_mem__ Tcs*)cosTensor.GetPhyAddr();
    __local_mem__ Tdy* dyUb = (__local_mem__ Tdy*)dyTensor.GetPhyAddr();
    __local_mem__ Tdy* outUb = (__local_mem__ Tdy*)outTensor.GetPhyAddr();
    uint32_t dLen = static_cast<uint32_t>(sliceLen_);
    if (rotaryMode_ == 0) {
        GradHalfVF<Tdy, Tcs>(sinUb, cosUb, dyUb, outUb, dLen, dAlignDy_, dAlignCs_, cosRowStride_, cosRowStride_,
            tileRows);
    } else if (rotaryMode_ == 1) {
        GradInterleaveVF<Tdy, Tcs>(sinUb, cosUb, dyUb, outUb, static_cast<uint16_t>(dLen), dAlignDy_, dAlignCs_,
            cosRowStride_, cosRowStride_, tileRows);
    } else if (rotaryMode_ == 2) {
        GradQuarterVF<Tdy, Tcs>(sinUb, cosUb, dyUb, outUb, dLen, dAlignDy_, dAlignCs_, cosRowStride_, cosRowStride_,
            tileRows);
    } else {
        GradInterleaveHalfVF<Tdy, Tcs>(sinUb, cosUb, dyUb, outUb, dLen, dAlignDy_, dAlignCs_, dAlignOut_,
            cosRowStride_, cosRowStride_, tileRows);
    }
}

template <typename Tdy, typename Tcs>
__aicore__ inline void InplacePartialRotaryMulGrad<Tdy, Tcs>::Process()
{
    if (coreRows_ <= 0 || sliceLen_ <= 0) {
        return;
    }
    int64_t tileCount = Ops::Base::CeilDiv(coreRows_, ubFactorM_);
    for (int64_t t = 0; t < tileCount; t++) {
        int64_t m0 = coreRowStart_ + t * ubFactorM_;
        uint16_t tileRows = static_cast<uint16_t>(
            (t == tileCount - 1) ? (coreRows_ - t * ubFactorM_) : ubFactorM_);

        LocalTensor<Tdy> dyTensor = dyInQue_.AllocTensor<Tdy>();
        CopyInDy(dyTensor, m0, tileRows);
        dyInQue_.EnQue(dyTensor);

        LocalTensor<Tcs> cosTensor = cosInQue_.AllocTensor<Tcs>();
        CopyInCosSin(cosTensor, cosGm_, m0, tileRows);
        cosInQue_.EnQue(cosTensor);

        LocalTensor<Tcs> sinTensor = sinInQue_.AllocTensor<Tcs>();
        CopyInCosSin(sinTensor, sinGm_, m0, tileRows);
        sinInQue_.EnQue(sinTensor);

        dyTensor = dyInQue_.DeQue<Tdy>();
        cosTensor = cosInQue_.DeQue<Tcs>();
        sinTensor = sinInQue_.DeQue<Tcs>();

        LocalTensor<Tdy> outTensor = outQue_.AllocTensor<Tdy>();
        Compute(dyTensor, cosTensor, sinTensor, outTensor, tileRows);
        outQue_.EnQue(outTensor);
        outTensor = outQue_.DeQue<Tdy>();

        CopyOutDy(outTensor, m0, tileRows);

        outQue_.FreeTensor(outTensor);
        dyInQue_.FreeTensor(dyTensor);
        cosInQue_.FreeTensor(cosTensor);
        sinInQue_.FreeTensor(sinTensor);
    }
}

} // namespace InplacePartialRotaryMulGrad

#endif // INPLACE_PARTIAL_ROTARY_MUL_GRAD_H_
