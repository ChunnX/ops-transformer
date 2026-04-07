/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compute.h
 * \brief MicroAPI VF helpers for masked_causal_conv1d_backward
 */

#ifndef MASK_EDCAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H
#define MASK_EDCAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H

#include "kernel_operator.h"

using namespace AscendC;

namespace AggHiddenGradVF {

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t REGSIZE = 256;
constexpr uint32_t B32_REP_SIZE = REGSIZE / sizeof(float); // 64 floats per vector reg



// gi[i] = go[i]*w2 + go[i+1]*w1 + go[i+2]*w0 (0-pad beyond sLen)
template <typename T>
__simd_vf__ void GradInputW3VF(__ubuf__ T *goAddr, __ubuf__ T *wAddr, __ubuf__ T *giAddr, 
                                uint32_t sEff, uint32_t sLen, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        // Load weights chunk (w0,w1,w2) for current H slice
        MicroAPI::RegTensor<T> w0B16, w1B16, w2B16;
        MicroAPI::RegTensor<float> w0B32, w1B32, w2B32;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, wAddr + 0 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, wAddr + 1 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, wAddr + 2 * dimLen + dimOff);
        MicroAPI::Cast<float, T, castTraitB162B32>(w0B32, w0B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w1B32, w1B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w2B32, w2B16, fullMask);

        for (uint32_t row = 0; row < sLen; ++row) {
            __ubuf__ T *goRow = goAddr + row * dimLen + dimOff;
            __ubuf__ T *giRow = giAddr + row * dimLen + dimOff;

            MicroAPI::RegTensor<T> goB16, goN1B16, goN2B16, giB16;
            MicroAPI::RegTensor<float> goB32, goN1B32, goN2B32, accB32, mulB32;

            // acc = go * w2
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goB16, goRow);
            MicroAPI::Cast<float, T, castTraitB162B32>(goB32, goB16, fullMask);
            MicroAPI::Mul(accB32, goB32, w2B32, fullMask);

            // + go[i+1] * w1
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN1B16, goRow + dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(goN1B32, goN1B16, fullMask);
            MicroAPI::Mul(mulB32, goN1B32, w1B32, fullMask);
            MicroAPI::Add(accB32, accB32, mulB32, fullMask);

            // + go[i+2] * w0
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN2B16, goRow + 2 * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(goN2B32, goN2B16, fullMask);
            MicroAPI::Mul(mulB32, goN2B32, w0B32, fullMask);
            MicroAPI::Add(accB32, accB32, mulB32, fullMask);

            MicroAPI::Cast<T, float, castTraitB322B16>(giB16, accB32, fullMask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giRow, giB16, fullMask);
        }
        dimOff += B32_REP_SIZE;
    }
}


// acc2 += sum(go[i]*in[i]), acc1 += sum(go[i+1]*in[i]), acc0 += sum(go[i+2]*in[i])
template <typename T>
__simd_vf__ void GradWeightW3VFAcc(__ubuf__ T *goAddr, __ubuf__ T *inAddr,
                                    __ubuf__ float *acc0Addr, __ubuf__ float *acc1Addr, __ubuf__ float *acc2Addr,
                                    uint32_t sEff, uint32_t sLen, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        MicroAPI::RegTensor<float> acc0, acc1, acc2, mulB32, goB32, goN1B32, goN2B32, inB32;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(acc0, acc0Addr + dimOff);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(acc1, acc1Addr + dimOff);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(acc2, acc2Addr + dimOff);

        for (uint32_t row = 0; row < sLen; ++row) {
            __ubuf__ T *goRow = goAddr + row * dimLen + dimOff;
            __ubuf__ T *inRow = inAddr + row * dimLen + dimOff;
            MicroAPI::RegTensor<T> goB16, inB16, goN1B16, goN2B16;
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goB16, goRow);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(inB16, inRow);
            MicroAPI::Cast<float, T, castTraitB162B32>(goB32, goB16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(inB32, inB16, fullMask);

            // acc2 += go * in
            MicroAPI::Mul(mulB32, goB32, inB32, fullMask);
            MicroAPI::Add(acc2, acc2, mulB32, fullMask);

            // acc1 += go[i+1] * in
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN1B16, goRow + dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(goN1B32, goN1B16, fullMask);
            MicroAPI::Mul(mulB32, goN1B32, inB32, fullMask);
            MicroAPI::Add(acc1, acc1, mulB32, fullMask);

            // acc0 += go[i+2] * in
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN2B16, goRow + 2 * dimLen);
            MicroAPI::Cast<float, T, castTraitB162B32>(goN2B32, goN2B16, fullMask);
            MicroAPI::Mul(mulB32, goN2B32, inB32, fullMask);
            MicroAPI::Add(acc0, acc0, mulB32, fullMask);

        }
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(acc0Addr + dimOff, acc0, fullMask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(acc1Addr + dimOff, acc1, fullMask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(acc2Addr + dimOff, acc2, fullMask);
        dimOff += B32_REP_SIZE;
    }
}

// Reduce across B for accumulated fp32 buffers and cast to DT
// accX layout: B slices, each of length dimLen (contiguous along H)
template <typename T>
__simd_vf__ void ReduceBToHAndCast(__ubuf__ float *acc0Addr, __ubuf__ float *acc1Addr, __ubuf__ float *acc2Addr,
                                   uint32_t B, uint32_t dimLen,
                                   __ubuf__ T *out0Addr, __ubuf__ T *out1Addr, __ubuf__ T *out2Addr)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        MicroAPI::RegTensor<float> s0, s1, s2, tmp;
        // load batch 0 as initial sum
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(s0, acc0Addr + dimOff);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(s1, acc1Addr + dimOff);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(s2, acc2Addr + dimOff);
        // accumulate remaining batches
        for (uint32_t b = 1; b < B; ++b) {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(tmp, acc0Addr + b * dimLen + dimOff);
            MicroAPI::Add(s0, s0, tmp, fullMask);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(tmp, acc1Addr + b * dimLen + dimOff);
            MicroAPI::Add(s1, s1, tmp, fullMask);
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(tmp, acc2Addr + b * dimLen + dimOff);
            MicroAPI::Add(s2, s2, tmp, fullMask);
        }
        // cast and store
        MicroAPI::RegTensor<T> o0B16, o1B16, o2B16;
        MicroAPI::Cast<T, float, castTraitB322B16>(o0B16, s0, fullMask);
        MicroAPI::Cast<T, float, castTraitB322B16>(o1B16, s1, fullMask);
        MicroAPI::Cast<T, float, castTraitB322B16>(o2B16, s2, fullMask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(out0Addr + dimOff, o0B16, fullMask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(out1Addr + dimOff, o1B16, fullMask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(out2Addr + dimOff, o2B16, fullMask);
        dimOff += B32_REP_SIZE;
    }
}

// Wrapper: reduce across B and cast, from LocalTensor to VF
template <typename T>
__aicore__ inline void DoReduceBToHAndCast(LocalTensor<float> &acc0Ub, LocalTensor<float> &acc1Ub, LocalTensor<float> &acc2Ub,
                                           uint32_t B, uint32_t dimLen, LocalTensor<T> &out0Ub, LocalTensor<T> &out1Ub,
                                           LocalTensor<T> &out2Ub)
{
    __ubuf__ float *a0 = (__ubuf__ float *)acc0Ub.GetPhyAddr();
    __ubuf__ float *a1 = (__ubuf__ float *)acc1Ub.GetPhyAddr();
    __ubuf__ float *a2 = (__ubuf__ float *)acc2Ub.GetPhyAddr();
    __ubuf__ T *o0 = (__ubuf__ T *)out0Ub.GetPhyAddr();
    __ubuf__ T *o1 = (__ubuf__ T *)out1Ub.GetPhyAddr();
    __ubuf__ T *o2 = (__ubuf__ T *)out2Ub.GetPhyAddr();
    ReduceBToHAndCast<T>(a0, a1, a2, B, dimLen, o0, o1, o2);
}

template <typename T>
__aicore__ inline void DoGradInput(LocalTensor<T> &goUb, LocalTensor<T> &wUb, LocalTensor<T> &giUb,
                                   uint32_t sEff, uint32_t sLen, uint32_t dimLen)
{
    __ubuf__ T *goAddr = (__ubuf__ T *)goUb.GetPhyAddr();
    __ubuf__ T *wAddr = (__ubuf__ T *)wUb.GetPhyAddr();
    __ubuf__ T *giAddr = (__ubuf__ T *)giUb.GetPhyAddr();
    GradInputW3VF<T>(goAddr, wAddr, giAddr, sEff, sLen, dimLen);
}



template <typename T>
__aicore__ inline void DoGradWeightAcc(LocalTensor<T> &goUb, LocalTensor<T> &inUb, LocalTensor<float> &acc0Ub,
                                       LocalTensor<float> &acc1Ub, LocalTensor<float> &acc2Ub,
                                       uint32_t sEff, uint32_t sLen, uint32_t dimLen)
{
    __ubuf__ T *goAddr = (__ubuf__ T *)goUb.GetPhyAddr();
    __ubuf__ T *inAddr = (__ubuf__ T *)inUb.GetPhyAddr();
    __ubuf__ float *a0Addr = (__ubuf__ float *)acc0Ub.GetPhyAddr();
    __ubuf__ float *a1Addr = (__ubuf__ float *)acc1Ub.GetPhyAddr();
    __ubuf__ float *a2Addr = (__ubuf__ float *)acc2Ub.GetPhyAddr();
    GradWeightW3VFAcc<T>(goAddr, inAddr, a0Addr, a1Addr, a2Addr, sEff, sLen, dimLen);
}

} // namespace AggHiddenGradVF

#endif // MASK_EDCAUSAL_CONV1D_BACKWARD_VF_COMPUTE_H
