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

// Fused VF: compute grad_input and per-row products in a single pass over grad_output
// gi[row,:] = go[row,:]*w2 + go[row+1,:]*w1 + go[row+2,:]*w0
// p2[row,:] = go[row,:]   * in[row,:]
// p1[row,:] = go[row+1,:] * in[row,:]
// p0[row,:] = go[row+2,:] * in[row,:]
template <typename T>
__simd_vf__ void GradInputAndWeight(__ubuf__ T *goAddr, __ubuf__ T *wAddr, __ubuf__ T *inAddr, __ubuf__ T *giAddr,
                                             __ubuf__ float *p0Addr, __ubuf__ float *p1Addr, __ubuf__ float *p2Addr,
                                             uint32_t sLen, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;

    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        // load weights once per dim chunk
        MicroAPI::RegTensor<T> w0B16, w1B16, w2B16;
        MicroAPI::RegTensor<float> w0B32, w1B32, w2B32;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, wAddr + 0 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, wAddr + 1 * dimLen + dimOff);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, wAddr + 2 * dimLen + dimOff);
        MicroAPI::Cast<float, T, castTraitB162B32>(w0B32, w0B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w1B32, w1B16, fullMask);
        MicroAPI::Cast<float, T, castTraitB162B32>(w2B32, w2B16, fullMask);

        MicroAPI::RegTensor<T> goB16, goN1B16, goN2B16, inB16, giB16;
        MicroAPI::RegTensor<float> goB32, goN1B32, goN2B32, inB32, accB32, mulB32;

        for (uint32_t r = 0; r < sLen; ++r) {
            __ubuf__ T *goRow   = goAddr + r * dimLen + dimOff;
            __ubuf__ T *goN1Row = goAddr + (r + 1U) * dimLen + dimOff;
            __ubuf__ T *goN2Row = goAddr + (r + 2U) * dimLen + dimOff;
            __ubuf__ T *inRow   = inAddr + r * dimLen + dimOff;
            __ubuf__ T *giRow   = giAddr + r * dimLen + dimOff;

            // load go/in once and cast
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goB16, goRow);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN1B16, goN1Row);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(goN2B16, goN2Row);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(inB16, inRow);
            MicroAPI::Cast<float, T, castTraitB162B32>(goB32, goB16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(goN1B32, goN1B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(goN2B32, goN2B16, fullMask);
            MicroAPI::Cast<float, T, castTraitB162B32>(inB32, inB16, fullMask);

            // grad_input accumulate: acc = go*w2 + goN1*w1 + goN2*w0
            MicroAPI::Mul(accB32, goB32, w2B32, fullMask);
            MicroAPI::Mul(mulB32, goN1B32, w1B32, fullMask);
            MicroAPI::Add(accB32, accB32, mulB32, fullMask);
            MicroAPI::Mul(mulB32, goN2B32, w0B32, fullMask);
            MicroAPI::Add(accB32, accB32, mulB32, fullMask);
            MicroAPI::Cast<T, float, castTraitB322B16>(giB16, accB32, fullMask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(giRow, giB16, fullMask);

            // products
            MicroAPI::Mul(mulB32, goB32, inB32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p2Addr + r * dimLen + dimOff, mulB32, fullMask);
            MicroAPI::Mul(mulB32, goN1B32, inB32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p1Addr + r * dimLen + dimOff, mulB32, fullMask);
            MicroAPI::Mul(mulB32, goN2B32, inB32, fullMask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(p0Addr + r * dimLen + dimOff, mulB32, fullMask);
        }
        dimOff += B32_REP_SIZE;
    }
}

template <typename T>
__simd_vf__ void CastRowB32ToDT(__ubuf__ float *srcAddr, __ubuf__ T *dstAddr, uint32_t dimLen)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t dimLoopNum = dimLen / B32_REP_SIZE;
    uint32_t dimOff = 0;
    for (uint32_t dl = 0; dl < dimLoopNum; ++dl) {
        MicroAPI::RegTensor<float> s;
        MicroAPI::RegTensor<T> dB16;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(s, srcAddr + dimOff);
        MicroAPI::Cast<T, float, castTraitB322B16>(dB16, s, fullMask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr + dimOff, dB16, fullMask);
        dimOff += B32_REP_SIZE;
    }
}


template <typename T>
__aicore__ inline void DoCastToDT(LocalTensor<float> &inUb, uint32_t dimLen, LocalTensor<T> &outUb)
{
    __ubuf__ float *src = (__ubuf__ float *)inUb.GetPhyAddr();
    __ubuf__ T *dst = (__ubuf__ T *)outUb.GetPhyAddr();
    CastRowB32ToDT<T>(src, dst, dimLen);
}


template <typename T>
__aicore__ inline void DoGradInputWeight(LocalTensor<T> &goUb, LocalTensor<T> &wUb, LocalTensor<T> &inUb,
                                                  LocalTensor<T> &giUb, LocalTensor<float> &p0Ub, LocalTensor<float> &p1Ub,
                                                  LocalTensor<float> &p2Ub, uint32_t sLen, uint32_t dimLen)
{
    __ubuf__ T *goAddr = (__ubuf__ T *)goUb.GetPhyAddr();
    __ubuf__ T *wAddr  = (__ubuf__ T *)wUb.GetPhyAddr();
    __ubuf__ T *inAddr = (__ubuf__ T *)inUb.GetPhyAddr();
    __ubuf__ T *giAddr = (__ubuf__ T *)giUb.GetPhyAddr();
    __ubuf__ float *p0Addr = (__ubuf__ float *)p0Ub.GetPhyAddr();
    __ubuf__ float *p1Addr = (__ubuf__ float *)p1Ub.GetPhyAddr();
    __ubuf__ float *p2Addr = (__ubuf__ float *)p2Ub.GetPhyAddr();
    GradInputAndWeight<T>(goAddr, wAddr, inAddr, giAddr, p0Addr, p1Addr, p2Addr, sLen, dimLen);
}

}

#endif