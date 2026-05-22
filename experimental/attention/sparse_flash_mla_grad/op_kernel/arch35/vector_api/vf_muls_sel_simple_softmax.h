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
 * \file vf_muls_sel_simple_softmax.h
 */
#ifndef VF_MULS_SEL_SIMPLE_SOFTMAX_H
#define VF_MULS_SEL_SIMPLE_SOFTMAX_H
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************

MulsSelSimpleSoftMax *

************************************************************************************************* /

@INGROUP MulsSelSimpleSoftMax
brief compute exp(sel(x * scale, mask) - max) / expSum
param [out] dst output LocalTensor
param [in] inSumTensor input expsum of last axis
param [in] inMaxTensor input max value of last axis
param [in] src input LocalTensor
param [in] tiling softmaxtiling
*/
template <typename T1, typename T, uint32_t srcN, bool hasAtten, bool hasPse, bool isDeter, bool IsSparseIndicesExist>
__simd_vf__ inline void MulsSelVFPseType1(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t softmaxL1LocalInt, uint64_t sumLocalInt, uint64_t maxLocalInt, 
                                          uint64_t maskLocalInt, uint64_t srcLocalIntTail, uint64_t dstLocalIntTail, uint64_t softmaxL1LocalIntTail, 
                                          uint64_t dstLocalIntTailZero, uint64_t maskLocalIntTail, uint64_t pseLocalInt, uint64_t pseLocalIntTail, 
                                          uint32_t tailSize, uint32_t pseStride, uint16_t repeatTimes, const T scale, const T minValue, 
                                          uint32_t pseTailStride, uint32_t srcM, int64_t lseLocalInt, float pScalar, bool needCleanSoftmaxL1)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregMax;
    RegTensor<float> vregSum;
    RegTensor<float> vregLse;

    RegTensor<float> vregSub;
    RegTensor<float> vregExp;
    RegTensor<float> vregDiv;
    RegTensor<float> vregMuls;
    // tail
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregSubTail;
    RegTensor<float> vregExpTail;
    RegTensor<float> vregDivTail;

    RegTensor<float> vregMulsTail;
    RegTensor<float> vregSoftmaxSum;
    RegTensor<float> vregReduceSum;

    UnalignRegForStore uregRes;
    UnalignRegForStore uregResSrc;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregAccu = CreateMask<T, MaskPattern::VL1>();
    MaskReg pregTailExe = UpdateMask<float>(tailSize);

    if constexpr(IsSparseIndicesExist) {
        if (needCleanSoftmaxL1) {
            Duplicate(vregSoftmaxSum, 0.0f);
            StoreAlign<float>(
                ((__ubuf__ float *&)softmaxL1LocalInt), vregSoftmaxSum, pregFullExe);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }

    // 64 * 128 max/sum: 64 * 8
    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        if constexpr (IsSparseIndicesExist) {
            uint64_t srcOffset = softmaxL1LocalInt + 4 * m;
            LoadUnAlignPre<float>(uregResSrc, ((__ubuf__ float *&)srcOffset));
            LoadUnAlign<float>(vregSoftmaxSum, uregResSrc, ((__ubuf__ float *&)srcOffset));
        }
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregLse, ((__ubuf__ float *&)lseLocalInt), 1);
        // 1 -> 64 【64，128】 【64，2，64】 64
        for (uint16_t n = 0; n < repeatTimes; n++) {
            LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vregSrc, ((__ubuf__ float *&)srcLocalInt), srcN);
            Muls(vregMuls, vregSrc, scale, pregFullExe);
            Sub(vregSub, vregMuls, vregLse, pregFullExe);
            Exp(vregExp, vregSub, pregFullExe);
            if constexpr (IsSparseIndicesExist) {
                Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregExp, pregFullExe);
                Add(vregSoftmaxSum, vregSoftmaxSum, vregReduceSum, pregAccu);
            }
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ((__ubuf__ float *&)dstLocalInt), vregExp, srcN, pregFullExe);
        }
        // tailLoop
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail), 128);
        Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
        Sub(vregSubTail, vregMulsTail, vregLse, pregTailExe);
        Exp(vregExpTail, vregSubTail, pregTailExe);
        if constexpr (IsSparseIndicesExist) {
            Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregExpTail, pregTailExe);
            Add(vregSoftmaxSum, vregSoftmaxSum, vregReduceSum, pregAccu);

            uint64_t dstOffset = softmaxL1LocalInt + 4 * m;
            StoreUnAlign<T>(
                    ((__ubuf__ T *&)dstOffset), vregSoftmaxSum, uregRes, 1);
            StoreUnAlignPost<T>(
                    ((__ubuf__ T *&)dstOffset), uregRes, 0);
        }
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
            ((__ubuf__ float *&)dstLocalIntTail), vregExpTail, 128, pregFullExe);
    }
}

template <typename T1, typename T, uint32_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0, bool IsSparseIndicesExist = 0>
__aicore__ inline void MulsSelSimpleSoftMax(const LocalTensor<float> &dstTensor, const LocalTensor<float> &inLseTensor,
                                            const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &srcTensor,
                                            const LocalTensor<float> &softmaxL1Tensor, const LocalTensor<T1> &pseTensor,
                                            const LocalTensor<uint8_t> &maskTensor, const T scale, const T minValue,
                                            bool needCleanSoftmaxL1, uint32_t srcM, uint32_t realN = srcN, float pScalar = 1.0, 
                                            uint32_t pseType = 1, uint32_t pse_layout_type = 1, float posShift = -1.0)
{
    const uint32_t fullExeSize = 64;
    uint16_t repeatTimes = srcN / fullExeSize - 1;
    uint32_t tailSize = realN % fullExeSize == 0 ? fullExeSize : realN % fullExeSize;
    uint32_t pseStride = pse_layout_type == 1 ? 0 : srcN;
    uint32_t pseTailStride = pse_layout_type == 1 ? 0 : 128;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t softmaxL1LocalInt = softmaxL1Tensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t lseLocalInt = inLseTensor.GetPhyAddr();
    uint64_t sumLocalInt = inLseTensor.GetPhyAddr();
    uint64_t maxLocalInt = inMaxTensor.GetPhyAddr();
    uint64_t maskLocalInt = maskTensor.GetPhyAddr();
    uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t softmaxL1LocalIntTail = softmaxL1Tensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTailZero = dstTensor.GetPhyAddr() + fullExeSize * 4 * (repeatTimes + 1);
    uint64_t maskLocalIntTail = maskTensor.GetPhyAddr() + fullExeSize * repeatTimes;
    uint64_t pseLocalInt = pseTensor.GetPhyAddr();
    uint64_t pseLocalIntTail = pseTensor.GetPhyAddr() + fullExeSize * sizeof(T1) * repeatTimes;
    MulsSelVFPseType1<T1, T, srcN, hasAtten, hasPse, isDeter, IsSparseIndicesExist>(srcLocalInt, dstLocalInt, softmaxL1LocalInt, sumLocalInt, 
        maxLocalInt, maskLocalInt, srcLocalIntTail, dstLocalIntTail, softmaxL1LocalIntTail, dstLocalIntTailZero, maskLocalIntTail, 
        pseLocalInt, pseLocalIntTail, tailSize, pseStride, repeatTimes, scale, minValue, pseTailStride, srcM, lseLocalInt, pScalar, needCleanSoftmaxL1);
}

template <typename T, uint32_t srcN>
__simd_vf__ inline void SoftmaxL1ReduceVF(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t softmaxL1LocalInt, uint64_t srcLocalIntTail, uint64_t dstLocalIntTail, uint64_t softmaxL1LocalIntTail, 
                                          uint32_t tailSize, uint16_t repeatTimes, uint32_t srcM, float pScalar)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregSoftmaxSum;
    RegTensor<float> vregReduceSumMuls;
    // tail
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregReduceSumMulsTail;

    UnalignRegForStore uregResSrc;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregTailExe = UpdateMask<float>(tailSize);

    Duplicate(vregReduceSumMuls, 0.0f);
    Duplicate(vregReduceSumMulsTail, 0.0f);
    // 64 * 128 max/sum: 64 * 8
    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregSoftmaxSum, ((__ubuf__ float *&)softmaxL1LocalInt), 1);
        // 1 -> 64 【64，128】 【64，2，64】 64
        for (uint16_t n = 0; n < repeatTimes; n++) {
            LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vregSrc, ((__ubuf__ float *&)srcLocalInt), srcN);
            Div(vregSrc, vregSrc, vregSoftmaxSum, pregFullExe);
            Add(vregReduceSumMuls, vregReduceSumMuls, vregSrc, pregFullExe);
        }
        // tailLoop
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail), 128);
        Div(vregSrcTail, vregSrcTail, vregSoftmaxSum, pregTailExe);
        Add(vregReduceSumMulsTail, vregReduceSumMulsTail, vregSrcTail, pregTailExe);
    }
    for (uint16_t n = 0; n < repeatTimes; n++) {
        Muls(vregReduceSumMuls, vregReduceSumMuls, pScalar, pregFullExe);
        StoreAlign<float>(((__ubuf__ float *&)dstLocalInt), vregReduceSumMuls, pregFullExe);
    }
    Muls(vregReduceSumMulsTail, vregReduceSumMulsTail, pScalar, pregTailExe);
    StoreAlign<float>(((__ubuf__ float *&)dstLocalIntTail), vregReduceSumMulsTail, pregFullExe);
}
#else
template <typename T1, typename T, uint32_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0, bool IsSparseIndicesExist = 0>
__aicore__ inline void MulsSelSimpleSoftMax(const LocalTensor<float> &dstTensor, const LocalTensor<float> &inLseTensor,
                                            const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &srcTensor,
                                            const LocalTensor<float> &softmaxL1Tensor, const LocalTensor<T1> &pseTensor,
                                            const LocalTensor<uint8_t> &maskTensor, const T scale, const T minValue,
                                            uint32_t srcM, uint32_t realN = srcN, float pScalar = 1.0, 
                                            uint32_t pseType = 1, uint32_t pse_layout_type = 1, float posShift = -1.0)
{
}
#endif
} // namespace AscendC

#endif // MY_SIMPLE_SOFTMAX_INTERFACE_H