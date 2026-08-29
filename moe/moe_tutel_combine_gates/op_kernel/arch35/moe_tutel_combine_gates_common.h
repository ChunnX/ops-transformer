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
 * \file moe_tutel_combine_gates_common.h
 * \brief Ascend 950 common definitions and deterministic reduction helpers.
 */
#ifndef MOE_TUTEL_COMBINE_GATES_COMMON_H_
#define MOE_TUTEL_COMBINE_GATES_COMMON_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/load_store_utils.h"
#include "moe_tutel_combine_gates_tiling_data.h"

namespace MoeTutelCombineGates {
using namespace AscendC;
using optiling::MoeTutelCombineGatesTilingData;
using namespace AscendC::MicroAPI;

constexpr uint32_t VL_FP32 = static_cast<uint32_t>(Ops::Base::GetVRegSize()) / sizeof(float);
constexpr int32_t DOUBLE_BUFFER = 2;

// The interval [start, start + len) assigned to one core along an axis.
struct CoreRange {
    int64_t start;
    int64_t len;
};

__aicore__ inline CoreRange SolveCoreRange(int64_t coreIdx, int64_t perCoreBase, int64_t remain)
{
    CoreRange range;
    if (coreIdx < remain) {
        range.len = perCoreBase + 1;
        range.start = coreIdx * range.len;
    } else {
        range.len = perCoreBase;
        range.start = coreIdx * perCoreBase + remain;
    }
    return range;
}

// Reduce each VL-sized product slice to one fp32 scalar. partialUb is cleared
// to its power-of-two padded length by the caller before this function runs.
template <typename T>
__aicore__ inline void BuildDotPartialsVF(__ubuf__ float *partialUb, __ubuf__ T *xUb, __ubuf__ T *yUb, uint32_t count)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> xReg;
        Reg::RegTensor<float> yReg;
        Reg::RegTensor<float> productReg;
        Reg::RegTensor<float> sumReg;
        Reg::MaskReg pregOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
        Reg::MaskReg pregLoop;

        uint32_t remain = count;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = Reg::UpdateMask<float>(remain);
            ops::LoadOneTensorForDtypeT<T>(xUb, xReg, pregLoop, i * VL_FP32);
            ops::LoadOneTensorForDtypeT<T>(yUb, yReg, pregLoop, i * VL_FP32);
            Reg::Mul(productReg, xReg, yReg, pregLoop);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, productReg, pregLoop);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(partialUb + i, sumReg, pregOne);
        }
    }
}

// Sum a power-of-two, VL-aligned fp32 scratch array in place. Each round adds
// equal left/right halves, so the merge order depends only on the logical
// reduction layout, never on core count or scheduling.
__aicore__ inline void ReducePartialTreeVF(__ubuf__ float *srcUb, uint32_t paddedCount, __ubuf__ float *dstUb,
                                           uint32_t dstOffset)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> leftReg;
        Reg::RegTensor<float> rightReg;
        Reg::RegTensor<float> sumReg;
        Reg::RegTensor<float> scalarReg;
        Reg::MaskReg pregAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t current = paddedCount;
        while (current > VL_FP32) {
            const uint32_t half = current / 2;
            const uint16_t loopCount = static_cast<uint16_t>(half / VL_FP32);
            for (uint16_t i = 0; i < loopCount; i++) {
                const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
                Reg::LoadAlign(leftReg, srcUb + offset);
                Reg::LoadAlign(rightReg, srcUb + half + offset);
                Reg::Add(sumReg, leftReg, rightReg, pregAll);
                Reg::StoreAlign(srcUb + offset, sumReg, pregAll);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            current = half;
        }
        Reg::LoadAlign(sumReg, srcUb);
        Reg::Reduce<Reg::ReduceType::SUM>(scalarReg, sumReg, pregAll);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dstUb + dstOffset, scalarReg, pregOne);
    }
}

// Cast fp32 results to the output dtype. This is the only output cast.
template <typename T>
__aicore__ inline void CastRowVF(__ubuf__ T *dstUb, __ubuf__ float *srcUb, uint32_t count)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcReg;
        Reg::MaskReg pregLoop;

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        uint32_t remain = count;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = Reg::UpdateMask<float>(remain);
            ops::LoadOneTensorForDtypeT<float>(srcUb, srcReg, pregLoop, i * VL_FP32);
            ops::StoreOneTensorForDtypeT<T>(dstUb, srcReg, pregLoop, i * VL_FP32);
        }
    }
}
} // namespace MoeTutelCombineGates

#endif // MOE_TUTEL_COMBINE_GATES_COMMON_H_
