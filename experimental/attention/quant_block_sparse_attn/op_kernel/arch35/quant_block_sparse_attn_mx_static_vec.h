/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_VEC_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_VEC_H_

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"

namespace FaVectorApi {

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnMxfp8Static(
    const AscendC::LocalTensor<T2> &dstTensor, const AscendC::LocalTensor<T> &expSumTensor,
    const AscendC::LocalTensor<T> &maxTensor, const AscendC::LocalTensor<T> &srcTensor,
    const AscendC::LocalTensor<T> &expMaxTensor, AscendC::LocalTensor<uint8_t> *vselrIndexesBuf,
    const AscendC::LocalTensor<uint8_t> &maskTensor,
    const AscendC::LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor, const uint32_t m, const uint32_t n,
    const uint32_t originN, const T scale, float deScaleQK, float pScale, const T minValue, float keepProb,
    const AscendC::LocalTensor<T> &preLoopMaxTensor, const AscendC::LocalTensor<T> &preLoopSumTensor,
    const AscendC::LocalTensor<T> &firstLoopSumTensor, int32_t subLoop)
{
    AscendC::LocalTensor<uint8_t> indexesTensor;
    if constexpr (AscendC::IsSameType<T2, fp8_e5m2_t>::value ||
                  AscendC::IsSameType<T2, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<T2, hifloat8_t>::value) {
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)];
    }
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdateMxfp8<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, m, n, originN,
            scale, deScaleQK, pScale, minValue, keepProb, subLoop, pScaleSubLoop0Tensor);
    } else {
        ProcessVec1DnUpdateMxfp8<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, m, n, originN,
            scale, deScaleQK, pScale, minValue, keepProb, preLoopMaxTensor, preLoopSumTensor, firstLoopSumTensor,
            subLoop, pScaleSubLoop0Tensor);
    }
}

} // namespace FaVectorApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_VEC_H_
