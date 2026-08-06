/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_FP8_STATIC_VEC_H_
#define QUANT_BLOCK_SPARSE_ATTN_FP8_STATIC_VEC_H_

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"

namespace FaVectorApi {

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint32_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnStatic(
    const AscendC::LocalTensor<T2> &dstTensor, const AscendC::LocalTensor<T> &expSumTensor,
    const AscendC::LocalTensor<T> &maxTensor, const AscendC::LocalTensor<T> &srcTensor,
    const AscendC::LocalTensor<T> &expMaxTensor, AscendC::LocalTensor<uint8_t> *vselrIndexesBuf,
    const AscendC::LocalTensor<uint8_t> &maskTensor, const AscendC::LocalTensor<T> &qScaleTensor,
    const AscendC::LocalTensor<T> &kScaleTensor, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, bool needAtten, const float pScale = 1.0f)
{
    __ubuf__ uint32_t *qGatherIdxUb =
        (__ubuf__ uint32_t *)(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::QSCALE_GATHER_INDEX)].GetPhyAddr());
    AscendC::LocalTensor<uint8_t> indexesTensor =
        vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)];
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdate<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, qScaleTensor,
            kScaleTensor, m, n, originN, scale, minValue, needAtten, pScale, qGatherIdxUb);
    } else {
        ProcessVec1DnUpdate<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, qScaleTensor,
            kScaleTensor, m, n, originN, scale, minValue, needAtten, pScale, qGatherIdxUb);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint32_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnPerTokenHeadStatic(
    const AscendC::LocalTensor<T2> &dstTensor, const AscendC::LocalTensor<T> &expSumTensor,
    const AscendC::LocalTensor<T> &maxTensor, const AscendC::LocalTensor<T> &srcTensor,
    const AscendC::LocalTensor<T> &expMaxTensor, AscendC::LocalTensor<uint8_t> *vselrIndexesBuf,
    const AscendC::LocalTensor<uint8_t> &maskTensor, const AscendC::LocalTensor<T> &qScaleTensor,
    const AscendC::LocalTensor<T> &kScaleTensor, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, const T minValue, bool needAtten, const float pScale = 1.0f)
{
    AscendC::LocalTensor<uint8_t> indexesTensor =
        vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)];
    if constexpr (!isUpdate) {
        ProcessVec1DnNoUpdatePerTokenHead<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, qScaleTensor,
            kScaleTensor, m, n, originN, scale, minValue, needAtten, pScale);
    } else {
        ProcessVec1DnUpdatePerTokenHead<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, qScaleTensor,
            kScaleTensor, m, n, originN, scale, minValue, needAtten, pScale);
    }
}

} // namespace FaVectorApi

#endif // QUANT_BLOCK_SPARSE_ATTN_FP8_STATIC_VEC_H_
