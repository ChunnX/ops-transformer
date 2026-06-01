/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_

#include "kernel_operator.h"

namespace QkvRmsNormRopeCacheWithKScale {
namespace MicroAPI = AscendC::MicroAPI;

constexpr uint32_t QKV_RRC_D128_FLOAT_REPEAT_SIZE = 64U;
constexpr uint32_t QKV_RRC_D128_FLOAT_REPEAT_NUM = 2U;
constexpr uint32_t QKV_RRC_D128_HALF_SIZE = 64U;

constexpr MicroAPI::CastTrait QKV_RRC_CAST_F32_TO_BF16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr MicroAPI::CastTrait QKV_RRC_CAST_BF16_TO_F32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

__simd_vf__ inline void QkRmsNormRopeD128Vf(__ubuf__ bfloat16_t *rowBf16, __ubuf__ float *gamma,
    __ubuf__ float *cosSin, __ubuf__ float *scratchFp32, __ubuf__ bfloat16_t *outBf16,
    float reciprocal, float epsilon)
{
    MicroAPI::RegTensor<float> squareSum;
    MicroAPI::RegTensor<float> reduceSum;
    MicroAPI::RegTensor<float> divisor;
    MicroAPI::RegTensor<float> rms;

    MicroAPI::MaskReg mask64 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

    for (uint32_t idx = 0U; idx < QKV_RRC_D128_FLOAT_REPEAT_NUM; ++idx) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::RegTensor<float> xFp32;
        const uint32_t offset = idx * QKV_RRC_D128_FLOAT_REPEAT_SIZE;
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xBf16, rowBf16 + offset);
        MicroAPI::Cast<float, bfloat16_t, QKV_RRC_CAST_BF16_TO_F32>(xFp32, xBf16, mask64);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(scratchFp32 + offset, xFp32, mask64);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::Duplicate<float, float>(squareSum, 0.0F);
    for (uint32_t idx = 0U; idx < QKV_RRC_D128_FLOAT_REPEAT_NUM; ++idx) {
        MicroAPI::RegTensor<float> x;
        MicroAPI::RegTensor<float> square;
        const uint32_t offset = idx * QKV_RRC_D128_FLOAT_REPEAT_SIZE;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(x, scratchFp32 + offset);
        MicroAPI::Mul(square, x, x, mask64);
        MicroAPI::Add(squareSum, squareSum, square, mask64);
    }

    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        reduceSum, squareSum, mask64);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, reduceSum, reciprocal, maskFirst);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(reduceSum, reduceSum, epsilon, maskFirst);
    MicroAPI::Sqrt(rms, reduceSum, maskFirst);
    MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(divisor, rms, mask64);

    for (uint32_t idx = 0U; idx < QKV_RRC_D128_FLOAT_REPEAT_NUM; ++idx) {
        MicroAPI::RegTensor<float> x;
        MicroAPI::RegTensor<float> gammaValue;
        const uint32_t offset = idx * QKV_RRC_D128_FLOAT_REPEAT_SIZE;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(x, scratchFp32 + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(gammaValue, gamma + offset);
        MicroAPI::Div(x, x, divisor, mask64);
        MicroAPI::Mul(x, x, gammaValue, mask64);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(scratchFp32 + offset, x, mask64);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<float> cosValue;
    MicroAPI::RegTensor<float> sinValue;
    MicroAPI::RegTensor<float> xLow;
    MicroAPI::RegTensor<float> xHigh;
    MicroAPI::RegTensor<float> outLow;
    MicroAPI::RegTensor<float> outHigh;
    MicroAPI::RegTensor<float> tmpLow;
    MicroAPI::RegTensor<float> tmpHigh;
    MicroAPI::RegTensor<bfloat16_t> outLowBf16;
    MicroAPI::RegTensor<bfloat16_t> outHighBf16;

    uint32_t halfMaskValue = QKV_RRC_D128_HALF_SIZE;
    mask64 = MicroAPI::UpdateMask<float>(halfMaskValue);

    MicroAPI::DataCopy(xLow, scratchFp32);
    MicroAPI::DataCopy(xHigh, scratchFp32 + QKV_RRC_D128_HALF_SIZE);
    MicroAPI::DataCopy(cosValue, cosSin);
    MicroAPI::DataCopy(sinValue, cosSin + QKV_RRC_D128_HALF_SIZE);

    MicroAPI::Mul(outLow, xLow, cosValue, mask64);
    MicroAPI::Mul(tmpLow, xHigh, sinValue, mask64);
    MicroAPI::Sub(outLow, outLow, tmpLow, mask64);

    MicroAPI::Mul(outHigh, xHigh, cosValue, mask64);
    MicroAPI::Mul(tmpHigh, xLow, sinValue, mask64);
    MicroAPI::Add(outHigh, outHigh, tmpHigh, mask64);

    MicroAPI::Cast<bfloat16_t, float, QKV_RRC_CAST_F32_TO_BF16>(outLowBf16, outLow, mask64);
    MicroAPI::Cast<bfloat16_t, float, QKV_RRC_CAST_F32_TO_BF16>(outHighBf16, outHigh, mask64);
    MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(outBf16, outLowBf16, mask64);
    MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(
        outBf16 + QKV_RRC_D128_HALF_SIZE, outHighBf16, mask64);
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VF_H_
