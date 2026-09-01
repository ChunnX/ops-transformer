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
 * \file scale_mx_bias.h
 * \brief Scale bias for MX MMAD accumulation.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Gemm::Tile {

template <typename BiasType_>
struct ScaleMxBiasParams {
    uint16_t loopNum;
    __ubuf__ BiasType_* biasInAddr;
    __ubuf__ BiasType_* biasOutAddr;
};

template <typename BiasType>
__simd_vf__ inline void ScaleMxBiasVf(ScaleMxBiasParams<BiasType> params);

template <typename BiasType, typename BiasInTensor, typename BiasOutTensor>
__aicore__ inline void ScaleMxBias(const BiasInTensor& biasInTensor, const BiasOutTensor& biasOutTensor)
{
    // This specialized tile consumes a contiguous bias vector. Reject other layouts
    // at instantiation time instead of silently treating their data as contiguous.
    using BiasInLayoutPattern = AscendC::Te::GetLayoutPattern<typename BiasInTensor::layoutType>;
    using BiasOutLayoutPattern = AscendC::Te::GetLayoutPattern<typename BiasOutTensor::layoutType>;
    static_assert(
        AscendC::Std::is_same_v<BiasInLayoutPattern, AscendC::Te::NDExtLayoutPtn> &&
            AscendC::Std::is_same_v<BiasOutLayoutPattern, AscendC::Te::NDExtLayoutPtn>,
        "ScaleMxBias requires contiguous NDExt bias tensors");

    constexpr uint64_t VECTOR_REG_BYTE_SIZE = static_cast<uint64_t>(asc_get_vf_len());
    constexpr uint64_t VECTOR_ELEMENTS = VECTOR_REG_BYTE_SIZE / sizeof(BiasType);
    uint64_t elementCount = static_cast<uint64_t>(AscendC::Te::GetTotalColumnShape(biasInTensor.Layout()));
    ScaleMxBiasParams<BiasType> params{
        static_cast<uint16_t>(CeilDiv(elementCount, VECTOR_ELEMENTS)), (__ubuf__ BiasType*)biasInTensor.Data().Get(),
        (__ubuf__ BiasType*)biasOutTensor.Data().Get()};
    asc_vf_call<ScaleMxBiasVf<BiasType>>(params);
}

template <typename BiasType>
__simd_vf__ inline void ScaleMxBiasVf(ScaleMxBiasParams<BiasType> params)
{
    namespace MicroAPI = AscendC::MicroAPI;
    constexpr uint64_t VECTOR_REG_BYTE_SIZE = static_cast<uint64_t>(asc_get_vf_len());
    constexpr uint64_t VECTOR_ELEMENTS = VECTOR_REG_BYTE_SIZE / sizeof(BiasType);
    constexpr BiasType MX_BIAS_FACTOR = static_cast<BiasType>(0.015625f);

    MicroAPI::RegTensor<BiasType> biasReg;
    MicroAPI::RegTensor<BiasType> factorReg;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<BiasType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate<BiasType, MicroAPI::MaskMergeMode::ZEROING>(factorReg, MX_BIAS_FACTOR, mask);
    for (uint16_t loopIdx = 0; loopIdx < params.loopNum; ++loopIdx) {
        MicroAPI::AddrReg addr = MicroAPI::CreateAddrReg<BiasType>(loopIdx, VECTOR_ELEMENTS);
        MicroAPI::LoadAlign<BiasType, MicroAPI::LoadDist::DIST_NORM>(biasReg, params.biasInAddr, addr);
        MicroAPI::Mul(biasReg, biasReg, factorReg, mask);
        MicroAPI::StoreAlign<BiasType, MicroAPI::StoreDist::DIST_NORM_B16>(params.biasOutAddr, biasReg, addr, mask);
    }
}

} // namespace Blaze::Gemm::Tile
