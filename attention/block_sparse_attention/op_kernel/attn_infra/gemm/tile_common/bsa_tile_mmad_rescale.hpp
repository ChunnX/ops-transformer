/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_TILE_MMAD_RESCALE_HPP
#define BSA_TILE_MMAD_RESCALE_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_arch.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Tile {

template <class ArchTag>
struct TileMmadRescaleTla;

template <>
struct TileMmadRescaleTla<Arch::AtlasA5> {
    __aicore__ inline TileMmadRescaleTla() {}

    template <class TensorC, class TensorA, class TensorB>
    __aicore__ inline void operator()(TensorC const &l0CTensor,
        TensorA const &l0ATensor,
        TensorB const &l0BTensor,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        bool initC = true,
        uint8_t unitFlag = 0)
    {
        static_assert(AscendC::IsSameType<typename TensorC::Element, float>::value &&
                          AscendC::IsSameType<typename TensorA::Element, float>::value &&
                          AscendC::IsSameType<typename TensorB::Element, float>::value,
            "TileMmadRescaleTla only supports FP32 A/B/C tensors");

        AscendC::MmadParams rescaleMmadParams;
        rescaleMmadParams.m = m;
        rescaleMmadParams.n = n;
        rescaleMmadParams.k = k;
        rescaleMmadParams.unitFlag = unitFlag;
        rescaleMmadParams.cmatrixInitVal = initC;
#if (__CCE_AICORE__ == 310)
        rescaleMmadParams.disableGemv = true;
#endif

        auto cOffset = l0CTensor.layout()(l0CTensor.coord());
        auto aOffset = l0ATensor.layout()(l0ATensor.coord());
        auto bOffset = l0BTensor.layout()(l0BTensor.coord());
        AscendC::Mmad(l0CTensor.data()[cOffset],
            l0ATensor.data()[aOffset],
            l0BTensor.data()[bOffset],
            rescaleMmadParams);
    }
};

} // namespace NpuArch::Gemm::Tile

#endif // BSA_TILE_MMAD_RESCALE_HPP
