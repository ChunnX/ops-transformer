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
 * \file copy_scale_l1_to_l0b.h
 * \brief Tile helper that copies MX scaleB data from L1 to L0B.
 */

#pragma once
#include "impl/atom/cube_datamove/copy_l12l0.h"
#include "kernel_utils/common_utils.h"

namespace Tile {
struct CopyL12L0MxScaleB3510 {
    static constexpr uint64_t MXFP_DIVISOR = 64UL;
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        static_assert(
            AscendC::Std::is_one_of_v<
                AscendC::Std::tuple<dstType, srcType>, AscendC::Std::tuple<__cb__ fp8_e8m0_t, __cbuf__ fp8_e8m0_t>>,
            "The data type is not supported.");
        uint16_t nStartPosition = CeilDiv(AscendC::Std::get<1>(coord), AscendC::BLOCK_CUBE);
        uint16_t kStartPosition = CeilDiv(AscendC::Std::get<0>(coord), MXFP_DIVISOR);
        auto nStep = AscendC::Std::get<1>(AscendC::Std::get<1>(dst.Layout().Shape()));
        auto kStep = AscendC::Std::get<1>(AscendC::Std::get<0>(dst.Layout().Shape()));
        auto srcStride = AscendC::Std::get<1>(AscendC::Std::get<1>(src.Layout().Stride())) >> 5;
        auto dstStride = kStep;
        // The intrinsic takes a 16-byte unit address, hence the right shift.
        uint64_t mxDstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.Data().Get())) >> 4;
        load_cbuf_to_cb_mx(
            mxDstAddr, static_cast<__cbuf__ void*>(src.Data().Get()), nStartPosition, kStartPosition, nStep, kStep,
            srcStride, dstStride);
    }
};

// Expose this helper through TE's generic copy-trait interface.
} // namespace Tile

template <>
struct AscendC::Te::CopyTraits<::Tile::CopyL12L0MxScaleB3510>
    : public CopyTraits<
        ::Tile::CopyL12L0MxScaleB3510, LoadDataTraitDefault, ::Tile::CopyL12L0MxScaleB3510,
        LoadDataTraitDefault> {};

namespace AscendC::Te {
constexpr LoadDataTrait LOAD_DATA_B_TRAIT{true};

// Reuse the standard MMAD traits but force the B-side load trait required by
// the MX scale copy helper above.
struct LoadData2BTrait {
    using TraitType = LoadDataTrait;
    static constexpr const TraitType value = LOAD_DATA_B_TRAIT;
};
}

