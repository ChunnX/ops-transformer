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
 * \file tile_mmad_mx.h
 * \brief
 */
#pragma once

#include "include/tensor_api/tensor.h"

namespace Blaze::Gemm::Tile {

struct MmadMx {
    template <typename Tp, const Tp& traits, typename T, typename U, typename S>
    __aicore__ inline static void Mad(
        const T& dst, const U& fm, const S& filter, uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlagCtrl,
        bool btBuffCtrl, bool initCMatrixCtrl)
    {
        mad_mx(
            dst.Data().Get(), fm.Data().Get(), filter.Data().Get(), m, k, n, unitFlagCtrl, true, btBuffCtrl,
            initCMatrixCtrl);
    }
};
}

namespace AscendC {
namespace Te {

constexpr MmadTrait MX_MMAD_TRAIT = MmadTrait{0, false, false, true, MmadType::MX};
struct MmadTraitMX {
    using TraitType = MmadTrait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};

template <>
struct MmadTraits<MmadOperation, MmadTraitMX>
    : public MmadTraits<MmadOperation, MmadTraitDefault, MmadOpWith, MmadTraitMX> {};

// Specialization for a8w4 MmadMx operation
template <>
struct MmadTraits<Blaze::Gemm::Tile::MmadMx>
    : public MmadTraits<Blaze::Gemm::Tile::MmadMx, MmadTraitDefault, Blaze::Gemm::Tile::MmadMx, MmadTraitDefault> {};
} // namespace Te
} // namespace AscendC
