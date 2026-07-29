/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_CATLASS_H
#define MATMUL_REDUCE_SCATTER_CATLASS_H

#include <cstdint>
#include <type_traits>

#if !defined(CATLASS_ARCH)
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass {

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using MatmulReduceScatterArchTag = Arch::Ascend950;
#else
using MatmulReduceScatterArchTag = Arch::AtlasA2;
#endif

namespace Gemm {

// The fixpipe path keeps the operator's cross-block preload schedule. This
// policy is architecture-neutral and is consumed by the operator-local block.
template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MatmulReduceScatterPreloadPolicy : public MmadBase<MatmulReduceScatterArchTag, false> {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
template <bool HAS_BIAS_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
using MatmulReduceScatterDispatchPolicy = MmadPingpong<MatmulReduceScatterArchTag, ENABLE_UNIT_FLAG_>;
#else
template <bool HAS_BIAS_, bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
using MatmulReduceScatterDispatchPolicy = std::conditional_t<
    HAS_BIAS_, MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG_>,
    MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>>;
#endif

namespace Block {

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

template <class BiasType>
struct MatmulReduceScatterBiasElement {
    using type = typename BiasType::Element;
};

template <>
struct MatmulReduceScatterBiasElement<void> {
    using type = void;
};

// The AIV-mode kernel uses the legacy non-TLA layout objects in its outer
// scheduling layer. This adapter converts only the block invocation to the
// Ascend950 TLA API supplied by current Catlass.
template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
    class BiasType_, class LegacyTileCopy_>
class MatmulReduceScatterBlockMmadAscend950 {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementBias = typename MatmulReduceScatterBiasElement<BiasType_>::type;
    using BiasStorageElement = std::conditional_t<std::is_void_v<ElementBias>, uint8_t, ElementBias>;

private:
    template <class Element, class LayoutTag>
    CATLASS_DEVICE static auto MakeTlaLayout(LayoutTag const &layoutTag)
    {
        if constexpr (std::is_same_v<LayoutTag, layout::zN> || std::is_same_v<LayoutTag, layout::nZ>) {
            return tla::MakeLayout<Element, LayoutTag>(layoutTag.orgShape(0), layoutTag.orgShape(1));
        } else if constexpr (std::is_same_v<LayoutTag, layout::VectorLayout>) {
            return tla::MakeLayout(layoutTag.shape(0));
        } else {
            return tla::MakeLayout<Element, LayoutTag>(layoutTag.shape(0), layoutTag.shape(1));
        }
    }

    using TlaL1TileShape = tla::Shape<tla::Int<L1TileShape::M>, tla::Int<L1TileShape::N>, tla::Int<L1TileShape::K>>;
    using TlaL0TileShape = tla::Shape<tla::Int<L0TileShape::M>, tla::Int<L0TileShape::N>, tla::Int<L0TileShape::K>>;
    using TlaTileCopy = Gemm::Tile::PackedTileCopyTla<
        ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementBias>;
    using Impl = BlockMmadTla<
        DispatchPolicy, TlaL1TileShape, TlaL0TileShape, ElementA, ElementB, ElementC, ElementBias, TlaTileCopy>;

public:
    CATLASS_DEVICE explicit MatmulReduceScatterBlockMmadAscend950(Arch::Resource<ArchTag> &resource)
        : impl_(resource)
    {}

    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &, AscendC::GlobalTensor<ElementB> const &,
        GemmCoord const &actualShape, GemmCoord const &, bool, bool)
    {
        // MmadPingpong owns the intra-block pipeline on Ascend950; the legacy
        // AtlasA2 next-block preload arguments are intentionally not consumed.
        Run(gmA, layoutA, gmB, layoutB, gmC, layoutC, actualShape);
    }

    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<BiasStorageElement> const &gmBias, GemmCoord const &actualShape)
    {
        auto tensorA = tla::MakeTensor(gmA, MakeTlaLayout<ElementA>(layoutA), Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, MakeTlaLayout<ElementB>(layoutB), Arch::PositionGM{});
        auto layoutBlockC = layoutC.GetTileLayout(actualShape.GetCoordMN());
        auto tensorC = tla::MakeTensor(gmC, tla::MakeLayoutFromTag(layoutBlockC), Arch::PositionGM{});
        auto layoutBias = tla::MakeLayout(actualShape.n());
        auto tensorBias = tla::MakeTensor(gmBias, layoutBias, Arch::PositionGM{});
        impl_(tensorA, tensorB, tensorC, actualShape, tensorBias);
    }

private:
    CATLASS_DEVICE void Run(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC, GemmCoord const &actualShape)
    {
        auto tensorA = tla::MakeTensor(gmA, MakeTlaLayout<ElementA>(layoutA), Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, MakeTlaLayout<ElementB>(layoutB), Arch::PositionGM{});
        auto layoutBlockC = layoutC.GetTileLayout(actualShape.GetCoordMN());
        auto tensorC = tla::MakeTensor(gmC, tla::MakeLayoutFromTag(layoutBlockC), Arch::PositionGM{});
        impl_(tensorA, tensorB, tensorC, actualShape);
    }

    Impl impl_;
};

template <
    class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
    class BiasType = void, class TileCopy = Gemm::Tile::TileCopy<
        typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>>
using MatmulReduceScatterBlockMmad = MatmulReduceScatterBlockMmadAscend950<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType, TileCopy>;

#else

template <
    class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
    class BiasType = void, class TileCopy = Gemm::Tile::TileCopy<
        typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>>
using MatmulReduceScatterBlockMmad = BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType, TileCopy>;

#endif

} // namespace Block
} // namespace Gemm
} // namespace Catlass

#endif // MATMUL_REDUCE_SCATTER_CATLASS_H
