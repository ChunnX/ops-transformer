#ifndef CATLASS_GEMV_TILE_TILE_FAULT_COPY_HPP_SELF
#define CATLASS_GEMV_TILE_TILE_FAULT_COPY_HPP_SELF

#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/gemv/tile/vec_copy_gm_to_ub.hpp"
#include "catlass/gemv/tile/vec_copy_ub_to_gm.hpp"
#include "catlass/gemv/tile/matrix_copy_gm_to_ub.hpp"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "FT_self/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"

#include "catlass/gemm/helper.hpp"
#include "FT_self/gemv/helper.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileFaultCopyGemvAiv {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;

    // the function of aiv
    using VecCopyGmToUb = Catlass::Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = FTSelf::Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Catlass::Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Catlass::Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileFaultCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aic
    using L1XType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L0AType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename FTSelf::Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;

    using CopyGmToL1A = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;
    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;

    using CopyL1ToL0A = Catlass::Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = FTSelf::Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>;
    using CopyL0CToGm = Catlass::Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;
};

}

#endif
