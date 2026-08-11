#ifndef CATLASS_GEMV_TILE_TILE_THRESHOLD_HPP
#define CATLASS_GEMV_TILE_TILE_THRESHOLD_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;
template <
    /// Tag indicating architecture
    class ArchTag,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    class AType,
    class XType,
    class YType,
    class BiasType = void
>
struct TileThreCalc
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileThreCalc, can not find the specialization.");
};

}

#include "catlass/gemv/tile/tile_threshold_compute.hpp"
#include "catlass/gemv/tile/tile_threshold_mean_max_std_fused.hpp"
#include "catlass/gemv/tile/tile_threshold_robust.hpp"
#include "catlass/gemv/tile/tile_threshold_simplified.hpp"
#endif
