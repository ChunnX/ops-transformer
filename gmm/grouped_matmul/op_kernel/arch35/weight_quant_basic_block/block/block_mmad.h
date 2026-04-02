#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_BLOCK_MMAD_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_BLOCK_MMAD_H

#include "../policy/dispatch_policy.h"
#include "../utils/integral_constant.h"

namespace Block {

template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class LayoutA, class BType,
          class LayoutB, class CType, class LayoutC, class BiasType, typename = void>
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace Block

#include "grouped_matmul_mxfp8fp4_block_mmad_resplit.h"

#endif
