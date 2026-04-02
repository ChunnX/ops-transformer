#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_PROLOGUE_BLOCK_PROLOGUE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_PROLOGUE_BLOCK_PROLOGUE_H

#include "../policy/dispatch_policy.h"
#include "../utils/integral_constant.h"

namespace Prologue {

template <class DispatchPolicy, class... Args>
class BlockPrologue {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockPrologue is not implemented for this DispatchPolicy");
};

} // namespace Prologue

#include "grouped_matmul_mxfp8fp4_prologue_mx_cast_w.h"

#endif
