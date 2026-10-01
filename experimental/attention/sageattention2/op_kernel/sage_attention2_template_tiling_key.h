#ifndef SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H
#define SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Numeric values follow ge::DataType (DT_FLOAT16=1, DT_BF16=27) for CANN code generators
#define SAGE_TPL_FP16 1
#define SAGE_TPL_BF16 27

// Template specialization permutations (bit layout verified on device):
//   [0:8)  INPUT_DTYPE  ge::DataType code (DT_FLOAT16=1, DT_BF16=27)
//   [8]    WIDE         engine selector (0 -> standard stream, 1 -> wide stream)
//   [9]    DP128        padded head_dim selector (0 -> DP64, 1 -> DP128)
//   [10]   CAUSAL       attention mask mode
// CAUSAL/DP128 are compile-time so each binary ships exactly one engine
// specialization (no runtime dispatch over dead instances, minimal icache
// footprint). Declaration order is load-bearing: the codegen packs key bits in
// declaration order and the host GET_TPL_TILING_KEY call must match.
ASCENDC_TPL_ARGS_DECL(
    SageAttention2,
    ASCENDC_TPL_DTYPE_DECL(INPUT_DTYPE, SAGE_TPL_FP16, SAGE_TPL_BF16),
    ASCENDC_TPL_BOOL_DECL(WIDE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(DP128, 0, 1),
    ASCENDC_TPL_BOOL_DECL(CAUSAL, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(INPUT_DTYPE, SAGE_TPL_FP16, SAGE_TPL_BF16),
        ASCENDC_TPL_BOOL_SEL(WIDE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(DP128, 0, 1),
        ASCENDC_TPL_BOOL_SEL(CAUSAL, 0, 1)));

#endif // SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H
