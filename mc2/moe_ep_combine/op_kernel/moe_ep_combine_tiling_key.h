#ifndef MOE_EP_COMBINE_TILING_KEY_H
#define MOE_EP_COMBINE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {

#define TILINGKEY_TPL_A5 2

ASCENDC_TPL_ARGS_DECL(MoeEpCombine,
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_HAS_TOPK_WEIGHTS, 0, 1))

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_HAS_TOPK_WEIGHTS, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(MoeEpCombineTilingData)),
);

}  // namespace Mc2Tiling
#endif  // MOE_EP_COMBINE_TILING_KEY_H