#ifndef MOE_EP_DISPATCH_EPILOGUE_TILING_KEY_H
#define MOE_EP_DISPATCH_EPILOGUE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {

#define TILINGKEY_TPL_A2 0
#define TILINGKEY_TPL_A3 1
#define TILINGKEY_TPL_A5 2

ASCENDC_TPL_ARGS_DECL(MoeEpDispatchEpilogue)

ASCENDC_TPL_SEL(MoeEpDispatchEpilogue,
    ASCENDC_TPL_SEL_ENTRY(MoeEpDispatchEpilogue, A5,
        ASCENDC_TPL_TILING_STRUCT_SEL(MoeEpDispatchEpilogueTilingData)),
    ASCENDC_TPL_SEL_ENTRY(MoeEpDispatchEpilogue, A3,
        ASCENDC_TPL_TILING_STRUCT_SEL(MoeEpDispatchEpilogueTilingData)),
    ASCENDC_TPL_SEL_ENTRY(MoeEpDispatchEpilogue, A2,
        ASCENDC_TPL_TILING_STRUCT_SEL(MoeEpDispatchEpilogueA2TilingData)))

}  // namespace Mc2Tiling

#endif  // MOE_EP_DISPATCH_EPILOGUE_TILING_KEY_H