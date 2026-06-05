#ifndef ENGRAM_FETCH_TILING_KEY_H
#define ENGRAM_FETCH_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {
#define TILINGKEY_TPL_A5 2
#define TILINGKEY_URMA 2

ASCENDC_TPL_ARGS_DECL(EngramFetch,
                      ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5), );
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(EngramFetchTilingData)), );
} // namespace Mc2Tiling
#endif