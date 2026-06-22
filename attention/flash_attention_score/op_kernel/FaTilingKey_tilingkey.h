#include "ascendc/host_api/tiling/template_argument.h"

#define ASCENDC_TPL_1_BW 1

ASCENDC_TPL_ARGS_DECL(flash_attention_score,
    // bit:0-0
    ASCENDC_TPL_UINT_DECL(NeedAttnMask, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(NeedAttnMask, ASCENDC_TPL_UI_LIST, 0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(NeedAttnMask, ASCENDC_TPL_UI_LIST, 1)
    )
);
