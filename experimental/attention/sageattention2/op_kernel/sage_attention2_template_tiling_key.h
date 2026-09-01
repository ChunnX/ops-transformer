/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H_
#define SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define SAGE_TPL_2_BW 2

// Declaration order deliberately matches the bit layout documented in sage_attention2_design.md.
ASCENDC_TPL_ARGS_DECL(
    SageAttention2,
    ASCENDC_TPL_BOOL_DECL(INPUT_BF16, 0, 1),
    ASCENDC_TPL_BOOL_DECL(LAYOUT_NHD, 0, 1),
    ASCENDC_TPL_BOOL_DECL(DPAD_128, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_CAUSAL, 0, 1),
    ASCENDC_TPL_BOOL_DECL(PER_THREAD, 0, 1),
    ASCENDC_TPL_UINT_DECL(ACCUM_MODE, SAGE_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_BOOL_DECL(SMOOTH_K, 0, 1),
    ASCENDC_TPL_BOOL_DECL(SMOOTH_V, 0, 1),
    ASCENDC_TPL_BOOL_DECL(RETURN_LSE, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(INPUT_BF16, 0, 1),
        ASCENDC_TPL_BOOL_SEL(LAYOUT_NHD, 0, 1),
        ASCENDC_TPL_BOOL_SEL(DPAD_128, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_CAUSAL, 0, 1),
        ASCENDC_TPL_BOOL_SEL(PER_THREAD, 0, 1),
        ASCENDC_TPL_UINT_SEL(ACCUM_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2),
        ASCENDC_TPL_BOOL_SEL(SMOOTH_K, 0, 1),
        ASCENDC_TPL_BOOL_SEL(SMOOTH_V, 0, 1),
        ASCENDC_TPL_BOOL_SEL(RETURN_LSE, 0, 1),
    ),
);

#endif // SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H_

