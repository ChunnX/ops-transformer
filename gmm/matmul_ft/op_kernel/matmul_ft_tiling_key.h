/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef MATMUL_FT_TILING_KEY_H
#define MATMUL_FT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define MATMUL_FT_TPL_SCH_MODE_BF16 0

ASCENDC_TPL_ARGS_DECL(MatmulFt,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, MATMUL_FT_TPL_SCH_MODE_BF16));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, MATMUL_FT_TPL_SCH_MODE_BF16)));

#endif
