/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H
#define SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Numeric values follow ge::DataType (DT_FLOAT16=1, DT_BF16=27) for CANN code generators
#define SAGE_TPL_FP16 1
#define SAGE_TPL_BF16 27

// Registers template specialization permutations: (FP16/BF16) x (STANDARD/WIDE_STREAM)
ASCENDC_TPL_ARGS_DECL(
    SageAttention2,
    ASCENDC_TPL_DTYPE_DECL(INPUT_DTYPE, SAGE_TPL_FP16, SAGE_TPL_BF16),
    ASCENDC_TPL_BOOL_DECL(WIDE, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(INPUT_DTYPE, SAGE_TPL_FP16, SAGE_TPL_BF16),
        ASCENDC_TPL_BOOL_SEL(WIDE, 0, 1)));

#endif // SAGE_ATTENTION2_TEMPLATE_TILING_KEY_H
