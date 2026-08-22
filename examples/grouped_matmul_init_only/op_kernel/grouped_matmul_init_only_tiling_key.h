/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_init_only_tiling_key.h
 * \brief grouped_matmul_init_only tiling key declare（单一 dtype fp16，tilingKey=0）
 */

#ifndef __GROUPED_MATMUL_INIT_ONLY_TILING_KEY_H__
#define __GROUPED_MATMUL_INIT_ONLY_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define GMM_INIT_ONLY_TPL_SCH_MODE_0 0

ASCENDC_TPL_ARGS_DECL(GroupedMatmulInitOnly,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, GMM_INIT_ONLY_TPL_SCH_MODE_0)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, GMM_INIT_ONLY_TPL_SCH_MODE_0)));

#endif
