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
 * \file BatchToSpace_tiling_key.h
 * \brief BatchToSpace tiling key declare
 */

#ifndef __BATCHTOSPACE_TILING_KEY_H__
#define __BATCHTOSPACE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_SCH_MODE_IS_LARGE_C 0
#define TPL_SCH_MODE_IS_SMALL_C 1
 
ASCENDC_TPL_ARGS_DECL(Batchtospace,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_IS_LARGE_C, TPL_SCH_MODE_IS_SMALL_C)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, TPL_SCH_MODE_IS_LARGE_C, TPL_SCH_MODE_IS_SMALL_C)));

#endif