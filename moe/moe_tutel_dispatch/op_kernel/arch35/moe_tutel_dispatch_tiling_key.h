/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_dispatch_tiling_key.h
 * \brief Ascend 950 template tiling-key declaration.
 */

#ifndef MOE_TUTEL_DISPATCH_TILING_KEY_H_
#define MOE_TUTEL_DISPATCH_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define MOE_TUTEL_DISPATCH_FULL_ROW 0
#define MOE_TUTEL_DISPATCH_SPLIT_H 1
#define MOE_TUTEL_DISPATCH_EMPTY_OUTPUT 2
#define MOE_TUTEL_DISPATCH_ZERO_ONLY 3

ASCENDC_TPL_ARGS_DECL(MoeTutelDispatch,
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            MOE_TUTEL_DISPATCH_FULL_ROW, MOE_TUTEL_DISPATCH_SPLIT_H,
                                            MOE_TUTEL_DISPATCH_EMPTY_OUTPUT, MOE_TUTEL_DISPATCH_ZERO_ONLY));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, MOE_TUTEL_DISPATCH_FULL_ROW))
        ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                             ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, MOE_TUTEL_DISPATCH_SPLIT_H))
            ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                                 ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST,
                                                      MOE_TUTEL_DISPATCH_EMPTY_OUTPUT))
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                                     ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST,
                                                          MOE_TUTEL_DISPATCH_ZERO_ONLY)));

#endif // MOE_TUTEL_DISPATCH_TILING_KEY_H_
