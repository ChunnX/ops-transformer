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
 * \file moe_tutel_combine_gates_tiling_key.h
 * \brief MoeTutelCombineGates template tiling-key declaration for Ascend 950.
 */
#ifndef MOE_TUTEL_COMBINE_GATES_TILING_KEY_H_
#define MOE_TUTEL_COMBINE_GATES_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define MTCG_TPL_NOT_CUT_H 0
#define MTCG_TPL_CUT_H 1

ASCENDC_TPL_ARGS_DECL(MoeTutelCombineGates, ASCENDC_TPL_UINT_DECL(schMode, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                                                                  MTCG_TPL_NOT_CUT_H, MTCG_TPL_CUT_H));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, MTCG_TPL_NOT_CUT_H)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, MTCG_TPL_CUT_H)));

#endif // MOE_TUTEL_COMBINE_GATES_TILING_KEY_H_
