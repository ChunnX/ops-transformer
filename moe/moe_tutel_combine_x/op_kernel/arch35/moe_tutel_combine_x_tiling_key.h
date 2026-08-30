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
 * \file moe_tutel_combine_x_tiling_key.h
 * \brief MoeTutelCombineX template tiling-key declaration for Ascend950.
 */

#ifndef MOE_TUTEL_COMBINE_X_TILING_KEY_H_
#define MOE_TUTEL_COMBINE_X_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// Preserve the established raw key 0x2710 (10000) while using template-argument dispatch.
// Fix the two 8-bit fields to 16 (0x10) and 39 (0x27), respectively.
#define MTCX_TILING_KEY_LOW_BYTE 16
#define MTCX_TILING_KEY_HIGH_BYTE 39

ASCENDC_TPL_ARGS_DECL(MoeTutelCombineX,
                      ASCENDC_TPL_UINT_DECL(keyLowByte, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
                      ASCENDC_TPL_UINT_DECL(keyHighByte, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ASCENDC_TPL_UINT_SEL(keyLowByte, ASCENDC_TPL_UI_RANGE, 1, MTCX_TILING_KEY_LOW_BYTE, MTCX_TILING_KEY_LOW_BYTE),
    ASCENDC_TPL_UINT_SEL(keyHighByte, ASCENDC_TPL_UI_RANGE, 1, MTCX_TILING_KEY_HIGH_BYTE, MTCX_TILING_KEY_HIGH_BYTE)));

#endif // MOE_TUTEL_COMBINE_X_TILING_KEY_H_
