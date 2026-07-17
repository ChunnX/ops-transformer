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
 * \file fused_moe_small_bs_w4a16_tiling_key.h
 * \brief fused_moe_small_bs_w4a16 tiling key declare
 */

#ifndef FUSED_MOE_SMALL_BS_W4A16_TILING_KEY_H_
#define FUSED_MOE_SMALL_BS_W4A16_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// schMode: 区分输入数据类型
#define FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_FP16 0
#define FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_BF16 1

ASCENDC_TPL_ARGS_DECL(FusedMoeSmallBsW4a16,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST,
        FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_FP16, FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_BF16)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
            FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_FP16, FUSED_MOE_SMALL_BS_W4A16_SCH_MODE_BF16)));

#endif // FUSED_MOE_SMALL_BS_W4A16_TILING_KEY_H_
