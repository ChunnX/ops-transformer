/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEMPLATE_TILING_KEY_FLASH_ATTN_TND_MASK_H_
#define TEMPLATE_TILING_KEY_FLASH_ATTN_TND_MASK_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "../utils/flash_attn_common_def.h"
#include "flash_attn_tiling_data.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

ASCENDC_TPL_ARGS_DECL(FlashAttnTndMask,
                      ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 2, 2),
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 0),
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, true, true),
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 0), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_TND),
                                     ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_BOOL_SEL(HasAttenMask, true),
                                     ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttnTilingData)), );

#endif // TEMPLATE_TILING_KEY_FLASH_ATTN_TND_MASK_H_
