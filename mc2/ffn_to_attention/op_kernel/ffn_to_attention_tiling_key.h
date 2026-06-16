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
 * \file ffn_to_attention_tiling_key.h
 * \brief
 */

#ifndef FFN_TO_ATTENTION_TILING_KEY_H
#define FFN_TO_ATTENTION_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {
ASCENDC_TPL_ARGS_DECL(FFNToAttention,
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_RANK_TABLE_MODE, 0, 1)
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_RANK_TABLE_MODE, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(FFNToAttentionTilingData)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_RANK_TABLE_MODE, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(FFNToAttentionTilingData)
    )
);
} // namespace Mc2Tiling
#endif