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
 * \file attention_to_ffn_tiling_key.h
 * \brief
 */
#ifndef ATTENTION_TO_FFN_V2_TILING_KEY_H
#define ATTENTION_TO_FFN_V2_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

#define TILINGKEY_TPL_A3 1
#define TILINGKEY_TPL_A5 2
#define ATTN_FFN_TILINGKEY_NO_QUANT 0
#define ATTN_FFN_TILINGKEY_PERTOKEN_QUANT 2
#define ATTN_FFN_TILINGKEY_PERGROUP_QUANT 3
#define ATTN_FFN_TILINGKEY_MX_QUANT 4
#define ATTN_FFN_TILINGKEY_MX_QUANT_CLIP 5
#define ATTN_FFN_TILINGKEY_DTYPE_INT8 0
#define ATTN_FFN_TILINGKEY_DTYPE_FP8_E4M3FN 1
#define ATTN_FFN_TILINGKEY_DTYPE_FP8_E5M2 2
#define ATTN_FFN_TILINGKEY_DTYPE_FP4_E2M1 3
#define ATTN_FFN_TILINGKEY_DTYPE_FP4_E1M2 4

namespace Mc2Tiling {
ASCENDC_TPL_ARGS_DECL(AttentionToFfnV2,
                      ASCENDC_TPL_UINT_DECL(ATTN_FFN_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            ATTN_FFN_TILINGKEY_NO_QUANT, ATTN_FFN_TILINGKEY_PERTOKEN_QUANT,
                                            ATTN_FFN_TILINGKEY_PERGROUP_QUANT, ATTN_FFN_TILINGKEY_MX_QUANT,
                                            ATTN_FFN_TILINGKEY_MX_QUANT_CLIP),
                      ASCENDC_TPL_UINT_DECL(ATTN_FFN_QUANT_DTYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            ATTN_FFN_TILINGKEY_DTYPE_INT8,
                                            ATTN_FFN_TILINGKEY_DTYPE_FP8_E4M3FN,
                                            ATTN_FFN_TILINGKEY_DTYPE_FP8_E5M2,
                                            ATTN_FFN_TILINGKEY_DTYPE_FP4_E2M1,
                                            ATTN_FFN_TILINGKEY_DTYPE_FP4_E1M2),
                      ASCENDC_TPL_BOOL_DECL(ATTN_FFN_SCALE, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_SYNC, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_ACTIVE_MASK, 0, 1),
                      ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5), );
ASCENDC_TPL_SEL(
    // Legacy kernels remain registered here. FP8/FP4 selections are registered with their kernel implementation.
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ATTN_FFN_QUANT_MODE, ASCENDC_TPL_UI_LIST,
                                              ATTN_FFN_TILINGKEY_NO_QUANT),
                         ASCENDC_TPL_UINT_SEL(ATTN_FFN_QUANT_DTYPE, ASCENDC_TPL_UI_LIST,
                                              ATTN_FFN_TILINGKEY_DTYPE_INT8),
                         ASCENDC_TPL_BOOL_SEL(ATTN_FFN_SCALE, 0), ASCENDC_TPL_BOOL_SEL(TILINGKEY_SYNC, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_ACTIVE_MASK, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
                         ASCENDC_TPL_TILING_STRUCT_SEL(AttentionToFfnV2TilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ATTN_FFN_QUANT_MODE, ASCENDC_TPL_UI_LIST,
                                              ATTN_FFN_TILINGKEY_PERTOKEN_QUANT),
                         ASCENDC_TPL_UINT_SEL(ATTN_FFN_QUANT_DTYPE, ASCENDC_TPL_UI_LIST,
                                              ATTN_FFN_TILINGKEY_DTYPE_INT8),
                         ASCENDC_TPL_BOOL_SEL(ATTN_FFN_SCALE, 0, 1), ASCENDC_TPL_BOOL_SEL(TILINGKEY_SYNC, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_ACTIVE_MASK, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
                         ASCENDC_TPL_TILING_STRUCT_SEL(AttentionToFfnV2TilingData)));
} // namespace Mc2Tiling
#endif
