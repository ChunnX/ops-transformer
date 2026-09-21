/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_FP8_TILING_KEY_H
#define QBSA_FP8_TILING_KEY_H

#include <cstdint>

#include "../common/quant_block_sparse_attn_tiling_key.h"

#define QBSA_FP8_CONFIG QBSA_TILING_CONFIG_0
#define QBSA_FP8_QUANT_MODE QBSA_TILING_QUANT_MODE_1

#define QBSA_FP8_TPL_SELECTION \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_UINT_SEL(QKV_DTYPE, ASCENDC_TPL_UI_LIST, QBSA_DTYPE_FP8_E4M3FN), \
        ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QBSA_LAYOUT_TND, QBSA_LAYOUT_NTD), \
        ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QBSA_KV_LAYOUT_CONTIGUOUS, QBSA_KV_LAYOUT_PA_BNSD), \
        ASCENDC_TPL_UINT_SEL(MASK_MODE, ASCENDC_TPL_UI_LIST, QBSA_MASK_NONE, QBSA_MASK_CAUSAL), \
        ASCENDC_TPL_BOOL_SEL(RETURN_SOFTMAX_LSE, 0, 1), \
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, QBSA_FP8_CONFIG), \
        ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, QBSA_FP8_QUANT_MODE))

namespace optiling {
inline uint64_t BuildQbsaFp8TilingKey(uint32_t layoutQ, uint32_t maskMode, bool returnSoftmaxLse)
{
    return GET_TPL_TILING_KEY(QBSA_DTYPE_FP8_E4M3FN, layoutQ, QBSA_KV_LAYOUT_PA_BNSD, maskMode,
                              returnSoftmaxLse ? 1U : 0U, QBSA_FP8_CONFIG, QBSA_FP8_QUANT_MODE);
}
} // namespace optiling

#endif // QBSA_FP8_TILING_KEY_H
