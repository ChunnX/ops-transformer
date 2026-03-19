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
 * \file flash_attn.cpp
 * \brief FlashAttn Kernel主入口（框架，参照flash_attn_score结构）
 */

#include "kernel_operator.h"
#include "arch35/flash_attn_entry_regbase.h"
#include "arch35/flash_attn_template_tiling_key.h"

using namespace AscendC;

// kernel主入口：
template<uint8_t KernelTypeKey, uint8_t implMode, uint8_t layout, uint16_t s1TemplateType, uint16_t s2TemplateType,
    uint16_t dTemplateType, uint16_t dvTemplateType, bool hasAtten, bool isPA, bool isSoftmaxLse,
    uint8_t regbase>
__global__ __aicore__ void flash_attn(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sinks,
    __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttentionScoreSimplifiedTilingData);
    // 框架根据tilingKey模板参数分发至flash_attn_regbase<...>()
    flash_attn_regbase<
        implMode, layout, s1TemplateType, s2TemplateType, dTemplateType, 
        dvTemplateType, hasAtten, isPA, isSoftmaxLse, regbase>(query, key, 
        value, blockTable,actualSeqLengthsQ, actualSeqLengthsKv, sequsedQ, 
        sequsedKv, sinks, metadata, attentionOut, softmaxLse, workspace, tiling);
}