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
 * \file mixed_quant_flash_attn_kernel_base_gqa_template.h
 * \brief MixedQuantFlashAttn GQA Kernel入口class骨架定义
 */

#ifndef MIXED_QUANT_FLASH_ATTN_KERNEL_BASE_GQA_TEMPLATE_H_
#define MIXED_QUANT_FLASH_ATTN_KERNEL_BASE_GQA_TEMPLATE_H_

#include "flash_attn_block_cube_mixedquant_gqa.h"
#include "flash_attn_block_vec_mixedquant_gqa.h"
#include "flash_attention_mixedquant_block_vec_flashdecode.h"
#include "mixed_quant_flash_attn_tiling_data.h"
#include "../utils/mixed_quant_flash_attn_common_def.h"

namespace BaseApi {

template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionMixedQuantGqaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;

    __aicore__ inline FlashAttentionMixedQuantGqaKernel()
    {
    }

    __aicore__ inline void
    Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *kDescale,
         __gm__ uint8_t *vDescale, __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
         __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *blocktable, __gm__ uint8_t *learnableSink,
         __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *metaData,
         const __gm__ optiling::MixedQuantFlashAttnTilingArch35 *__restrict tilingData, TPipe *tPipe)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void Process()
    {
        // TODO: 待后续填充实现
    }
};

} // namespace BaseApi

#endif // MIXED_QUANT_FLASH_ATTN_KERNEL_BASE_GQA_TEMPLATE_H_
