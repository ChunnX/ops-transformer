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
 * \file mixed_quant_flash_attn_entry_regbase.h
 * \brief MixedQuantFlashAttn arch35 kernel类型推导与实例化（伪量化场景）
 *
 * 5字段 tiling key → 模板参数: inOutLayoutType, KvLayoutType, hasAttenMask, config, QuantMode
 */

#ifndef MIXED_QUANT_FLASH_ATTN_ENTRY_REGBASE_H_
#define MIXED_QUANT_FLASH_ATTN_ENTRY_REGBASE_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "mixed_quant_flash_attn_kernel_base_gqa_template.h"
#include "mixed_quant_flash_attn_tiling_data.h"

template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask,
          uint8_t config, uint8_t QuantMode>
inline __aicore__ void
mixed_quant_flash_attn_kernel_run(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                  __gm__ uint8_t *kDescale, __gm__ uint8_t *vDescale, __gm__ uint8_t *attenMask,
                                  __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
                                  __gm__ uint8_t *sequsedKv, __gm__ uint8_t *blocktable, __gm__ uint8_t *learnableSink,
                                  __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                  __gm__ uint8_t *tiling, __gm__ uint8_t *metaData)
{
    constexpr inferFaLayOutTypeEnum inputLayoutType =
        static_cast<inferFaLayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr inferFaLayOutTypeEnum outputLayoutType =
        static_cast<inferFaLayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr inferS1TemplateType s1TemplateType = static_cast<inferS1TemplateType>(MixedQuantConfigValue[config].s1);
    constexpr inferS2TemplateType s2TemplateType = static_cast<inferS2TemplateType>(MixedQuantConfigValue[config].s2);
    constexpr inferDTemplateType dTemplateType = static_cast<inferDTemplateType>(MixedQuantConfigValue[config].d);

    using CubeBlock = BaseApi::FAMixedQuantGqaBlockCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                                        dTemplateType, KvLayoutType>;
    using VecFaBlock =
        BaseApi::FAMixedQuantGqaBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                         s2TemplateType, dTemplateType, hasAttenMask, KvLayoutType>;
    using VecFdBlock = BaseApi::FAMixedQuantGqaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType,
                                                                   outputLayoutType, s1TemplateType, s2TemplateType,
                                                                   dTemplateType, hasAttenMask, KvLayoutType>;
    using VecBlockDummy =
        BaseApi::MixedQuantGqaVecBlockBase<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                           s2TemplateType, dTemplateType, hasAttenMask, KvLayoutType>;
    using CubeBlockDummy = BaseApi::MixedQuantGqaCubeBlockBase<INPUT_T, float, inputLayoutType, s1TemplateType,
                                                               s2TemplateType, dTemplateType, KvLayoutType>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::FlashAttentionMixedQuantGqaKernel<CubeBlock, VecBlockDummy, VecBlockDummy>;
#else
    using Kernel = BaseApi::FlashAttentionMixedQuantGqaKernel<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    size_t offset = (size_t)(&((optiling::MixedQuantFlashAttnTilingData *)0)->baseTiling);
    const __gm__ optiling::MixedQuantFlashAttnTilingArch35 *__restrict tilingData =
        (const __gm__ optiling::MixedQuantFlashAttnTilingArch35 *__restrict)(tiling + offset);

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, kDescale, vDescale, attenMask, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, blocktable,
            learnableSink, softmaxLse, attentionOut, workspace, metaData, tilingData, &tPipe);
    op.Process();
}

#endif // MIXED_QUANT_FLASH_ATTN_ENTRY_REGBASE_H_
