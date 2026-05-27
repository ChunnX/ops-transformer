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
 * \file flash_attn_entry_regbase.h
 * \brief FlashAttn arch35 kernel类型推导与实例化（非量化场景）
 *
 * 4字段 tiling key → 模板参数: inOutLayoutType, KvLayoutType, hasAttenMask, config
 * 其余参数硬编码: pseMode=NONE, quantMode=NoQuant, hasRope=false, isFd=true, ...
 */

#ifndef FLASH_ATTN_ENTRY_REGBASE_H_
#define FLASH_ATTN_ENTRY_REGBASE_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../utils/flash_attn_utils.h"
#include "../utils/flash_attn_common_def.h"
#include "flash_attn_kernel_base_gqa_cube.h"
#include "flash_attn_kernel_base_gqa_vector.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "flash_attn_tiling_data.h"

// ─────────────────────────────────────────────────────────────────────────────
// flash_attn_kernel_run: 4字段 tiling key → layout / s1 / s2 / d / dv
// ─────────────────────────────────────────────────────────────────────────────
template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask,
          uint8_t config>
inline __aicore__ void
flash_attn_kernel_run(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask,
                      __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ,
                      __gm__ uint8_t *sequsedKv, __gm__ uint8_t *blocktable, __gm__ uint8_t *learnableSink,
                      __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                      __gm__ uint8_t *tiling, __gm__ uint8_t *runtimeMetaData)
{
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    constexpr bool hasRopeConst = false;
    constexpr bool emptyTensorConst = false;
    constexpr uint8_t pFAMaskConst = 0;
    constexpr uint8_t pFAMatMulTypeConst = 0;
    constexpr bool enableKVPrefixConst = false;
    constexpr bool enableS1OutSplitConst = false;

    constexpr TPosition bmm2OutPos =
        BaseApi::GetC2Position(dVTemplateType,
                               BaseApi::UbOutCondition<INPUT_T>(false, PseTypeEnum::PSE_NONE_TYPE, hasAttenMask, false,
                                                                false, (uint32_t)s1TemplateType == 64),
                               ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn =
        BaseApi::IsDn(false, false, PseTypeEnum::PSE_NONE_TYPE, hasAttenMask, false, (uint32_t)s1TemplateType == 64,
                      dTemplateType, hasRopeConst, enableKVPrefixConst, true, IsSameType<INPUT_T, hifloat8_t>::value);
    constexpr bool bmm2Write2Ub = (bmm2OutPos == TPosition::VECCALC);
    constexpr bool splitD = ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256);

    using CubeBlock =
        BaseApi::FANoQuantGqaBlockCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                                       dVTemplateType, KvLayoutType, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock =
        BaseApi::FANoQuantGqaBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                      s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, false, KvLayoutType,
                                      useDn, bmm2Write2Ub, splitD>;
    using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                       s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                       hasAttenMask, false, KvLayoutType, useDn, bmm2Write2Ub, splitD>;

#ifdef __DAV_C310_CUBE__
    using Processor = BaseApi::FlashAttentionCubeProcessor<CubeBlock>;
#else
    using Processor = BaseApi::FlashAttentionVectorProcessor<VecFaBlock, VecFdBlock, INPUT_T,
                                                              CubeBlock::LAYOUT, CubeBlock::PAGE_ATTENTION>;
#endif

    size_t offset = (size_t)(&((FlashAttnTilingData *)0)->baseTiling);
    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData =
        (const __gm__ FlashAttnNoQuantTilingArch35 *__restrict)(tiling + offset);

    TPipe tPipe;
    Processor op;
    op.Init(query, key, value, attenMask, cuSeqLensQ, cuSeqLensKv, blocktable, learnableSink, softmaxLse, attentionOut,
            workspace, runtimeMetaData, sequsedQ, sequsedKv, tilingData, &tPipe);
    op.Process();
}

#endif // FLASH_ATTN_ENTRY_REGBASE_H_