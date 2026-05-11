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
 * \file ifa_public_define.h
 * \brief
 */
 
 #ifndef FIA_TEMPLATE_DISPATCHER_H
 #define FIA_TEMPLATE_DISPATCHER_H

#include "../utils/flash_attn_utils.h"
#include "../utils/flash_attn_common_def.h"
#include "flash_attn_kernel_base_gqa_template.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "flash_attn_tiling_data.h"


#define PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, ...) \
    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]); \
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]); \
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1); \
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2); \
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d); \
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv)


template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode,
          uint8_t quantMode, bool hasAttenMask, bool hasRope, uint8_t KvLayoutType, bool isFd, bool emptyTensor, uint8_t pFAMask,
          uint8_t pFAMatMulType, bool enableKVPrefix, bool enableS1OutSplit>
inline __aicore__ void run_fia_noquant_gqa_kernel(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling, __gm__ uint8_t *runtimeMetaData)
{
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, ...);

    // constexpr bool isS2Base64 = (uint32_t)s1TemplateType == 64;
    constexpr TPosition bmm2OutPos =
        BaseApi::GetC2Position(dVTemplateType,
                      BaseApi::UbOutCondition<INPUT_T>(false, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope,
                                              (uint32_t)s1TemplateType == 64),
                      ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn =
        BaseApi::IsDn(false, false, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, (uint32_t)s1TemplateType == 64,
             dTemplateType, hasRope, enableKVPrefix, true, IsSameType<INPUT_T, hifloat8_t>::value);
    // constexpr bool useDn = false;
    constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;

    using CubBlock =
        BaseApi::FANoQuantGqaBlockCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                                       dVTemplateType, hasRope, KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock =
        BaseApi::FANoQuantGqaBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                      s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                      static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, KvLayoutType, isFd,
                                      enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFdBlock =
        BaseApi::FiaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                     s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                      static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, KvLayoutType,
                                      enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using Kernel = BaseApi::FlashAttentionNoQuantGqaKernel<CubBlock, VecFaBlock, VecFdBlock>;
    GET_TILING_DATA_MEMBER(FlashAttnTilingData, baseTiling, baseTilingIn, tiling);
    const FlashAttnNoQuantTilingArch35 *__restrict tilingData = &baseTilingIn;
    __gm__ uint8_t *fiaMetaData = runtimeMetaData;
    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,
            kvPaddingSize, postQuantScale, postQuantOffset, keySharedPrefix, valueSharedPrefix,
            actualSharedPrefixLen, queryRope, keyRope, learnableSink, softmaxLse, attentionOut, workspace, fiaMetaData,
            tilingData, &tPipe);
    op.Process();
}


#endif