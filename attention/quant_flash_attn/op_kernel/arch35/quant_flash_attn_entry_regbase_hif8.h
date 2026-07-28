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
 * \file quant_flash_attn_entry_regbase_hif8.h
 * \brief QuantFlashAttn arch35 HIF8 per-tensor kernel type deduction and instantiation.
 *
 * quantMode: QFA_HIF8_FP32_PREFILL(prefill, isFd=false) / QFA_HIF8_FP32_DECODE(decode, isFd=true)
 */

#ifndef QUANT_FLASH_ATTN_ENTRY_REGBASE_HIF8_H_
#define QUANT_FLASH_ATTN_ENTRY_REGBASE_HIF8_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../utils/quant_flash_attn_utils.h"
#include "../utils/quant_flash_attn_common_def.h"
#include "quant_flash_attn_kernel_fullquant_hif8.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "quant_flash_attn_tiling_data.h"

template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask,
          uint8_t config, uint8_t quantMode>
inline __aicore__ void
quant_flash_attn_kernel_run_hif8(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                             __gm__ uint8_t *qDescale, __gm__ uint8_t *kDescale, __gm__ uint8_t *vDescale,
                             __gm__ uint8_t *blockTable, __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqLensQ,
                             __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
                             __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
                             __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                             __gm__ uint8_t *tiling)
{
    fa_base_matmul::idCounterNum = 0;

    constexpr LayOutTypeEnum inputLayoutType =
        static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType =
        static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    constexpr bool isFdConst = false;
    constexpr PseTypeEnum pseModeConst = PseTypeEnum::PSE_NONE_TYPE;
    constexpr bool enableKVPrefixConst = false;

    constexpr TPosition bmm2OutPos =
        BaseApi::GetC2Position(dVTemplateType,
                               BaseApi::UbOutCondition<INPUT_T>(false, pseModeConst, hasAttenMask, false, false,
                                                                (uint32_t)s1TemplateType == 64),
                               ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn = true;
    constexpr bool bmm2Write2Ub = (bmm2OutPos == TPosition::VECCALC);
    constexpr bool splitD = ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256);

    using CubeBlock =
        BaseApi::FAFullQuantHif8BlockCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                          dTemplateType, dVTemplateType, KvLayoutType,
                                          enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock =
        BaseApi::FAFullQuantHif8BlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                         s2TemplateType, dTemplateType, dVTemplateType, pseModeConst, hasAttenMask,
                                         false, KvLayoutType, isFdConst, enableKVPrefixConst, useDn,
                                         bmm2Write2Ub, splitD>;
    using VecFdBlock =
        BaseApi::FiaBlockVecFlashDecodeFullQuant<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                 s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                 pseModeConst, hasAttenMask, false, KvLayoutType,
                                                 enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;

    using CubeBlockDummy =
        BaseApi::FAFullQuantHif8BlockCubeDummy<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                               dTemplateType, dVTemplateType, KvLayoutType,
                                               enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlockDummy =
        BaseApi::FAFullQuantHif8BlockVecDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                             s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                             pseModeConst, hasAttenMask, false, KvLayoutType,
                                             isFdConst, enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;
    using VecFdBlockDummy =
        BaseApi::FiaBlockVecFlashDecodeFullQuantDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                      s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                      pseModeConst, hasAttenMask, false, KvLayoutType,
                                                      enableKVPrefixConst, useDn, bmm2Write2Ub, splitD>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::FlashAttentionFullQuantHif8Kernel<CubeBlock, VecFaBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::FlashAttentionFullQuantHif8Kernel<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    const __gm__ QuantFlashAttnTilingData *__restrict tilingData =
        (const __gm__ QuantFlashAttnTilingData *__restrict)tiling;

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, sinks, attnMask, cuSeqLensQ, cuSeqLensKv, blockTable, qDescale, kDescale, vDescale,
            pScale, softmaxLse, attnOut, workspace, metadata, sequsedQ, sequsedKv, tilingData,
            &tPipe);
    op.Process();
}

#endif // QUANT_FLASH_ATTN_ENTRY_REGBASE_HIF8_H_
