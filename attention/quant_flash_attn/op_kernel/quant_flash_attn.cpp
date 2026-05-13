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
 * \file quant_flash_attn.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_kernel_dn.h"

using namespace optiling;
using namespace AscendC;

template <bool PAGE_ATTENTION, bool HAS_MASK, uint8_t Q_LAYOUT_T, uint8_t KV_LAYOUT_T>
__global__ __aicore__ void quant_flash_attn(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *q_descale,
    __gm__ uint8_t *k_descale,
    __gm__ uint8_t *v_descale,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *cuSeqlensKv,
    __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *attnMask,
    __gm__ uint8_t *metadata,
    __gm__ uint8_t *attnOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    // constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    // constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);
    // constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    // constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    // constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    // constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);
    // constexpr bool useDn = false;
    // constexpr bool bmm2Write2Ub = false;
    // constexpr bool splitD = false;

    // using INPUT_T = half;
    // using OUT_T = half;
    // using CubBlock =
    //     QFA_KERNEL::FANoQuantGqaBlockCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
    //                                    dVTemplateType, hasRope, KvLayoutType, useDn, bmm2Write2Ub, splitD>;
    // using VecFaBlock =
    //     QFA_KERNEL::FANoQuantGqaBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
    //                                   s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
    //                                   static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, KvLayoutType, isFd,
    //                                   useDn, bmm2Write2Ub, splitD>;
    // using VecFdBlock =
    //     QFA_KERNEL::FiaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
    //                                  s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
    //                                   static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, KvLayoutType,
    //                                   useDn, bmm2Write2Ub, splitD>;

    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;

    #if (ORIG_DTYPE_Q == DT_FLOAT4_E2M1) && (ORIG_DTYPE_Q_DESCALE == DT_FLOAT8_E8M0) && (ORIG_DTYPE_ATTN_OUT == DT_BF16)
        GET_TILING_DATA_MEMBER(QuantFlashAttnTilingData, baseTiling, baseTilingIn, tiling);
        const FlashAttnTilingData *__restrict tilingData = &baseTilingIn;
        QFA_KERNEL::QuantFlashAttnKernelDn<QFAType<fp4x2_e2m1_t, fp8_e8m0_t, bfloat16_t, PAGE_ATTENTION,
                                                (QFA_LAYOUT)Q_LAYOUT_T, (QFA_LAYOUT)KV_LAYOUT_T, HAS_MASK>> op;
        op.Init(query, key, value, q_descale, k_descale, v_descale, blockTable, cuSeqlensQ, cuSeqlensKv,
                sequsedQ, sequsedKv, attnMask, learnableSink, softmaxLse, attnOut,
                workspace, metadata, tilingData, &tPipe);
        op.Process();
    #endif
}