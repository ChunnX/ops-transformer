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
 * 调用链:  flash_attn  →  flash_attn_kernel_run  →  Kernel::Init / Process
 *         (flash_attn.cpp)  (本文件，类型推导+实例化)     (算子逻辑)
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
#include "flash_attn_kernel_base_gqa_template.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "flash_attn_tiling_data.h"

// ─────────────────────────────────────────────────────────────────────────────
// flash_attn_kernel_run: 解析Config，构造Block/Kernel类型，AIC/AIV分别编译实例化
// ─────────────────────────────────────────────────────────────────────────────
template <typename INPUT_T, typename OUT_T,
          uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode,
          bool hasAttenMask, bool hasRope, uint8_t KvLayoutType, bool isFd,
          bool emptyTensor, uint8_t pFAMask, uint8_t pFAMatMulType,
          bool enableKVPrefix, bool enableS1OutSplit>
inline __aicore__ void flash_attn_kernel_run(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling, __gm__ uint8_t *runtimeMetaData)
{
    // 1. 解析 Config → layout / s1 / s2 / d / dv
    constexpr LayOutTypeEnum inputLayoutType  = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);
    constexpr S1TemplateType s1TemplateType   = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType   = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType  dTemplateType    = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType  dVTemplateType   = static_cast<DTemplateType>(ConfigValue[config].dv);

    // 2. 推导编译期常量
    constexpr TPosition bmm2OutPos =
        BaseApi::GetC2Position(dVTemplateType,
                      BaseApi::UbOutCondition<INPUT_T>(false, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false,
                                              hasRope, (uint32_t)s1TemplateType == 64),
                      ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn =
        BaseApi::IsDn(false, false, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false,
             (uint32_t)s1TemplateType == 64, dTemplateType, hasRope, enableKVPrefix, true,
             IsSameType<INPUT_T, hifloat8_t>::value);
    constexpr bool bmm2Write2Ub = (bmm2OutPos == TPosition::VECCALC);
    constexpr bool splitD = ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256);

    // 3. Block 类型别名
    using CubeBlock =
        BaseApi::FANoQuantGqaBlockCube<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                       dTemplateType, dVTemplateType, hasRope, KvLayoutType,
                                       enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock =
        BaseApi::FANoQuantGqaBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                      s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                      static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope,
                                      KvLayoutType, isFd, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFdBlock =
        BaseApi::FiaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                        static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope,
                                        KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;

    // 4. AIC/AIV分别编译：用Dummy Block避免交叉编译不需要的代码
    using VecBlockDummy =
        BaseApi::VecBlockBase<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                              s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                              static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope,
                              KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using CubeBlockDummy =
        BaseApi::CubeBlockBase<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                               dTemplateType, dVTemplateType, hasRope, KvLayoutType,
                               enableKVPrefix, useDn, bmm2Write2Ub, splitD>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::FlashAttentionNoQuantGqaKernel<CubeBlock, VecBlockDummy, VecBlockDummy>;
#else
    using Kernel = BaseApi::FlashAttentionNoQuantGqaKernel<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    // 5. Tiling 解析、实例化并执行
    // GET_TILING_DATA_MEMBER(FlashAttnTilingData, baseTiling, baseTilingIn, tiling);
    // const FlashAttnNoQuantTilingArch35 *__restrict tilingData = &baseTilingIn;
    size_t offset = (size_t)(&((FlashAttnTilingData*)0)->baseTiling);
    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData = (const __gm__ FlashAttnNoQuantTilingArch35 *__restrict)(tiling + offset);

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, nullptr, attenMask, cuSeqLensQ, cuSeqLensKv,
            blocktable, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, learnableSink, softmaxLse, attentionOut,
            workspace, runtimeMetaData, sequsedQ, sequsedKv, tilingData, &tPipe);
    op.Process();
}

#endif // FLASH_ATTN_ENTRY_REGBASE_H_