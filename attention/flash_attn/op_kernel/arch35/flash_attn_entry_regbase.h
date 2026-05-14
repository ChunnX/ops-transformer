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
 * \brief FlashAttn arch35 kernel入口（非量化场景，框架桩）
 *
 * 参照flash_attn_score/op_kernel/arch35/flash_attn_score_entry_regbase.h框架，
 * 去除PSE/dropout/rope相关参数，新增PA layout和metadata支持。
 * 具体的kernel class调用待实现。
 */

#ifndef FLASH_ATTN_ENTRY_REGBASE_H_
#define FLASH_ATTN_ENTRY_REGBASE_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "fa_template_dispatcher.h"


// FlashAttn kernel核心函数（arch35）

template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope, 
    uint8_t KvLayoutType, bool isFd, bool emptyTensor, uint8_t PFAMask, uint8_t pFAMatMulType, bool enableKVPrefix, bool enableS1OutSplit>
inline __aicore__ void flash_attn_regbase(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *attnMask,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
__gm__ uint8_t* user = GetUserWorkspace(workspace);

// 从 Config 解析 s1/s2/d/dv
constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
constexpr DTemplateType  dTemplateType  = static_cast<DTemplateType>(ConfigValue[config].d);
constexpr DTemplateType  dvTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);


// 从 InOutLayoutType 解析 layout（取 input layout）
constexpr LayOutTypeEnum layout = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
// implMode 和 regbase 写死默认値
constexpr uint8_t implMode = 0;  // AA_HIGH_PRECISION
(void)implMode;

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

if (actualSeqLengthsQ != nullptr) {
    auto *sq64 = (__gm__ uint64_t *)actualSeqLengthsQ;
    if (sq64[0] == 0) {
        actualSeqLengthsQ += sizeof(uint64_t);
    }
}
if (actualSeqLengthsKv != nullptr) {
    auto *sk64 = (__gm__ uint64_t *)actualSeqLengthsKv;
    if (sk64[0] == 0) {
        actualSeqLengthsKv += sizeof(uint64_t);
    }
}
#if (ORIG_DTYPE_Q == DT_BF16)
run_fia_noquant_gqa_kernel<bfloat16_t, bfloat16_t, inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, KvLayoutType,
                            isFd, emptyTensor, PFAMask, pFAMatMulType, enableKVPrefix, enableS1OutSplit>(
    query, key, value, nullptr, attnMask, actualSeqLengthsQ, actualSeqLengthsKv, nullptr,
    nullptr, blockTable, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, sinks, attentionOut, softmaxLse, user, tiling, metadata);
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
run_fia_noquant_gqa_kernel<half, half, inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, KvLayoutType,
                            isFd, emptyTensor, PFAMask, pFAMatMulType, enableKVPrefix, enableS1OutSplit>(
    query, key, value, nullptr, attnMask, actualSeqLengthsQ, actualSeqLengthsKv, nullptr,
    nullptr, blockTable, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, sinks, attentionOut, softmaxLse, user, tiling, metadata);
#endif

}

#endif // FLASH_ATTN_ENTRY_REGBASE_H_