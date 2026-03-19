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

#include "../../../common/op_kernel/arch35/flash_attention_score_tiling_regbase.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_kernel_infer.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_kernel_base.h"

#define FA_COPY_TILING_DATA(tiling)                                                                    \
    GET_TILING_DATA_WITH_STRUCT(FlashAttnScoreSimplifiedTilingData, tilingDataIn, tiling);        \
    const FlashAttnScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn;              \

#ifdef __DAV_C310_CUBE__
//todo kernel tempale 实例化
#define INVOKE_FA_IMPL(templateClass, ...)                                                             \
    do {                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                           \
        TPipe tPipe;                                                                                   \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,                   \
            BaseApi::FABlockCube<__VA_ARGS__>, BaseApi::FABlockCubeDummy<__VA_ARGS__>>::type;          \
        using VecBlockType  = typename std::conditional<g_coreType == AscendC::AIC,                   \
            BaseApi::FABlockVecDummy<__VA_ARGS__>, BaseApi::FABlockVecTrain<__VA_ARGS__>>::type;       \
    } while (0)

#else // VECTOR 实现

#ifndef __CCE_KT_TEST__
#define INVOKE_FA_IMPL(templateClass, ...)                                                             \
    do {                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                           \
        FA_COPY_TILING_DATA(tiling);                                                                   \
        TPipe tPipe;                                                                                   \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,                   \
            BaseApi::FABlockCube<__VA_ARGS__>, BaseApi::FABlockCubeDummy<__VA_ARGS__>>::type;          \
        using VecBlockType  = typename std::conditional<g_coreType == AscendC::AIC,                   \
            BaseApi::FABlockVecDummy<__VA_ARGS__>, BaseApi::FABlockVecTrain<__VA_ARGS__>>::type;       \
    } while (0)
#else  // test模式
#define INVOKE_FA_IMPL(templateClass, ...)                                                             \
    do {                                                                                               \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                           \
        FA_COPY_TILING_DATA(tiling);                                                                   \
        TPipe tPipe;                                                                                   \
        using CubeBlockType = typename BaseApi::FABlockCube<__VA_ARGS__>;                             \
        using VecBlockType  = typename BaseApi::FABlockVecTrain<__VA_ARGS__>;                         \
    } while (0)
#endif // __CCE_KT_TEST__

#endif // __DAV_C310_CUBE__

// FlashAttn kernel核心函数（arch35）
template<uint8_t implMode, uint8_t layout, uint16_t s1TemplateType, uint16_t s2TemplateType,
    uint16_t dTemplateType, uint16_t dvTemplateType, bool hasAtten, bool isPA, bool isSoftmaxLse,
    uint8_t regbase>
inline __aicore__ void flash_attn_regbase(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
    __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // TODO: 根据isPA选择推理kernel（PA场景）或训练kernel（非PA场景）
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (!isPA) {
        if (isSoftmaxLse) {
            // 训练场景（returnSoftmaxLse=1）
            INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelTrain,
                half, float, half, ImplModeEnum(implMode), LayOutTypeEnum(layout),
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
                PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
                hasAtten, false, false);
        } else {
            // 推理场景（returnSoftmaxLse=0）
            INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelInfer,
                half, float, half, ImplModeEnum(implMode), LayOutTypeEnum(layout),
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
                PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
                hasAtten, false, false);
        }
    }
    // TODO: PA场景kernel分发
    return;
    #endif

    #if (ORIG_DTYPE_QUERY == DT_BF16)
    if (!isPA) {
        if (isSoftmaxLse) {
            INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelTrain,
                bfloat16_t, float, bfloat16_t, ImplModeEnum(implMode), LayOutTypeEnum(layout),
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
                PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
                hasAtten, false, false);
        } else {
            INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelInfer,
                bfloat16_t, float, bfloat16_t, ImplModeEnum(implMode), LayOutTypeEnum(layout),
                S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
                DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
                PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
                hasAtten, false, false);
        }
    }
    return;
    #endif

#endif // __CCE_AICORE__ == 310
}

#endif // FLASH_ATTN_ENTRY_REGBASE_H_