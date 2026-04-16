/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prompt_flash_attention_310p.h
 * \brief
 */
#include "kernel_operator.h"
#include "./arch32/prompt_flash_attention_tilingkey.h"
#include "./arch32/unpad_flash_attention_common.h"
#include "./arch32/prompt_attention_prefill.h"
#include "./arch32/prompt_flash_attention_s1s2_bns1_x310_base.h"
#include "./arch32/prompt_flash_attention_s1s2_bns1_x310.h"

#define INVOKE_PFA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
            kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitMsd(key_antiquant_scale, key_antiquant_offset,value_antiquant_scale, value_antiquant_offset);                     \
        op.Process();                                                                                                   \
    } while (0)

#ifdef __DAV_C220_CUBE__
#define INVOKE_PFA_TILING_DATA(tiling)                                                                                 \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm1TilingDataRect, bmm1TilingData, tiling);                \
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm2TilingDataRect, bmm2TilingData, tiling);                \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingData;                                                        \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingData;                                                        \
    const PromptFlashAttentionTilingData* __restrict tiling_data = nullptr

#define INVOKE_PFA_TILING_DATA_BASE_API(tiling)                                                                        \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionBaseApiTilingData, tiling_data_in, tiling);                        \
    const PromptFlashAttentionBaseApiTilingData* __restrict tiling_data = &tiling_data_in
#else

#define INVOKE_PFA_TILING_DATA(tiling)                                                                                 \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionTilingData, tiling_data_in, tiling);                               \
    const PromptFlashAttentionTilingData* __restrict tiling_data = &tiling_data_in;                                    \
    const TCubeTiling* __restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);                                     \
    const TCubeTiling* __restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect)

#define INVOKE_PFA_TILING_DATA_BASE_API(tiling)                                                                        \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionBaseApiTilingData, tiling_data_in, tiling);                        \
    const PromptFlashAttentionBaseApiTilingData* __restrict tiling_data = &tiling_data_in
#endif

#define INVOKE_PFA_NEW_GQA_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        if (query == nullptr) {return;}                                                                              \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                                \
        templateClass<__VA_ARGS__> op;                                                                          \
        op.Init(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, attentionOut, user, tiling_data);                                 \
        op.Process();                                                                                                  \
    } while (0)

inline __aicore__ void prompt_flash_attention_FIAS_310P(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                        __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                        __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                        __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                        __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquant_scale, __gm__ uint8_t* antiquant_offset,
                                                        __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                        __gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset,
                                                        __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset,
                                                        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                        __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* learnableSink,
                                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                        __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
    auto maskByteNum = baseParams.maskTypeByteNum;

    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TILING_KEY_IS(QINT8_KVFP16_OUTBF16_BSH_HIGHLEVELAPI_MDL_310TILING);
    TILING_KEY_IS(QINT8_KVFP16_OUTINT8_BSH_HIGHLEVELAPI_MDL_310TILING);
    TILING_KEY_IS(QINT8_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    TILING_KEY_IS(QINT8_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    TILING_KEY_IS(QFP4E1M2_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    TILING_KEY_IS(QFP4E1M2_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING);
    #if TILING_KEY_VAR == QINT8_KVFP16_OUTBF16_BSH_HIGHLEVELAPI_MDL_310TILING
        INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
    #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_BSH_HIGHLEVELAPI_MDL_310TILING
        INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
    #elif TILING_KEY_VAR == QINT8_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half>);
    #elif TILING_KEY_VAR == QINT8_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half>);
    #elif TILING_KEY_VAR == QFP4E1M2_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
    #elif TILING_KEY_VAR == QFP4E1M2_KVFP16_OUTINT8_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
    #endif
}