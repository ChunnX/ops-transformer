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
 * \file prompt_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#if (__CCE_AICORE__ > 200)
#include "prompt_flash_attention_base.h"
#include "prompt_flash_attention_bnstilling_n_s_no_tail.h"
#include "prompt_flash_attention_bnstilling_n_s_tail.h"
#include "prompt_flash_attention_bnstilling_n_s_no_tailWBNSD.h"
#include "prompt_flash_attention_bnstilling_n_s_tailWBNSD.h"
#include "prompt_flash_attention_s1s2_bns1_x910.h"
#include "prompt_flash_attention_base_api.h"
#include "prompt_flash_attention_base_api_high_precision_no_mask.h"
#include "prompt_flash_attention_base_mla.h"
#include "prompt_flash_attention_base_mla_high_precision.h"
#include "prompt_flash_attention_s1s2_bns1_mla.h"
#include "prompt_flash_attention_var_len_score_sab.h"
#include "prompt_flash_attention_s1s2_bns1_mla_baseapi.h"
#include "prompt_flash_attention_var_len_score_sab_baseapi.h"
#include "prompt_flash_attention_empty_tensor.h"
#else
#include "unpad_flash_attention_common.h"
#include "prompt_attention_prefill.h"
#include "prompt_flash_attention_s1s2_bns1_x310_base.h"
#include "prompt_flash_attention_s1s2_bns1_x310.h"
#endif

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
#define INVOKE_PFA_INT8_OP_IMPL(templateClass, ...)                                                                     \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.InitMsd(key_antiquant_scale, key_antiquant_offset,value_antiquant_scale, value_antiquant_offset);            \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_KVANTIQUANT_OP_IMPL(templateClass, ...)                                                              \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA(tiling);                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitKvAntiquant(antiquant_scale, antiquant_offset, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, value_antiquant_offset);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_WITH_MMPOLICY(templateClass, ...)                                                    \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        matmul::GlobalL1Array l1Array[matmul::L1_BUF_NUM];                                                              \
        matmul::l1Global = l1Array;                                                                                     \
        matmul::InitL1Buffer(&tPipe, matmul::l1Global);                                                                 \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);                      \
        op.Init(query, key, value, attenMask, attentionOut, softmaxLse, user, tilingData, &tPipe);                                  \
        op.Process();                                                                                                   \
    } while (0)
// PFA TND
#define INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm1Nz, bmm1tiling, op.bmm2, bmm2tiling); \
        op.UnpackInit(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, queryRope, keyRope, attentionOut, softmaxLse, user, tilingData, &tPipe);   \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(templateClass, ...)                                                          \
    TPipe tPipe;                                                                                                        \
    do {                                                                                                                \
        INVOKE_PFA_NO_KFC_TILING_DATA_MLA(tiling);                                                                             \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.UnpackInit(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV, queryRope, keyRope, blocktable, learnableSink, \
                     attentionOut, softmaxLse, user, tilingData, &tPipe);                                               \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(templateClass, ...)                                                              \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                        \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, nullptr, \
                deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, nullptr);                                    \
    } while (0)
#define INVOKE_PFA_GENERAL_OP_IMPL_MLA(templateClass, ...)                                                              \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        INVOKE_PFA_TILING_DATA_BASE_API(tiling);                                                                        \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, nullptr, nullptr); \
    } while (0)
#define INVOKE_PFA_NO_KFC_TILING_DATA_MLA(tiling)                                                                      \
    GET_TILING_DATA_WITH_STRUCT(MLAGeneralTilingData, tiling_data_in, tiling);                                         \
    const MLAGeneralTilingData *__restrict tilingData = &tiling_data_in;                                               \
    const TCubeTiling *__restrict bmm1tiling = nullptr;                                                                \
    const TCubeTiling *__restrict bmm2tiling = nullptr
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

#define INVOKE_PFA_TILING_DATA_MLA(tiling)                                                                             \
    GET_TILING_DATA_MEMBER(MLAGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);                           \
    GET_TILING_DATA_MEMBER(MLAGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);                           \
    const MLAGeneralTilingData * __restrict tilingData = nullptr;                                                      \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingDataVar;                                                     \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingDataVar 
#else
#define INVOKE_PFA_TILING_DATA(tiling)                                                                                 \
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionTilingData, tiling_data_in, tiling);                               \
    const PromptFlashAttentionTilingData* __restrict tiling_data = &tiling_data_in;                                    \
    const TCubeTiling* __restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);                                     \
    const TCubeTiling* __restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect)

#define INVOKE_PFA_TILING_DATA_MLA(tiling)                                                                              \
    GET_TILING_DATA_WITH_STRUCT(MLAGeneralTilingData, tiling_data_in, tiling);                                          \
    const MLAGeneralTilingData * __restrict tilingData = &tiling_data_in;                                               \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->bmm1TilingData);                                           \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->bmm2TilingData)

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


constexpr uint32_t FLOATBYTENUM = 8;
constexpr uint32_t FLOAT16BYTENUM = 16;
constexpr uint32_t INT8BYTENUM = 32;

extern "C" __global__ __aicore__ void prompt_flash_attention_FIAS(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquant_scale, __gm__ uint8_t* antiquant_offset,
                                                             __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                             __gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset,
                                                             __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset,
                                                             __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                             __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope, __gm__ uint8_t* learnableSink,
                                                             __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                             __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    {
    #if (__CCE_AICORE__ > 200)
        GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
        auto maskByteNum = baseParams.maskTypeByteNum;
    
        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY != DT_INT4)
            TILING_KEY_IS(1000000000000000010);
            TILING_KEY_IS(1000000000000000011);
            TILING_KEY_IS(1000000000000000015);
            TILING_KEY_IS(1000000000000000016);
            TILING_KEY_IS(1000000000000101012);
            TILING_KEY_IS(1000000000000001012);
            TILING_KEY_IS(2000000002004000012);
            TILING_KEY_IS(2000000000004001012);
            #if TILING_KEY_VAR == 1000000000000000010
                // Non-BNSD layout, split NS no tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, half, half, CubeFormat::ND, half);
                }
                else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, half, bool, CubeFormat::ND, half);
                }
            #elif TILING_KEY_VAR == 1000000000000000011
                // Non-BNSD layout, split NS with tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, half, half, CubeFormat::ND, half);
                }
                else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, half, bool, CubeFormat::ND, half);
                }
            #elif TILING_KEY_VAR == 1000000000000000015
                // BNSD layout, split NS no tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, half, half, CubeFormat::ND,
                                            half);
                }
                else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, half, bool, CubeFormat::ND,
                                            half);
                }
            #elif TILING_KEY_VAR == 1000000000000000016
                // BNSD layout, split NS with tail
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, half, half, CubeFormat::ND, half);
                }
                else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, half, bool, CubeFormat::ND, half);
                }
            #elif TILING_KEY_VAR == 1000000000000101012
                // no anti-quant path for CVDIFF-BSH, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, half>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool>);
                }
            #elif TILING_KEY_VAR == 1000000000000001012
                // no anti-quant path for CVDIFF-BNSD, half in half out
                if (maskByteNum == FLOAT16BYTENUM) {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, half>);
                } else {
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t>);
                }
            #elif TILING_KEY_VAR == 2000000002004000012
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionNoMask, PFAHighPrecisionBaseType<PromptFlashAttentionBaseApiTilingData, float, half, half, half, half, float>);
            #elif TILING_KEY_VAR == 2000000000004001012
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPerformance, PFATypeNew<PromptFlashAttentionBaseApiTilingData, half, half, half, half, half, float, half>);
            #elif TILING_KEY_VAR == 2000000010004001012
                INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLA, PFAMLAType<PromptFlashAttentionBaseApiTilingData>);
            #endif
    
            #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY != DT_INT4) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
                TILING_KEY_IS(1000000000000101612);
                TILING_KEY_IS(1000000000010101612);
                TILING_KEY_IS(1000000000100101612);
                TILING_KEY_IS(1000000000002101612);
                TILING_KEY_IS(1000000000000001612);
                TILING_KEY_IS(1000000000010001612);
                TILING_KEY_IS(1000000000100001612);
                TILING_KEY_IS(1000000800000101612);
                TILING_KEY_IS(1000000800100101612);
                TILING_KEY_IS(1000000800000001612);
                TILING_KEY_IS(1000000800100001612);
                TILING_KEY_IS(1000000000001001612);
                TILING_KEY_IS(1000000000101001612);
                TILING_KEY_IS(1000000000002001612);
                TILING_KEY_IS(1000000000010101012);
                TILING_KEY_IS(1000000000002101012);
                TILING_KEY_IS(1000000000100101012);
                TILING_KEY_IS(1000000800000101012);
                TILING_KEY_IS(1000000800100101012);
                TILING_KEY_IS(1000000000100001012);
                TILING_KEY_IS(1000000000001001012);
                TILING_KEY_IS(1000000000010001012);
                TILING_KEY_IS(1000000000101001012);
                TILING_KEY_IS(1000000000002001012);
                TILING_KEY_IS(1000000800000001012);
                TILING_KEY_IS(1000000800100001012);
                TILING_KEY_IS(1000000400300101612);
                TILING_KEY_IS(1000000400300001612);
                TILING_KEY_IS(1000000400200101612);
                TILING_KEY_IS(1000000400200001612);
                TILING_KEY_IS(2000000000004000012);
                TILING_KEY_IS(2000000010004000012);
                TILING_KEY_IS(1000000800010101012);
                TILING_KEY_IS(1000000800010001012);
                #if TILING_KEY_VAR == 1000000000000101612
                    // BSH layout HighPrecision
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000000010101612
                    // BSH layout HighPrecision, enable PA
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100101612
                    // Prefix BSH layout HighPrecision
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000002101612
                    // BSH layout HighPrecision, enable L1 reuse
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_IBSHARE_NORM>);
                #elif TILING_KEY_VAR == 1000000000000001612
                    // BNSD layout HighPrecision
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000000010001612
                    // BNSD layout HighPrecision, enable PA
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100001612
                    // Prefix BNSD layout HighPrecision
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000800000101612
                    // BSH layout HighPrecision
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000800100101612
                    // Prefix BSH layout HighPrecision
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000800000001612
                    // BNSD layout HighPrecision
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000800100001612
                    // Prefix BNSD layout HighPrecision
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000001001612
                    // BNSD layout HighPrecision
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_NORM>);
                #elif TILING_KEY_VAR == 1000000000101001612
                    // Prefix BNSD layout HighPrecision
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_NORM, true>);
                #elif TILING_KEY_VAR == 1000000000002001612
                    // BNSD layout HighPrecision
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, OptimizationMode::HighPrecision, MatMulType::MM_IBSHARE_NORM>);
                #elif TILING_KEY_VAR == 1000000000010101012
                    // no anti-quant path for CVDIFF-BSH, half in half out, enable PA
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000002101012
                    // no anti-quant path for CVDIFF-BSH, half in half out, enable L1 reuse
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
                #elif TILING_KEY_VAR == 1000000000100101012
                    // Prefix no anti-quant path for CVDIFF-BSH, half in half out
                    if (maskByteNum == FLOAT16BYTENUM) {
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, half, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                    } else {
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                    }
                #elif TILING_KEY_VAR == 1000000800000101012
                    // anti-quant path for CVDIFF-BSH, half in half out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t>);
                #elif TILING_KEY_VAR == 1000000800100101012
                    // Prefix anti-quant path for CVDIFF-BSH, half in half out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000100001012
                    // Prefix no anti-quant path for CVDIFF-BNSD, half in half out
                    if (maskByteNum == FLOAT16BYTENUM) {
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, half, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                    } else {
                        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                    }
                #elif TILING_KEY_VAR == 1000000000001001012
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_NORM>);
                #elif TILING_KEY_VAR == 1000000000010001012
                    // no anti-quant path for CVDIFF-BNSD, half in half out, enable PA
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000101001012  // enable prefix
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_NORM, true>);
                #elif TILING_KEY_VAR == 1000000000002001012
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
                #elif TILING_KEY_VAR == 1000000800000001012
                    // anti-quant path for CVDIFF-BNSD, half in half out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t>);
                #elif TILING_KEY_VAR == 1000000800100001012
                    // Prefix anti-quant path for CVDIFF-BNSD, half in half out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000400300101612  // enable prefix, enable MSD
                    // BSH layout fp16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400300001612
                    // BNSD layout fp16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200101612
                    // BSH layout fp16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200001612
                    // BNSD layout fp16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 2000000000004000012
                    INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionV, PFATypeNew<PromptFlashAttentionBaseApiTilingData, half, half, half, float, half, float, half, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 2000000010004000012
                    INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLAHighPrecision, PFAHighPrecisionMLAType<PromptFlashAttentionBaseApiTilingData, half, false>);
                #elif TILING_KEY_VAR == 1000000800010101012
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA_ANTIQUANT>);
                #elif TILING_KEY_VAR == 1000000800010001012
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t>);
                #endif
            #endif
    
            #if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY != DT_INT4) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
                TILING_KEY_IS(1000000000000121012);
                TILING_KEY_IS(1000000000010121012);
                TILING_KEY_IS(1000000000100121012);
                TILING_KEY_IS(1000000800000121012);
                TILING_KEY_IS(1000000800100121012);
                TILING_KEY_IS(1000000000000021012);
                TILING_KEY_IS(1000000000010021012);
                TILING_KEY_IS(1000000000100021012);
                TILING_KEY_IS(1000000800000021012);
                TILING_KEY_IS(1000000800100021012);
                TILING_KEY_IS(1000000000000121612);
                TILING_KEY_IS(1000000000010121612);
                TILING_KEY_IS(1000000000100121612);
                TILING_KEY_IS(1000000800000121612);
                TILING_KEY_IS(1000000800100121612);
                TILING_KEY_IS(1000000000000021612);
                TILING_KEY_IS(1000000000010021612);
                TILING_KEY_IS(1000000000100021612);
                TILING_KEY_IS(1000000800000021612);
                TILING_KEY_IS(1000000800100021612);
                TILING_KEY_IS(1000000400300121612);
                TILING_KEY_IS(1000000400300021612);
                TILING_KEY_IS(1000000400200121612);
                TILING_KEY_IS(1000000400200021612);
                #if TILING_KEY_VAR == 1000000000000121012
                    // no anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t>);
                #elif TILING_KEY_VAR == 1000000000010121012
                    // no anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100121012
                    // Prefix no anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000800000121012
                    // anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t>);
                #elif TILING_KEY_VAR == 1000000800100121012
                    // Prefix anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000000021012
                    // no anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t>);
                #elif TILING_KEY_VAR == 1000000000010021012
                    // no anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100021012
                    // Prefix no anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000800000021012
                    // anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t>);
                #elif TILING_KEY_VAR == 1000000800100021012
                    // Prefix anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000000121612
                    // no anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000000010121612
                    // no anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100121612
                    // Prefix no anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000800000121612
                    // anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000800100121612
                    // Prefix anti-quant path for CVDIFF-BSH, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000000021612
                    // no anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000000010021612
                    // no anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100021612
                    // Prefix no anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000800000021612
                    // anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision>);
                #elif TILING_KEY_VAR == 1000000800100021612
                    // Prefix anti-quant path for CVDIFF-BNSD, half in int8 out
                    INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000400300121612  // enable prefix, enable MSD
                    // BSH layout fp16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400300021612
                    // BNSD layout fp16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200121612
                    // BSH layout fp16 in int8 out cvdiff, enable MSD
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200021612
                    // BNSD layout fp16 in int8 out cvdiff, enable MSD
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, OptimizationMode::HighPrecision, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #endif
            #endif
        #endif
        #if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY != DT_INT4)
            TILING_KEY_IS(1000000000000000110);
            TILING_KEY_IS(1000000000000000111);
            TILING_KEY_IS(1000000000000000115);
            TILING_KEY_IS(1000000000000000116);
            TILING_KEY_IS(2000000002004010112);
            TILING_KEY_IS(2000000000004010112);
            TILING_KEY_IS(2000000010004010112);
            TILING_KEY_IS(4000000000000000000);
            TILING_KEY_IS(4000000000000000001);
            TILING_KEY_IS(4000000000000000002);
            TILING_KEY_IS(4000000000000000003);
            TILING_KEY_IS(4000000000000100002);
            TILING_KEY_IS(4000000000000100003);
            TILING_KEY_IS(4000000000010000002);
            TILING_KEY_IS(4000000000010000003);
            TILING_KEY_IS(4000000000010100002);
            TILING_KEY_IS(4000000000010100003);
            TILING_KEY_IS(4000000000020000002);
            TILING_KEY_IS(4000000000020000003);
            TILING_KEY_IS(4000000000020100002);
            TILING_KEY_IS(4000000000020100003);
            #if TILING_KEY_VAR == 1000000000000000110
                // Non-BNSD layout, split NS no tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == 1000000000000000111
                // Non-BNSD layout, split NS with tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == 1000000000000000115
                // BNSD layout, split NS no tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == 1000000000000000116
                // BNSD layout, split NS with tail
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
            #elif TILING_KEY_VAR == 2000000002004010112
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionNoMask, PFAHighPrecisionBaseType<PromptFlashAttentionBaseApiTilingData, float, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, float>);
            #elif TILING_KEY_VAR == 2000000000004010112
                INVOKE_PFA_GENERAL_OP_IMPL_BASE_API(PromptFlashAttentionBaseApiHighPrecisionV,
                                                    PFATypeNew<PromptFlashAttentionBaseApiTilingData, bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t, float, bfloat16_t, OptimizationMode::HighPrecision>);
            #elif TILING_KEY_VAR == 2000000010004010112
                INVOKE_PFA_GENERAL_OP_IMPL_MLA(PromptFlashAttentionBaseMLAHighPrecision, PFAHighPrecisionMLAType<PromptFlashAttentionBaseApiTilingData, bfloat16_t, true>);
            #elif TILING_KEY_VAR == 4000000000000000000
                INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameAB, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float);
            #elif TILING_KEY_VAR == 4000000000000000001
                INVOKE_PFA_GENERAL_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameAB, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float);
            #elif TILING_KEY_VAR == 4000000000000000002
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == 4000000000000000003
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == 4000000000000100002
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == 4000000000000100003
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::NORMAL, false);
            #elif TILING_KEY_VAR == 4000000000010000002
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == 4000000000010000003
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == 4000000000010100002
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == 4000000000010100003
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_ND, true);
            #elif TILING_KEY_VAR == 4000000000020000002
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #elif TILING_KEY_VAR == 4000000000020000003
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #elif TILING_KEY_VAR == 4000000000020100002
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, true, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #elif TILING_KEY_VAR == 4000000000020100003
                INVOKE_PFA_NO_KFC_MLA_OP_IMPL_VAR_LEN(PromptFlashAttentionVarLenScoreSameABBaseApi, MLAGeneralTilingData, ImplModeEnum::AA_HIGH_PRECISION,
                                                        LayOutTypeEnum::LAYOUT_NTD_TND, false, bfloat16_t, float, CubeFormat::NZ,
                                                        MmPolicyType::PA_NZ, true);
            #endif
    
            #if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY != DT_INT4) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
                TILING_KEY_IS(1000000000000111112);
                TILING_KEY_IS(1000000000010111112);
                TILING_KEY_IS(1000000000002111112);
                TILING_KEY_IS(1000000000100111112);
                TILING_KEY_IS(1000000000000011112);
                TILING_KEY_IS(1000000000010011112);
                TILING_KEY_IS(1000000000002011112);
                TILING_KEY_IS(1000000000100011112);
                TILING_KEY_IS(1000000400300111112);
                TILING_KEY_IS(1000000400300011112);
                TILING_KEY_IS(1000000400200111112);
                TILING_KEY_IS(1000000400200011112);
                #if TILING_KEY_VAR == 1000000000000111112
                    // BSH layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t>);
                #elif TILING_KEY_VAR == 1000000000010111112
                    // BSH layout bf16 cvdiff, enable PA
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000002111112
                    // BSH layout bf16 cvdiff, enable L1 reuse
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
                #elif TILING_KEY_VAR == 1000000000100111112  // enable prefix
                    // BSH layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000000011112
                    // BNSD layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t>);
                #elif TILING_KEY_VAR == 1000000000010011112
                    // BNSD layout bf16 cvdiff, enable PA
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000002011112
                    // BNSD layout bf16 cvdiff, enable L1 reuse
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
                #elif TILING_KEY_VAR == 1000000000100011112  // enable prefix
                    // BNSD layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000400300111112  // enable prefix, enable MSD
                    // BSH layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400300011112  // enable prefix, enable MSD
                    // BNSD layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200111112
                    // BSH layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200011112
                    // BNSD layout bf16 cvdiff
                    INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #endif
    
            #endif
    
            #if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY != DT_INT4) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
                TILING_KEY_IS(1000000000000121112);
                TILING_KEY_IS(1000000000010121112);
                TILING_KEY_IS(1000000000100121112);
                TILING_KEY_IS(1000000000000021112);
                TILING_KEY_IS(1000000000010021112);
                TILING_KEY_IS(1000000000100021112);
                TILING_KEY_IS(1000000400300121112);
                TILING_KEY_IS(1000000400300021112);
                TILING_KEY_IS(1000000400200121112);
                TILING_KEY_IS(1000000400200021112);
                #if TILING_KEY_VAR == 1000000000000121112
                    // BSH layout bf16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t>);
                #elif TILING_KEY_VAR == 1000000000010121112
                    // BSH layout bf16 in int8 out cvdiff, enable PA
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100121112  // enable prefix
                    // BSH layout bf16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000000000021112
                    // BNSD layout bf16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t>);
                #elif TILING_KEY_VAR == 1000000000010021112
                    // BNSD layout bf16 in int8 out cvdiff, enable PA
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
                #elif TILING_KEY_VAR == 1000000000100021112  // enable prefix
                    // BNSD layout bf16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, bfloat16_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
                #elif TILING_KEY_VAR == 1000000400300121112  // enable prefix, enable MSD, enable MSD
                    // BSH layout bf16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400300021112  // enable prefix, enable MSD
                    // BNSD layout bf16 in int8 out cvdiff
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200121112
                    // BSH layout bf16 in int8 out cvdiff, enable MSD
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #elif TILING_KEY_VAR == 1000000400200021112
                    // BNSD layout bf16 in int8 out cvdiff, enable MSD
                    INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, false, MsdMode::MSD_ON>);
                #endif
            #endif
        #endif
        #if (ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8) && (ORIG_DTYPE_KEY != DT_INT4)
            TILING_KEY_IS(1000000000000020210);
            TILING_KEY_IS(1000000000000020211);
            TILING_KEY_IS(1000000000000020215);
            TILING_KEY_IS(1000000000000020216);
            TILING_KEY_IS(1000000000000021212);
            TILING_KEY_IS(1000000000010021212);
            TILING_KEY_IS(1000000000100021212);
            TILING_KEY_IS(1000000000000021217);
            TILING_KEY_IS(1000000000010021217);
            TILING_KEY_IS(1000000000100021217);
            #if TILING_KEY_VAR == 1000000000000020210
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, int8_t, bool, CubeFormat::ND, int8_t);
            #elif TILING_KEY_VAR == 1000000000000020211
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, int8_t, bool, CubeFormat::ND, int8_t);
            #elif TILING_KEY_VAR == 1000000000000020215
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, int8_t, bool, CubeFormat::ND, int8_t);
            #elif TILING_KEY_VAR == 1000000000000020216
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, int8_t, bool, CubeFormat::ND, int8_t);
            #elif TILING_KEY_VAR == 1000000000000021212
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t>);
            #elif TILING_KEY_VAR == 1000000000010021212  // enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == 1000000000100021212  // enable prefix
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == 1000000000000021217
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t>);
            #elif TILING_KEY_VAR == 1000000000010021217  // enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == 1000000000100021217  // enable prefix
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #endif
        #endif
        #if (ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY != DT_INT4)
            TILING_KEY_IS(1000000000000000210);
            TILING_KEY_IS(1000000000000000211);
            TILING_KEY_IS(1000000000000000215);
            TILING_KEY_IS(1000000000000000216);
            TILING_KEY_IS(1000000000000001212);
            TILING_KEY_IS(1000000000010001212);
            TILING_KEY_IS(1000000000000001217);
            TILING_KEY_IS(1000000000010001217);
            TILING_KEY_IS(1000000000100001212);
            TILING_KEY_IS(1000000000100001217);
            #if TILING_KEY_VAR == 1000000000000000210
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, int8_t, bool, CubeFormat::ND, half);
            #elif TILING_KEY_VAR == 1000000000000000211
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, int8_t, bool, CubeFormat::ND, half);
            #elif TILING_KEY_VAR == 1000000000000000215
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, int8_t, bool, CubeFormat::ND, half);
            #elif TILING_KEY_VAR == 1000000000000000216
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, int8_t, bool, CubeFormat::ND, half);
            #elif TILING_KEY_VAR == 1000000000000001212
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half>);
            #elif TILING_KEY_VAR == 1000000000010001212  // enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == 1000000000000001217
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half>);
            #elif TILING_KEY_VAR == 1000000000010001217  // enable PA
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_PA>);
            #elif TILING_KEY_VAR == 1000000000100001212  // enable prefix
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #elif TILING_KEY_VAR == 1000000000100001217  // enable prefix
                INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half, int8_t, OptimizationMode::HighPerformance, MatMulType::MM_MDL, true>);
            #endif
        #endif
        TILING_KEY_IS(1000000000000000020);
        #if TILING_KEY_VAR == 1000000000000000020
            // kv is empty tensor, return zero output
            TPipe tPipe;
            INVOKE_PFA_TILING_DATA(tiling);
            PromptFlashAttentionEmptyTensor<half> op;
            op.Init(attentionOut, tiling_data, &tPipe);
            op.Process();
            return;
        #endif
    #else
        GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
        auto maskByteNum = baseParams.maskTypeByteNum;
    
        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        TILING_KEY_IS(1000000000000112288);
        TILING_KEY_IS(1000000000000122288);
        TILING_KEY_IS(1000000000000012288);
        TILING_KEY_IS(1000000000000022288);
        TILING_KEY_IS(1000000000000012888);
        TILING_KEY_IS(1000000000000022888);
        #if TILING_KEY_VAR == 1000000000000112288
            INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
        #elif TILING_KEY_VAR == 1000000000000122288
            INVOKE_PFA_NEW_GQA_OP_IMPL(PromptAttentionPrefill, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t>, PrecType::BMM1_FP16_EXP_FP32);//高性能
        #elif TILING_KEY_VAR == 1000000000000012288
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half>);
        #elif TILING_KEY_VAR == 1000000000000022288
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half>);
        #elif TILING_KEY_VAR == 1000000000000012888
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
        #elif TILING_KEY_VAR == 1000000000000022888
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half, half, ModeNZ::HighPrecisionNZ>);
        #endif
    #endif
    }    
}

extern "C" __global__ __aicore__ void prompt_flash_attention(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* attentionOut,
                                                             __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    prompt_flash_attention_FIAS(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                quant_offset2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, attentionOut, nullptr, workspace, tiling);
}