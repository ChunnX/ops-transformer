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
 * \file sparse_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "sparse_flash_attention_pioneer_template_tiling_key.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/sparse_flash_attention_pioneer_kernel_mla.h"
#endif

using namespace AscendC;

#if (__CCE_AICORE__ == 310)
#if defined(__DAV_C310_CUBE__)
#define QSFA_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::QSFAMatmulService<__VA_ARGS__>, BaseApi::QSFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::QSFAVectorServiceDummy<__VA_ARGS__>, BaseApi::QSFAVectorService<__VA_ARGS__>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                       \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
	    blocktable, queryRope, keyRope, attentionOut, softmaxMax, softmaxSum, user, nullptr, tiling, &tPipe);         \
        op.Process();                                                                             \
    } while (0)
#else
#define QSFA_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::QSFAMatmulService<__VA_ARGS__>, BaseApi::QSFAMatmulServiceDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::QSFAVectorServiceDummy<__VA_ARGS__>, BaseApi::QSFAVectorService<__VA_ARGS__>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                       \
        const tilingdataClass *__restrict tilingData = &tiling_data_in;                                     \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
	    blocktable, queryRope, keyRope, attentionOut, softmaxMax, softmaxSum, user, tilingData, tiling, &tPipe);         \
        op.Process();                                                                             \
    } while (0)
#endif
#endif

template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
 __global__ __aicore__ void
sparse_flash_attention_pioneer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                       __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blocktable,
                       __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV,
                       __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
                       __gm__ uint8_t *attentionOut, __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum,
                       __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    
#if (__CCE_AICORE__ == 310)
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionPioneerMla, SparseFlashAttentionPioneerTilingDataMla, bfloat16_t, bfloat16_t,
                float, bfloat16_t, FLASH_DECODE, true, static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
                static_cast<QSFATemplateMode>(TEMPLATE_MODE));
    } else {
        QSFA_OP_IMPL(BaseApi::KvQuantSparseFlashAttentionPioneerMla, SparseFlashAttentionPioneerTilingDataMla, half, half,
                float, half, FLASH_DECODE, true, static_cast<QSFA_LAYOUT>(LAYOUT_T), static_cast<QSFA_LAYOUT>(KV_LAYOUT_T),
                static_cast<QSFATemplateMode>(TEMPLATE_MODE));
    }
#endif
}