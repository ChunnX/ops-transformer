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
 * \file fused_infer_attention_score_v3.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "fused_infer_attention_score_tilingkey.h"

#ifdef FIA_ENABLE_MLA
// mla模板使用私有tiling结构，框架编译时根据一组DType预编译获取keylist，根据keylist找到对应的tiling结构
// 在这组DType中，若没有mla模板的key，包含mla模板编译会报错：unknown type name 'FusedInferAttentionScoreTilingData'
#if ((ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)) || \
    ((ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16))
#include "../../common/op_kernel/arch32/fia_kernel_nonquant_mla.h"
#endif
#endif // FIA_ENABLE_MLA

using namespace AscendC;

#define INVOKE_FIA_NO_KFC_MLA_OP_IMPL(templateClass, ...)                                                              \
    do {                                                                                                               \
        templateClass<FIAType<__VA_ARGS__>> op;                                                                        \
        FIA_COPY_TILING_DATA(FusedInferAttentionScoreTilingData, tiling);                                              \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, blocktable, kvPaddingSize,\
            queryRope, keyRope, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                          \
        op.InitQuant(deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,    \
                     keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,                 \
                     keyRopeAntiquantScale, user);                                                                     \
        op.Process();                                                                                                  \
    } while (0)

#define FIA_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                 \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;

extern "C" __global__ __aicore__ void fused_infer_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)

#elif (__CCE_AICORE__ == 200)

#else
    TPipe tPipe;

    /*
    获取Op可用WorkSpace空间
    **/
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    // fp16 7buf_nz
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // fp16 7buf_nd
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
// fp16 7buf_nz
#if TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, false,
                                  FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, true,
                                  FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, false,
                                  FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, true,
                                  FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
// fp16 7buf_nd
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, false,
                                  FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, true,
                                  FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, false,
                                  FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, true,
                                  FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, false, false,
                                  FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, false, true,
                                  FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, false,
                                  FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, true,
                                  FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, false,
                                  FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, true,
                                  FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, false,
                                  FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QF16_KVF16_OUTF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, half, half, half, half, true, true,
                                  FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16) && (ORIG_DTYPE_KEY == DT_BF16)
    // bfl6 7buf_nz
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    // 7buf_nd
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING);
    TILING_KEY_IS(QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING);

// bf16 7buf_nz
#if TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVNZ_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::NZ);
// #ifdef ND_7BUFFER
// 7buf_nd
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BNSD_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BNSD, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, false, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                   bfloat16_t, false, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_BSH_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::BSH, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBSH_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BSH);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_TND_KVBNSD_PAGEDCACHE_FLASHDECODING_MLA_TILING // 7buf
    INVOKE_FIA_NO_KFC_MLA_OP_IMPL(FiaKernelNonQuantMla, bfloat16_t, bfloat16_t, bfloat16_t,
                                  bfloat16_t, true, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::BNSD);
#endif
#endif

#endif
}
