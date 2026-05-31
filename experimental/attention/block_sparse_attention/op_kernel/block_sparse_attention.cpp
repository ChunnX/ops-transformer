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
 * \file block_sparse_attention.cpp
 * \brief BlockSparseAttention kernel entry: dispatch into the kernel-side Process().
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "block_sparse_attention_template_tiling_key.h"
#include "block_sparse_attention_common.h"
#include "arch35/block_sparse_attention_kernel.h"

using namespace AscendC;

// 仿 kv_quant_sparse_attn_sharedkv.cpp 的 kernel 侧调用写法：
//   - 按 g_coreType 选择实/Dummy 的 Cube/Vec block（AIC 跑真 Cube + Dummy Vec，AIV 跑 Dummy Cube + 真 Vec）
//   - 实例化 BlockSparseAttentionKernel<CubeBlockType, VecBlockType>
//   - GET_TILING_DATA -> InitBaseAPI -> Process()
// __VA_ARGS__ 为 block 模板参数（严格对齐 CUBE_BLOCK_TRAITS 字段顺序，共 17 个）。
#define BSA_OP_IMPL(...)                                                                                 \
    do {                                                                                                 \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,                      \
            BaseApi::BSABlockCube<__VA_ARGS__>, BaseApi::BSABlockCubeDummy<__VA_ARGS__>>::type;          \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                       \
            BaseApi::BSABlockVecDummy<__VA_ARGS__>, BaseApi::BSABlockVec<__VA_ARGS__>>::type;            \
        BaseApi::BlockSparseAttentionKernel<CubeBlockType, VecBlockType> op;                             \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling);      \
        const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn;            \
        op.InitBaseAPI(query, key, value, /*dropMask*/nullptr, /*paddingMask*/nullptr,   \
            attenMask, /*prefix*/nullptr, /*actualSeqLengths*/seqUsedQ, /*actualSeqLengthsKv*/seqUsedKv, \
            blockTable, /*queryPaddingSize*/nullptr, /*kvPaddingSize*/nullptr,                           \
            /*deqScaleQ*/qScale, /*deqScaleK*/kScale, /*deqScaleV*/vScale, /*pScale*/pScale,             \
            /*postQuantScale*/nullptr, /*postQuantOffset*/nullptr,                                       \
            /*keySharedPrefix*/nullptr, /*valueSharedPrefix*/nullptr, /*actualSharedPrefixLen*/nullptr,  \
            /*queryRope*/nullptr, /*keyRope*/nullptr, /*learnableSink*/nullptr,                          \
            /*softmaxMax*/nullptr, /*softmaxSum*/nullptr, /*softmaxOut*/nullptr,                         \
            softmaxLse, attentionOut, /*workspace*/user, tilingData, &tPipe);                            \
        op.Process();                                                                                    \
    } while (0)

template <uint32_t QKV_DTYPE, uint32_t LAYOUT_T, uint32_t KV_LAYOUT_T, uint32_t COMPRESS_MODE,
          bool RETURN_SOFTMAX_LSE>
__global__ __aicore__ void block_sparse_attention(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *qScale,
    __gm__ uint8_t *kScale,
    __gm__ uint8_t *vScale,
    __gm__ uint8_t *pScale,
    __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *cuSeqlensKv,
    __gm__ uint8_t *seqUsedQ,
    __gm__ uint8_t *seqUsedKv,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseSeqLen,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *attenMask,
    __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // TODO: 待实现 - 以下 BSA 专有输入暂未接入 InitBaseAPI（其形参无对应位），随 sparse 流水后续接入：
    //   cuSeqlensQ/cuSeqlensKv（变长 TND）、sparseIndices/sparseSeqLen（稀疏块寻址）、metadata（调度）。
    (void)cuSeqlensQ;
    (void)cuSeqlensKv;
    (void)sparseIndices;
    (void)sparseSeqLen;
    (void)metadata;
    // 压缩 mask 模式 / softmaxLse 使能属运行期信息（经 tilingData / 输出张量），不进 block 模板参数。
    (void)COMPRESS_MODE;
    (void)RETURN_SOFTMAX_LSE;

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    // layout: BSND -> LAYOUT_BSH（B,S,H≡B,S,N*D，LayOutTypeEnum 无 LAYOUT_BSND）; TND -> LAYOUT_TND
    constexpr LayOutTypeEnum bsaLayout =
        (LAYOUT_T == BSA_LAYOUT_TND) ? LayOutTypeEnum::LAYOUT_TND : LayOutTypeEnum::LAYOUT_BSH;
    // isPa: KV 连续 -> false; PA_ND -> true
    constexpr bool bsaIsPa = (KV_LAYOUT_T == BSA_KV_LAYOUT_PA_ND);

    // block 模板参数顺序对齐 CUBE_BLOCK_TRAITS：
    // INPUT_T, T(计算 float), OUTPUT_T(BF16), implMode, layout, s1Tpl, s2Tpl, dTpl, dVTpl,
    // pseMode, hasAtten, hasDrop, hasRope, isInfer, isPa, enableKVPrefix, enableS1OutSplit
    // TODO: 待实现 - host tiling 当前产出 BlockSparseAttentionTilingData，kernel 侧 InitBaseAPI 取
    //       FlashAttentionScoreSimplifiedTilingData，host/kernel tiling 结构对齐留作后续。
    if constexpr (QKV_DTYPE == BSA_DTYPE_FP8_E4M3FN) {
        BSA_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, ImplModeEnum::AA_HIGH_PRECISION, bsaLayout,
            S1TemplateType::Aligned128, S2TemplateType::Aligned256, DTemplateType::Aligned128,
            DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, bsaIsPa,
            false, false);
    } else {
        BSA_OP_IMPL(hifloat8_t, float, bfloat16_t, ImplModeEnum::AA_HIGH_PRECISION, bsaLayout,
            S1TemplateType::Aligned128, S2TemplateType::Aligned256, DTemplateType::Aligned128,
            DTemplateType::Aligned128, PseTypeEnum::PSE_NONE_TYPE, true, false, false, true, bsaIsPa,
            false, false);
    }
}
