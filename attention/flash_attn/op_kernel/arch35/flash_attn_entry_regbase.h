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

#include "fia_template_dispatcher.h"


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
// AICPU metadata buffer layout: [0]=sectionNum, [1..15]=padding, [16..]=FA+FD data.
// Skip the 16 × uint32 header to reach FAMetadata[0][0].
// metadata must be non-null (provided by npu_flash_attn_metadata AICPU operator).
// constexpr uint32_t FA_METADATA_HEADER_OFFSET = 16U * sizeof(uint32_t);
// __gm__ uint8_t *fiaMetaDataPtr = metadata + FA_METADATA_HEADER_OFFSET;
// [DBG] Dump raw metadata: sectionNum header + FA[0] 7 fields (BN2_START..FD_WS)
// if (AscendC::GetBlockIdx() == 0) {
//     auto *rawMeta = (__gm__ uint32_t *)metadata;
//     auto *fa0     = (__gm__ uint32_t *)fiaMetaDataPtr;  // points to FAMetadata[0][0]
//     // AscendC::printf("[META raw] sectionNum=%u FA[0]={bn2s=%u ms=%u s2s=%u bn2e=%u me=%u s2e=%u fdws=%u}\n",
//     //     rawMeta[0], fa0[0], fa0[1], fa0[2], fa0[3], fa0[4], fa0[5], fa0[6]);
// }
// BNSD
// AscendC::printf("query: %p, key: %p, value: %p", query, key, value);
// GlobalTensor<half> qBuff;
// GlobalTensor<half> kBuff;
// GlobalTensor<half> vBuff;
// qBuff.SetGlobalBuffer((__gm__ half *)query);
// kBuff.SetGlobalBuffer((__gm__ half *)key);
// vBuff.SetGlobalBuffer((__gm__ half *)value);
// AscendC::DumpTensor(qBuff, 1, 128);
// AscendC::DumpTensor(kBuff, 2, 128);
// AscendC::DumpTensor(vBuff, 3, 128);

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

// #if __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // REGISTER_TILING_FOR_TILINGKEY(
    //     "((TILING_KEY_VAR >> 22) & 0x1f) != 30 && ((TILING_KEY_VAR >> 8) & 0x12) != 9",
    //     FusedInferAttentionScoreTilingData);
// #if (ORIG_DTYPE_Q == DT_FLOAT)
//     AscendC::printf("-----------DT_FLOAT----------\n");
//     INVOKE_FA_IMPL(BaseApi::FlashAttnKernelBase, float, float, float, ImplModeEnum(0), layout,
//                    s1TemplateType, s2TemplateType, dTemplateType, dvTemplateType, PseTypeEnum(9), hasAttenMask, 0,
//                    0);
//     return;
// #endif
// #if (ORIG_DTYPE_Q == DT_BF16)
//     AscendC::printf("-----------DT_BF16----------\n");
//     INVOKE_FA_IMPL(BaseApi::FlashAttnKernelBase, bfloat16_t, float, bfloat16_t, ImplModeEnum(0), layout,
//                    s1TemplateType, s2TemplateType, dTemplateType, dvTemplateType, PseTypeEnum(9), hasAttenMask, 0,
//                    0);
//     return;
// #endif
// #if (ORIG_DTYPE_Q == DT_FLOAT16)
//     AscendC::printf("-----------DT_FLOAT16----------\n");
//     INVOKE_FA_IMPL(BaseApi::FlashAttnKernelBase, half, float, half, ImplModeEnum(0), layout,
//                    s1TemplateType, s2TemplateType, dTemplateType, dvTemplateType, PseTypeEnum(9), hasAttenMask, 0,
//                    0);
//     return;
// #endif
// #endif
// return;
// if (AscendC::GetBlockIdx() == 0) {
//     // ---- 模板参数（编译期常量）----
//     AscendC::printf("[run_fia] inOutLayoutType=%u  config=%u  layout=%u\n",
//         (uint32_t)inOutLayoutType, (uint32_t)config, (uint32_t)layout);
//     AscendC::printf("[run_fia] s1=%u  s2=%u  d=%u  dv=%u\n",
//         (uint32_t)s1TemplateType, (uint32_t)s2TemplateType,
//         (uint32_t)dTemplateType,  (uint32_t)dvTemplateType);
//     AscendC::printf("[run_fia] pseMode=%u  quantMode=%u  hasAttenMask=%u  hasRope=%u\n",
//         (uint32_t)pseMode, (uint32_t)quantMode,
//         (uint32_t)hasAttenMask, (uint32_t)hasRope);
//     AscendC::printf("[run_fia] KvLayoutType=%u  isFd=%u  emptyTensor=%u"
//                     "  PFAMask=%u  pFAMatMulType=%u\n",
//         (uint32_t)KvLayoutType, (uint32_t)isFd, (uint32_t)emptyTensor,
//         (uint32_t)PFAMask, (uint32_t)pFAMatMulType);
//     AscendC::printf("[run_fia] enableKVPrefix=%u  enableS1OutSplit=%u\n",
//         (uint32_t)enableKVPrefix, (uint32_t)enableS1OutSplit);
//     // ---- 运行时指针（非空/空状态）----
//     AscendC::printf("[run_fia] ptrs: query=%p  key=%p  value=%p\n",
//         query, key, value);
//     AscendC::printf("[run_fia] ptrs: sequsedQ=%p  sequsedKv=%p  blockTable=%p\n",
//         sequsedQ, sequsedKv, blockTable);
//     AscendC::printf("[run_fia] ptrs: sinks=%p  attentionOut=%p  softmaxLse=%p\n",
//         sinks, attentionOut, softmaxLse);
//     AscendC::printf("[run_fia] ptrs: user=%p  tiling=%p  fiaMetaDataPtr=%p\n",
//         user, tiling, fiaMetaDataPtr);
//     // ---- seqused 内容（int32，最多打印前 8 个）----
//     if (sequsedQ != nullptr) {
//         auto *sq = (__gm__ int32_t *)sequsedQ;
//         AscendC::printf("[run_fia] sequsedQ[0..3]= %d %d %d %d\n",
//             sq[0], sq[1], sq[2], sq[3]);
//     }
//     if (sequsedKv != nullptr) {
//         auto *sk = (__gm__ int32_t *)sequsedKv;
//         AscendC::printf("[run_fia] sequsedKv[0..3]= %d %d %d %d\n",
//             sk[0], sk[1], sk[2], sk[3]);
//     }
// }
// cu_seqlens 格式为 [0, s1, s1+s2, ..., T]（B+1 个元素，有前导 0）
// ACCUM parser 期望 seqused 格式 [s1, s1+s2, ..., T]（B 个元素，无前导 0）
// 若检测到前导 0，则剥离：指针前移 8 字节
// [DBG] Dump attenMask values

// if (AscendC::GetBlockIdx() == 0) {
//     AscendC::printf("[flash_attn_entry] attnMask=%p hasAttenMask=%u\n", attnMask, (uint32_t)hasAttenMask);
//     if (attnMask != nullptr) {
//         AscendC::GlobalTensor<int8_t> maskGm;
//         maskGm.SetGlobalBuffer((__gm__ int8_t *)attnMask, 2048);
//         AscendC::DumpTensor(maskGm, 100, 2048);
//     }
// }

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

// // TODO: 根据isPA选择推理kernel（PA场景）或训练kernel（非PA场景）
// #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
// if (!isPA) {
//     if (isSoftmaxLse) {
//         // 训练场景（returnSoftmaxLse=1）
//         INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelTrain,
//             half, float, half, ImplModeEnum(implMode), LayOutTypeEnum(layout),
//             S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
//             DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
//             PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
//             hasAtten, false, false);
//     } else {
//         // 推理场景（returnSoftmaxLse=0）
//         INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelInfer,
//             half, float, half, ImplModeEnum(implMode), LayOutTypeEnum(layout),
//             S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
//             DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
//             PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
//             hasAtten, false, false);
//     }
// }
// // TODO: PA场景kernel分发
// return;
// #endif

// #if (ORIG_DTYPE_QUERY == DT_BF16)
// if (!isPA) {
//     if (isSoftmaxLse) {
//         INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelTrain,
//             bfloat16_t, float, bfloat16_t, ImplModeEnum(implMode), LayOutTypeEnum(layout),
//             S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
//             DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
//             PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
//             hasAtten, false, false);
//     } else {
//         INVOKE_FA_IMPL(BaseApi::FlashAttnScoreKernelInfer,
//             bfloat16_t, float, bfloat16_t, ImplModeEnum(implMode), LayOutTypeEnum(layout),
//             S1TemplateType(s1TemplateType), S2TemplateType(s2TemplateType),
//             DTemplateType(dTemplateType), DTemplateType(dvTemplateType == 0 ? dTemplateType : dvTemplateType),
//             PseTypeEnum(static_cast<uint8_t>(PseType::PSE_NONE_TYPE)),
//             hasAtten, false, false);
//     }
// }
// return;
// #endif
}

#endif // FLASH_ATTN_ENTRY_REGBASE_H_