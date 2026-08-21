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
 * \file fia_arch35_noquant.h
 * \brief arch35 FIA 非量化路由
 */

#ifndef FIA_ARCH35_NOQUANT_H_
#define FIA_ARCH35_NOQUANT_H_

#include "fia_arch35_common.h"
#include "fia_arch35_template_tiling_key_enum.h"
#include "flash_attention_noquant_kernel_infer.h"
#include "flash_attention_kernel_noquant_mla.h"
#include "fia_template_dispatcher.h"

using namespace regbaseutil;

#define REGBASE_COPY_TILING_DATA_ASCEND950_KVSAME_BASEAPI(tiling) \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreSimplifiedTilingData, tilingDataIn, tiling); \
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = &tilingDataIn

template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope,
          uint8_t KvLayoutType, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit,
          bool isReconstructTemp>
inline __aicore__ void FiaDiagPrintNoQuantEntry(
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *blockTable,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData)
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0 || GetSubBlockIdx() != 0) {
            return;
        }
        const auto &input = tilingData->inputParamsRegbase;
        const auto &multiCore = tilingData->multiCoreParamsRegbase;
        AscendC::PRINTF("[FIA_DIAG][KERNEL][NOQUANT] stage=entry layout=%u config=%u quant=%u mask=%u rope=%u pa=%u\n",
                        inOutLayoutType, config, quantMode, hasAttenMask, hasRope, KvLayoutType);
        AscendC::PRINTF("[FIA_DIAG][KERNEL][NOQUANT] b=%ld n2=%ld g=%ld s1=%ld s2=%ld d=%ld dv=%ld ropeD=%ld\n",
                        input.bSize, input.n2Size, input.gSize, input.s1Size, input.s2Size, input.dSize, input.dSizeV,
                        input.dSizeRope);
        AscendC::PRINTF("[FIA_DIAG][KERNEL][NOQUANT] coreNum=%d qDims=%u kvDims=%u transpose=%u sparse=%u maskMode=%u\n",
                        multiCore.coreNum, input.actualSeqLengthsSize, input.actualSeqLengthsKVSize,
                        input.transposeLayout, input.sparseType, input.attenMaskCompressMode);
        AscendC::PRINTF("[FIA_DIAG][KERNEL][NOQUANT] total=%ld s1Outer=%ld split=%ld tail=%ld splitMode=%u firstFull=%ld\n",
                        multiCore.totalSize, multiCore.s1OuterSize, multiCore.splitFactorSize,
                        multiCore.splitFactorTailSize, multiCore.splitCoreMode, multiCore.firstFullLoadS1OuterIdx);
        AscendC::PRINTF("[FIA_DIAG][KERNEL][NOQUANT] blockSize=%d tableDim2=%d totalBlocks=%d qPad=%u kvPad=%u postCh=%u postBf16=%u\n",
                        input.blockSize, input.blockTableDim2, input.paBlockNumSum, input.isQHasLeftPadding,
                        input.isKVHasLeftPadding, input.isPostQuantPerChnl, input.isPostQuantBF16);

        const bool isPaDiagCase = KvLayoutType != 0 && input.bSize == 36 && input.s1Size == 6998 &&
                                  input.s2Size == 7920 && input.dSize == 64 && input.dSizeV == 64 &&
                                  input.blockSize == 128 && input.blockTableDim2 == 62;
        if (isPaDiagCase && actualSeqLengths != nullptr && actualSeqLengthsKV != nullptr && blockTable != nullptr) {
            __gm__ int64_t *actualQ = (__gm__ int64_t *)actualSeqLengths;
            __gm__ int64_t *actualKv = (__gm__ int64_t *)actualSeqLengthsKV;
            __gm__ int32_t *table = (__gm__ int32_t *)blockTable;
            for (int32_t batch = 0; batch < input.bSize; ++batch) {
                const int64_t usedBlocks = (actualKv[batch] + input.blockSize - 1) / input.blockSize;
                const int64_t scanBlocks = usedBlocks < input.blockTableDim2 ? usedBlocks : input.blockTableDim2;
                const int64_t rowOffset = static_cast<int64_t>(batch) * input.blockTableDim2;
                int64_t invalidCount = 0;
                int64_t firstBadSlot = -1;
                int32_t firstBadValue = -1;
                for (int64_t slot = 0; slot < scanBlocks; ++slot) {
                    const int32_t blockId = table[rowOffset + slot];
                    if (blockId < 0 || blockId >= input.paBlockNumSum) {
                        ++invalidCount;
                        if (firstBadSlot < 0) {
                            firstBadSlot = slot;
                            firstBadValue = blockId;
                        }
                    }
                }
                const int32_t firstBlock = usedBlocks > 0 ? table[rowOffset] : -1;
                const int32_t lastBlock = usedBlocks > 0 && usedBlocks <= input.blockTableDim2 ?
                                              table[rowOffset + usedBlocks - 1] :
                                              -1;
                AscendC::PRINTF("[FIA_DIAG][KERNEL][PA] b=%d q=%ld kv=%ld used=%ld first=%d last=%d invalid=%ld\n",
                                batch, actualQ[batch], actualKv[batch], usedBlocks, firstBlock, lastBlock,
                                invalidCount);
                if (firstBadSlot >= 0) {
                    AscendC::PRINTF("[FIA_DIAG][KERNEL][PA] b=%d firstBadSlot=%ld firstBadValue=%d totalBlocks=%d\n",
                                    batch, firstBadSlot, firstBadValue, input.paBlockNumSum);
                }
            }
        }
    }
}

inline __aicore__ void FiaDiagPrintNoQuantStage(
    uint32_t stage, const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData)
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0 && GetSubBlockIdx() == 0) {
            AscendC::PRINTF("[FIA_DIAG][KERNEL][NOQUANT] stage=%u coreNum=%d needInit=%u\n", stage,
                            tilingData->multiCoreParamsRegbase.coreNum, tilingData->initOutputParams.needInit);
        }
    }
}

#define INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(templateClass, ...) \
    do { \
        TPipe tPipe; \
        __gm__ uint8_t *user = GetUserWorkspace(workspace); \
        REGBASE_COPY_TILING_DATA_ASCEND950_KVSAME_BASEAPI(tiling); \
        using CubeBlockType = FABlockCubeNoquantMla<__VA_ARGS__>; \
        using VecBlockType = BaseApi::FANoQuantBlockVecInfer<__VA_ARGS__>; \
        templateClass<CubeBlockType, VecBlockType> op; \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, \
                postQuantScale, postQuantOffset, queryRope, keyRope, softmaxLse, attentionOut, user, tilingData, \
                &tPipe); \
        op.Process(); \
    } while (0)

#define INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(templateClass, vec1ResultSize, qkvSize, ...) \
    do { \
        if (query == nullptr) { \
            return; \
        } \
        FIA_REGBASE_COPY_TILING_DATA(tiling); \
        TPipe tPipe; \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FANoQuantBlockCube<__VA_ARGS__>, \
                                      BaseApi::FANoQuantBlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::FANoQuantBlockVecDummy<__VA_ARGS__>, \
                                      BaseApi::FANoQuantBlockVecInfer<__VA_ARGS__>>::type; \
        templateClass<CubeBlockType, VecBlockType> op; \
        FiaDiagPrintNoQuantEntry<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, KvLayoutType, \
                                 isFd, emptyTensor, enableKVPrefix, enableS1OutSplit, isReconstructTemp>( \
            actualSeqLengths, actualSeqLengthsKV, blocktable, tilingData); \
        FiaDiagPrintNoQuantStage(0, tilingData); \
        op.InitBaseAPI(query, key, value, pseShift, nullptr, nullptr, attenMask, nullptr, actualSeqLengths, \
                       actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, dequantScaleQuery, \
                       key_antiquant_scale, value_antiquant_scale, nullptr, postQuantScale, postQuantOffset, \
                       keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, learnableSink, \
                       nullptr, nullptr, nullptr, softmaxLse, attentionOut, user, tilingData, &tPipe); \
        FiaDiagPrintNoQuantStage(1, tilingData); \
        op.Process(); \
        FiaDiagPrintNoQuantStage(2, tilingData); \
    } while (0)

template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope,
          uint8_t KvLayoutType, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit,
          bool isReconstructTemp>
inline __aicore__ void fia_noquant_regbase(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *deq_scale1, __gm__ uint8_t *quant_scale1, __gm__ uint8_t *deq_scale2,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *antiquant_scale,
    __gm__ uint8_t *antiquant_offset, __gm__ uint8_t *blocktable, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *key_antiquant_scale, __gm__ uint8_t *key_antiquant_offset,
    __gm__ uint8_t *value_antiquant_scale, __gm__ uint8_t *value_antiquant_offset, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr bool isPa = KvLayoutType != 0;
#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    // 解析两个合并字段
    if constexpr (isReconstructTemp == true) {
        if constexpr (config == Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 && hasRope) {
            run_fia_noquant_mla_kernel<half, half, inOutLayoutType, config, hasAttenMask, KvLayoutType,
                                       isFd, emptyTensor, enableS1OutSplit>(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV,
                                                                            blocktable, queryRope, keyRope, attentionOut, softmaxLse, user, tiling);
        } else {
            run_fia_noquant_gqa_kernel<half, half, inOutLayoutType, config, hasAttenMask, KvLayoutType, isFd,
                                       emptyTensor, enableS1OutSplit>(query, key, value, attenMask, actualSeqLengths,
                                                                      actualSeqLengthsKV, blocktable, queryRope, keyRope, attentionOut, softmaxLse, user, tiling);
        }
    } else {
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd,
                             emptyTensor, enableKVPrefix, enableS1OutSplit);
        // 计算参数，这个地方必须先用constexpr将表达式的值计算出来，否则INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI会报结构体的某些变量不存在
        // 原因：不使用constexpr，所有组合都会在编译阶段进入该函数，因此会出现hasRope字段为false的情况导致变量不存在
        if constexpr (dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
                FAKernelNoquantMla, half, float, half, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType,
                s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false,
                true, true, isPa,
                false); // 实际模板参数hasRope为false，但模板需要其为true，选择在kernel直接写入，tiling不做修改
            return;
        }
        constexpr uint64_t vec1ResultSize =
            static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr (dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 =
                MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                        (static_cast<uint64_t>(dTemplateType) >> 1),
                    static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
                2;
            INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
                BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
                enableKVPrefix, enableS1OutSplit);
        } else {
            constexpr uint64_t qkvSizeRsv2 =
                MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                        static_cast<uint64_t>(dTemplateType),
                    static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
                2;
            INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
                BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, half,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
                enableKVPrefix, enableS1OutSplit);
        }
    }

#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    // 解析两个合并字段
    if constexpr (isReconstructTemp == true) {
        if constexpr (config == Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 && hasRope) {
            run_fia_noquant_mla_kernel<bfloat16_t, bfloat16_t, inOutLayoutType, config, hasAttenMask, KvLayoutType,
                                       isFd, emptyTensor, enableS1OutSplit>(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsKV,
                                                                            blocktable, queryRope, keyRope, attentionOut, softmaxLse, user, tiling);
        } else {
            run_fia_noquant_gqa_kernel<bfloat16_t, bfloat16_t, inOutLayoutType, config, hasAttenMask, KvLayoutType,
                                       isFd, emptyTensor, enableS1OutSplit>(query, key, value, attenMask, actualSeqLengths,
                                                                            actualSeqLengthsKV, blocktable, queryRope, keyRope, attentionOut, softmaxLse, user, tiling);
        }
    } else {
        PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd,
                             emptyTensor, enableKVPrefix, enableS1OutSplit);
        // 计算参数，这个地方必须先用constexpr将表达式的值计算出来，否则INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI会报结构体的某些变量不存在
        // 原因：不使用constexpr，所有组合都会在编译阶段进入该函数，因此会出现hasRope字段为false的情况导致变量不存在
        if constexpr (dTemplateType == DTemplateType::Aligned576) {
            INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
                FAKernelNoquantMla, bfloat16_t, float, bfloat16_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
                s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
                hasAttenMask, false, true, true, isPa, false);
            return;
        }

        constexpr uint64_t vec1ResultSize =
            static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
        if constexpr (dTemplateType == DTemplateType::Aligned512) {
            constexpr uint64_t qkvSizeRsv2 =
                MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                        (static_cast<uint64_t>(dTemplateType) >> 1),
                    static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
                2;
            INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
                BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
                enableKVPrefix, enableS1OutSplit);
        } else {
            constexpr uint64_t qkvSizeRsv2 =
                MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                        static_cast<uint64_t>(dTemplateType),
                    static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
                2;
            INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
                BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, bfloat16_t,
                ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
                enableKVPrefix, enableS1OutSplit);
        }
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    // 解析两个合并字段
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                         enableKVPrefix, enableS1OutSplit);
    // 计算参数，这个地方必须先用constexpr将表达式的值计算出来，否则INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI会报结构体的某些变量不存在
    // 原因：不使用constexpr，所有组合都会在编译阶段进入该函数，因此会出现hasRope字段为false的情况导致变量不存在
    if constexpr (dTemplateType == DTemplateType::Aligned576) {
        INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
            FAKernelNoquantMla, half, float, int8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType,
            s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, true,
            true, isPa, false);
        return;
    }
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (dTemplateType == DTemplateType::Aligned512) {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, int8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    } else {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, int8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    // 解析两个合并字段
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                         enableKVPrefix, enableS1OutSplit);
    if constexpr (dTemplateType == DTemplateType::Aligned576) {
        INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
            FAKernelNoquantMla, half, float, hifloat8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
            hasAttenMask, false, true, true, isPa, false);
        return;
    }
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (dTemplateType == DTemplateType::Aligned512) {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, hifloat8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    } else {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, hifloat8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    // 解析两个合并字段
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                         enableKVPrefix, enableS1OutSplit);
    if constexpr (dTemplateType == DTemplateType::Aligned576) {
        INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
            FAKernelNoquantMla, half, float, fp8_e4m3fn_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
            hasAttenMask, false, true, true, isPa, false);
        return;
    }
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (dTemplateType == DTemplateType::Aligned512) {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, fp8_e4m3fn_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    } else {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, half, float, fp8_e4m3fn_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
    // 解析两个合并字段
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                         enableKVPrefix, enableS1OutSplit);
    if constexpr (dTemplateType == DTemplateType::Aligned576) {
        INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
            FAKernelNoquantMla, bfloat16_t, float, int8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
            hasAttenMask, false, true, true, isPa, false);
        return;
    }
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (dTemplateType == DTemplateType::Aligned512) {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, int8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    } else {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, int8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_HIFLOAT8)
    // 解析两个合并字段
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                         enableKVPrefix, enableS1OutSplit);
    if constexpr (dTemplateType == DTemplateType::Aligned576) {
        INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
            FAKernelNoquantMla, bfloat16_t, float, hifloat8_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
            hasAttenMask, false, true, true, isPa, false);
        return;
    }
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (dTemplateType == DTemplateType::Aligned512) {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, hifloat8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    } else {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, hifloat8_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT8_E4M3FN)
    // 解析两个合并字段
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor,
                         enableKVPrefix, enableS1OutSplit);
    if constexpr (dTemplateType == DTemplateType::Aligned576) {
        INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI(
            FAKernelNoquantMla, bfloat16_t, float, fp8_e4m3fn_t, ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType,
            s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
            hasAttenMask, false, true, true, isPa, false);
        return;
    }
    constexpr uint64_t vec1ResultSize =
        static_cast<uint64_t>(s1TemplateType) * static_cast<uint64_t>(s2TemplateType) * 2;
    if constexpr (dTemplateType == DTemplateType::Aligned512) {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    (static_cast<uint64_t>(dTemplateType) >> 1),
                static_cast<uint64_t>(s2TemplateType) * (static_cast<uint64_t>(dTemplateType) >> 1)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, fp8_e4m3fn_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    } else {
        constexpr uint64_t qkvSizeRsv2 =
            MAX(MAX(static_cast<uint64_t>(s1TemplateType), static_cast<uint64_t>(s2TemplateType)) *
                    static_cast<uint64_t>(dTemplateType),
                static_cast<uint64_t>(s2TemplateType) * static_cast<uint64_t>(dTemplateType)) *
            2;
        INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(
            BaseApi::FlashAttentionNoQuantKernelInfer, vec1ResultSize, qkvSizeRsv2, bfloat16_t, float, fp8_e4m3fn_t,
            ImplModeEnum::AA_HIGH_PRECISION, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
            dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, true, isPa, isFd,
            enableKVPrefix, enableS1OutSplit);
    }
#endif
}

#endif // FIA_ARCH35_NOQUANT_H_
