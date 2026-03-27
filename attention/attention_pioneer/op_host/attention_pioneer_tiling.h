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
 * \file attention_pioneer_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ATTENTIONPIONEER_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ATTENTIONPIONEER_H_
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "attention_pioneer_tiling_compile_info.h"
#include "attention_pioneer_tiling_index.h"

#ifdef ASCENDC_OP_TEST
#define AP_EXTERN_C extern "C"
#else
#define AP_EXTERN_C
#endif

namespace optiling {
const uint32_t FIA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
// 基础参数
BEGIN_TILING_DATA_DEF(AttentionPioneerBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, headDimRope)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS1Dims)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS2Dims)
TILING_DATA_FIELD_DEF(uint32_t, accumQSeqFlag)
TILING_DATA_FIELD_DEF(uint32_t, accumKVSeqFlag)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, batchContinuous)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, needInit)
TILING_DATA_FIELD_DEF(uint32_t, slidingFlag)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
TILING_DATA_FIELD_DEF(uint32_t, isLegacyIfa)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerBaseParamsOp, AttentionPioneerBaseParams)

// PageAttention 参数
BEGIN_TILING_DATA_DEF(AttentionPioneerPageAttentionParams)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerPageAttentionParamsOp, AttentionPioneerPageAttentionParams)

// AttenMask 参数
BEGIN_TILING_DATA_DEF(AttentionPioneerMaskParams)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatchStride)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskStride)
TILING_DATA_FIELD_DEF(int32_t, preToken)
TILING_DATA_FIELD_DEF(int32_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, isExistRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerMaskParamsOp, AttentionPioneerMaskParams)

// 内切基本块参数
BEGIN_TILING_DATA_DEF(AttentionPioneerInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerInnerSplitParamsOp, AttentionPioneerInnerSplitParams)

// workspace参数
BEGIN_TILING_DATA_DEF(AttentionPioneerWorkspaceParams)
TILING_DATA_FIELD_DEF(uint32_t, mm1ResSize)
TILING_DATA_FIELD_DEF(uint32_t, mm2ResSize)
TILING_DATA_FIELD_DEF(uint32_t, fdAccumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, fdLogSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerWorkspaceParamsOp, AttentionPioneerWorkspaceParams)

// 外切分核参数
BEGIN_TILING_DATA_DEF(AttentionPioneerOuterSplitParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2End)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerOuterSplitParamsOp, AttentionPioneerOuterSplitParams)

// FlashDecode规约参数
BEGIN_TILING_DATA_DEF(AttentionPioneerFlashDecodeParams)
TILING_DATA_FIELD_DEF(uint32_t, numOfFdHead)
TILING_DATA_FIELD_DEF(uint32_t, reserved)
TILING_DATA_FIELD_DEF(uint32_t, gS1BaseSizeOfFd)                                    // FD负载均衡中，每个FD任务按gS1切分的基本size
TILING_DATA_FIELD_DEF(uint32_t, usedVecNumOfFd)                                     // FD负载均衡中，用到的vector数
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitNumOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitStartIdxOfCore)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1SplitNumOfFdHead)          // FD负载均衡中，每个FD任务按gS1基本size切分后的份数
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1LastPartSizeOfFdHead)      // FD负载均衡中，每个FD任务按gS1基本size切分后，最后一份的gS1大小，即尾块大小
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHead)        // FD负载均衡中，每个vector核处理的最后一个FD任务的序号
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHeadSplit)   // FD负载均衡中，每个vector核处理的最后一个FD任务的子划分的序号
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerFlashDecodeParamsOp, AttentionPioneerFlashDecodeParams)

// 公共前缀
BEGIN_TILING_DATA_DEF(AttentionPioneerPrefixParams)
TILING_DATA_FIELD_DEF(uint64_t, prefixMaxLen)
TILING_DATA_FIELD_DEF(uint64_t, prefixLen)
TILING_DATA_FIELD_DEF(bool, prefixFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerPrefixParamsOp, AttentionPioneerPrefixParams)
// Pse 注册参数
BEGIN_TILING_DATA_DEF(AttentionPioneerPseParams)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftByBatch)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerPseParamsOp, AttentionPioneerPseParams)

// Left Padding 参数
BEGIN_TILING_DATA_DEF(AttentionPioneerLeftPaddingParams)
TILING_DATA_FIELD_DEF(uint32_t, qPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerLeftPaddingParamsOp, AttentionPioneerLeftPaddingParams)
// 后量化 参数
BEGIN_TILING_DATA_DEF(AttentionPioneerPostQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerPostQuantParamsOp, AttentionPioneerPostQuantParams)

//MLA非量化模板TilingData
BEGIN_TILING_DATA_DEF(AttentionPioneerTilingData)
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerPageAttentionParams, pageAttenParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerMaskParams, maskParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerWorkspaceParams, workspaceParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerOuterSplitParams, outerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerFlashDecodeParams, fdParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerPrefixParams, prefixParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerPseParams, pseParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerLeftPaddingParams, leftPaddingParams);
TILING_DATA_FIELD_DEF_STRUCT(AttentionPioneerPostQuantParams, postquantParams);
END_TILING_DATA_DEF

// empty tenmsor 模板TilingData
BEGIN_TILING_DATA_DEF(AttentionPioneerEmptyTensorTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalOutputSize)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreSize)
TILING_DATA_FIELD_DEF(uint64_t, totalLseSize)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreLseSize)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
END_TILING_DATA_DEF

// 全量化 参数 当前无
BEGIN_TILING_DATA_DEF(AttentionPioneerFullQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, placeHolder)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerFullQuantParamsOp, AttentionPioneerFullQuantParams)

// L2 Cache 参数
BEGIN_TILING_DATA_DEF(AttentionPioneerL2CacheParams)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerL2CacheParamsOp, AttentionPioneerL2CacheParams)

// MSD 参数
BEGIN_TILING_DATA_DEF(AttentionPioneerMsdParams)
TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerMsdParamsOp, AttentionPioneerMsdParams)

// 伪量化 参数
BEGIN_TILING_DATA_DEF(AttentionPioneerAntiqParams)
TILING_DATA_FIELD_DEF(uint32_t, antiqSeqSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(AttentionPioneerAntiqParamsOp, AttentionPioneerAntiqParams)

extern "C" {
ge::graphStatus DeviceDoOpTilingAttentionPioneer(gert::TilingContext *context);
}
AP_EXTERN_C ge::graphStatus DoOpTilingAttentionPioneer(gert::TilingContext *context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_ATTENTIONPIONEER_H_
