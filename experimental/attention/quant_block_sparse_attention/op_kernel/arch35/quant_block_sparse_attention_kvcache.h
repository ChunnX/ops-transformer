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
 * \file quant_block_sparse_attention_kvcache.h
 * \brief
 */
#ifndef INFER_FLASH_ATTENTION_KVCACHE_H
#define INFER_FLASH_ATTENTION_KVCACHE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "quant_block_sparse_attention_common_arch35.h"
#include "quant_block_sparse_attention_attenmask.h"
using namespace matmul;

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS1Size(RunParamStr& runParam,
    const ConstInfo& constInfo, int32_t bIdx, __gm__ int64_t* actualSeqQlenAddr)
{
    int64_t actualS1Size = 0;
    if constexpr (layout == BSALayout::TND) {
        if (constInfo.isActualLenDimsNull) {
            actualS1Size = constInfo.s1Size;
        } else {
            actualS1Size = (bIdx == 0) ? actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
        }
    }
    
    return actualS1Size;
}

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS2Size(RunParamStr& runParam,
    const ConstInfo& constInfo, int32_t bIdx, __gm__ int64_t* actualSeqKvlenAddr)
{
    int64_t actualS2Size = 0;
    if constexpr (layout == BSALayout::TND) {
        if (constInfo.isActualLenDimsNull) {
            actualS2Size = constInfo.s2Size;
        } else {
            actualS2Size = (bIdx == 0) ? actualSeqKvlenAddr[0] : actualSeqKvlenAddr[bIdx] - actualSeqKvlenAddr[bIdx - 1];
        }
    }
    return actualS2Size;
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& runParam, const ConstInfo &constInfo,
    const AttenMaskInfo &attenMaskInfo,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t *actualSeqKvlenAddr)
{
    // 计算实际序列长度
    runParam.actualS1Size = CalculateActualS1Size<TEMPLATE_INTF_ARGS>(
        runParam, constInfo, runParam.boIdx, actualSeqQlenAddr);
    runParam.actualS2Size = CalculateActualS2Size<TEMPLATE_INTF_ARGS>(
        runParam, constInfo, runParam.boIdx, actualSeqKvlenAddr);

    if constexpr (hasAtten) {
        runParam.preTokensPerBatch = attenMaskInfo.preTokens;
        runParam.nextTokensPerBatch = attenMaskInfo.nextTokens;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            runParam.preTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
            if constexpr (layout == BSALayout::TND) {
                runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;
            }
        }
    }
    // 计算query偏移量
    if constexpr (layout == BSALayout::TND) {
        runParam.qBOffset = (runParam.boIdx == 0) ? 0 : actualSeqQlenAddr[runParam.boIdx - 1] * constInfo.n2GD;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t s1StartIdx, uint32_t s1EndIdx, bool firstBN, bool lastBN)
{
    uint32_t s1OuterSize = CeilDiv(runParam.actualS1Size, constInfo.s1BaseSize);
    runParam.s1LoopStart = firstBN ? s1StartIdx : 0;
    runParam.s1LoopEnd = lastBN ? s1EndIdx : s1OuterSize;

    if constexpr (hasAtten) {
        if (runParam.nextTokensPerBatch < 0) {
            uint32_t nextTokensOuterSize = (-runParam.nextTokensPerBatch) / constInfo.s1BaseSize;       // 向下取整
            runParam.s1LoopStart = Max(runParam.s1LoopStart, nextTokensOuterSize);
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr& runParam, const ConstInfo &constInfo, uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * constInfo.s1BaseSize;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
    } else {
        runParam.s1RealSize = Min(constInfo.s1BaseSize, runParam.actualS1Size - cubeSOuterOffset);
    }
   
    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
        runParam.sOuterOffset = cubeSOuterOffset + runParam.firstHalfS1RealSize;
    } else {
        runParam.sOuterOffset = cubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr& runParam, const ConstInfo &constInfo,
    int32_t bIdx, __gm__ int64_t *actualSeqQlenAddr)
{
    int64_t actualSeqLen = 0;
    int64_t seqOffset = 0;
    if constexpr (layout == BSALayout::TND) {
        actualSeqLen = (bIdx == 0) ? actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
        seqOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1];
    }
    if ASCEND_IS_AIC {
        if constexpr (layout == BSALayout::TND) {
            runParam.tensorQOffset = runParam.qBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                runParam.n2oIdx * constInfo.gD + runParam.goIdx * constInfo.dSize;
        }
    } else {
        if constexpr (layout == BSALayout::TND) {
            // 起始地址  需要stride偏移
            runParam.attentionOutOffset = seqOffset * constInfo.n2GDv +
                runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                runParam.goIdx * constInfo.dSizeV;
        }
        if constexpr (layout == BSALayout::TND) {
            runParam.softmaxLseOffset = seqOffset * constInfo.n2G + 
                                runParam.sOuterOffset * constInfo.n2G +
                                runParam.n2oIdx * constInfo.gSize + runParam.goIdx;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamS1(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int64_t *actualSeqQlenAddr)
{
    // 后续的函数依赖 sOuterOffset
    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);
    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqQlenAddr);
}

TEMPLATE_INTF
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo)
{
    
    // sInnerFirstToken在nomask和maskmode=3场景下均为0
    runParam.s2LineStartIdx = 0;
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr& runParam, RunInfo &runInfo)
{
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatch = runParam.preTokensPerBatch;
    runInfo.nextTokensPerBatch = runParam.nextTokensPerBatch;
    runInfo.b1SSAttenMaskOffset = runParam.b1SSAttenMaskOffset;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
}

__aicore__ inline void UpdateLoopFlagsBasedOnS1Index(
    int32_t extras1, 
    bool& notLast,
    bool& notLastTwoLoop,
    bool& notLastThreeLoop
)
{
    switch (extras1) {
        case -1:
            break;
        case 0:
            notLastThreeLoop = false;
            break;
        case 1:
            notLastThreeLoop = false;
            notLastTwoLoop = false;
            break;
        case 2:
            notLast = false;
            notLastThreeLoop = false;
            notLastTwoLoop = false;
            break;
        default:
            break;
    }
}


#endif  // INFER_FLASH_ATTENTION_KVCACHE_H