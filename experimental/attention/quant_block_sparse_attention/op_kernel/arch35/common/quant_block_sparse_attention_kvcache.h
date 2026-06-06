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
#include "quant_block_sparse_attention_comm.h"
#define NUM_128 128
using namespace matmul;

TEMPLATE_INTF
__aicore__ inline void GetSparseParam(const ConstInfo &constInfo,
    const AttenMaskInfo &attenMaskInfo, RunParamStr &runParam)
{
    if constexpr (hasAtten) {
        runParam.preTokensPerBatch = attenMaskInfo.preTokens;
        runParam.nextTokensPerBatch = attenMaskInfo.nextTokens;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            runParam.preTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
            if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BNSD) {
                runParam.nextTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
                runParam.nextTokensOfMlaPerBatch = runParam.actualS2Size - runParam.actualSeqLengthOfMlaPerBatch;
            } else {
                runParam.nextTokensOfMlaPerBatch = runParam.actualS2Size - runParam.actualSeqLengthOfMlaPerBatch;
                runParam.nextTokensPerBatch = runParam.nextTokensOfMlaPerBatch * constInfo.gSize;
            }
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            runParam.preTokensPerBatch = attenMaskInfo.preTokens - runParam.actualS2Size +
                runParam.actualS1Size;
            runParam.nextTokensPerBatch = attenMaskInfo.nextTokens + runParam.actualS2Size -
                runParam.actualS1Size;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitQueryLeftPaddingSize(RunParamStr& runParam, const ConstInfo& constInfo, int64_t& actualS1Size)
{
    if (!constInfo.isQHasLeftPadding) {
        runParam.queryLeftPaddingSize = 0;
    } else {
        int64_t qLeftPaddingSize = 0;
        qLeftPaddingSize = constInfo.s1Size - actualS1Size - constInfo.queryRightPaddingSize;
        runParam.queryLeftPaddingSize = qLeftPaddingSize > 0 ? qLeftPaddingSize : 0;
        if (qLeftPaddingSize < 0) {
            actualS1Size = 0;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitKVLeftPaddingSize(RunParamStr& runParam, const ConstInfo& constInfo, int64_t& actualS2Size)
{
    if (!constInfo.isKVHasLeftPadding) {
        runParam.kvLeftPaddingSize = 0;
    } else {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualS2Size - constInfo.kvRightPaddingSize;
        runParam.kvLeftPaddingSize = kvLeftPaddingSize > 0 ? kvLeftPaddingSize : 0;
        if (kvLeftPaddingSize < 0) {
            actualS2Size = 0;
        }
    }
}

static constexpr uint32_t DIM_NUM2 = 2;
TEMPLATE_INTF
__aicore__ inline void GetKVSeqLengthForTensorList(RunParamStr& runParam,
    const ConstInfo& constInfo, int32_t bIdx, GlobalTensor<INPUT_T>& keyGm)
{
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void*)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, bIdx);
        // if constexpr (kvLayout == LayOutTypeEnum::LAYOUT_BNSD) {
        //     runParam.s2InCurrentBatch = kvTensorDesc.GetShape(DIM_NUM2);
        // } else {
        //     runParam.s2InCurrentBatch = kvTensorDesc.GetShape(1);
        // }
    }
}

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS1Size(RunParamStr& runParam,
    const ConstInfo& constInfo, int32_t bIdx,
    __gm__ int64_t* actualSeqQlenAddr, int64_t actualSeqMin)
{
    if (constInfo.isActualLenDimsNull) {
        int64_t actualMSize = constInfo.s1Size;
        return actualMSize;
    }
    
    int64_t actualMSize = 0;
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND || 
                         qLayout == LayOutTypeEnum::LAYOUT_NTD) {
        actualMSize = (bIdx == 0) ? actualSeqQlenAddr[0] :
            actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
    } else {
        actualMSize = (constInfo.actualSeqLenSize == actualSeqMin) ? 
            actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx];
    }
    
    return actualMSize;
}

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS2Size(RunParamStr& runParam,
    const ConstInfo& constInfo, int32_t bIdx,
    __gm__ int64_t* actualSeqKvlenAddr, int64_t actualSeqKVMin)
{
    int64_t actualS2Size = 0;
    
    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = (constInfo.isKvContinuous == 1) ? constInfo.s2Size :
            runParam.s2InCurrentBatch;
    } else {
        // if constexpr (kvLayout == LayOutTypeEnum::LAYOUT_TND || kvLayout == LayOutTypeEnum::LAYOUT_NTD) {
        //     actualS2Size = (isPa && constInfo.actualSeqLenKVSize == actualSeqKVMin) ? 
        //         actualSeqKvlenAddr[0] : actualSeqKvlenAddr[bIdx];
        //     if ((bIdx > 0) && (!isPa)) {
        //         actualS2Size -= actualSeqKvlenAddr[bIdx - 1];
        //     }
        // } else {
        //     actualS2Size = (constInfo.actualSeqLenKVSize == actualSeqKVMin) ? 
        //         actualSeqKvlenAddr[0] : actualSeqKvlenAddr[bIdx];
        // }
    }
    
    return actualS2Size;
}

TEMPLATE_INTF
__aicore__ inline void AdjustActualS1Size(RunParamStr& runParam,
    const ConstInfo& constInfo)
{
    if (!constInfo.isGqa) {
        runParam.actualS1Size = (runParam.actualS1Size >
            runParam.actualS2Size + runParam.preTokensPerBatch) ?
            runParam.actualS2Size + runParam.preTokensPerBatch :
            runParam.actualS1Size;
        // 计算S1的尾块大小，非对齐
        if (runParam.nextTokensPerBatch < 0) {
            runParam.actualS1Size = runParam.actualS1Size + runParam.nextTokensPerBatch;
        }
    }

    if (runParam.actualS1Size < 0) { // 修正preToken/nextToken导致全无效场景的qs值
        runParam.actualS1Size = 0;
    }
}

TEMPLATE_INTF
__aicore__ inline void CalculateQueryOffset(RunParamStr& runParam,
    const ConstInfo& constInfo, int32_t bIdx,
    __gm__ int64_t* actualSeqQlenAddr)
{
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH) {
        runParam.qBOffset = bIdx * constInfo.s1Size * constInfo.n2GD + 
                            runParam.queryLeftPaddingSize * constInfo.n2GD;
    } else if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND) {
        runParam.qBOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1] * constInfo.n2GD;
    } else {
        runParam.qBOffset = bIdx * constInfo.s1Size * constInfo.n2GD + 
                            runParam.queryLeftPaddingSize * constInfo.dSize;
    }
}


TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr& runParam, 
    const ConstInfo &constInfo, const AttenMaskInfo &attenMaskInfo, int32_t bIdx,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t * actualSeqKvlenAddr)
{
    constexpr int64_t actualSeqMin = 1;
    constexpr int64_t actualSeqKVMin = 1;
    
    // TensorList场景获取不同batch的KvSeq长度
    GetKVSeqLengthForTensorList<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, keyGm);

    // 计算实际序列长度
    int64_t actualS1Size = CalculateActualS1Size<TEMPLATE_INTF_ARGS>(
        runParam, constInfo, bIdx, actualSeqQlenAddr, actualSeqMin);
    int64_t actualS2Size = CalculateActualS2Size<TEMPLATE_INTF_ARGS>(
        runParam, constInfo, bIdx, actualSeqKvlenAddr, actualSeqKVMin);

    // 初始化padding大小
    InitQueryLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS1Size);
    InitKVLeftPaddingSize<TEMPLATE_INTF_ARGS>(runParam, constInfo, actualS2Size);

    runParam.actualS1Size = actualS1Size;
    runParam.actualS2Size = actualS2Size;
    GetSparseParam<TEMPLATE_INTF_ARGS>(constInfo, attenMaskInfo, runParam);

    // 调整S1大小以适应per token、next token设置
    AdjustActualS1Size<TEMPLATE_INTF_ARGS>(runParam, constInfo);

    // 计算query偏移量
    CalculateQueryOffset<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqQlenAddr);
    
    // 推理的TND场景的mask都是padding过的
    runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
    // 推理的mask的sequence length可能大于qk的sequence length
    runParam.b1SSAttenMaskOffset = runParam.boIdx * (uint64_t)attenMaskInfo.attenMaskS1Size *
        (uint64_t)attenMaskInfo.attenMaskS2Size;
}

TEMPLATE_INTF
__aicore__ inline void GetKeyCoreOffsetParam(RunParamStr& runParam,
    const ConstInfo &constInfo, int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t keyInnerOffsetSize = 0;
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH) {
        if (constInfo.isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            keyInnerOffsetSize = bIdx * constInfo.n2S2D + runParam.kvLeftPaddingSize * constInfo.n2D;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;

    } else if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            keyInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2D;
        } else {
            keyInnerOffsetSize = bIdx * constInfo.n2S2D;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * constInfo.dSize;
    } else {
        uint64_t headStrideK = 0;
        if (constInfo.isKvContinuous == 1) {
            headStrideK = constInfo.s2D;
            keyInnerOffsetSize = bIdx * constInfo.n2Size * headStrideK +
                runParam.kvLeftPaddingSize * constInfo.dSize;
        } else {
            headStrideK = constInfo.dSize * runParam.s2InCurrentBatch;
            keyInnerOffsetSize = 0;
        }
        runParam.keyCoreOffset = keyInnerOffsetSize + runParam.n2oIdx * headStrideK;
    }
}

TEMPLATE_INTF
__aicore__ inline void GetValueCoreOffsetParam(RunParamStr& runParam, const ConstInfo &constInfo, 
    int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH) {
        if (constInfo.isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            valueInnerOffsetSize = bIdx * constInfo.n2S2Dv + runParam.kvLeftPaddingSize * constInfo.n2Dv;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            valueInnerOffsetSize = 0;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
    } else if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!isPa) {
            valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2Dv;
        } else {
            valueInnerOffsetSize = bIdx * constInfo.n2S2Dv;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
    } else if constexpr (qLayout == LayOutTypeEnum::LAYOUT_NTD) {
        uint64_t actualSeqKVLen = 0;
        if constexpr (isPa) {
            actualSeqKVLen = constInfo.s2Size;
            valueInnerOffsetSize = bIdx * constInfo.s2Dv;
        } else {
            actualSeqKVLen = (bIdx == 0) ? actualSeqKvlenAddr[0] : actualSeqKvlenAddr[bIdx] - actualSeqKvlenAddr[bIdx - 1];
            valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.dSizeV;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.bSize * actualSeqKVLen * constInfo.dSizeV;
    } else {
        uint64_t headStrideV = 0;
        if (constInfo.isKvContinuous == 1) {
            headStrideV = constInfo.s2Dv;
            valueInnerOffsetSize = bIdx * constInfo.n2Size * headStrideV +
                runParam.kvLeftPaddingSize * constInfo.dSizeV;
        } else {
            headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
            valueInnerOffsetSize = 0;
        }
        runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
    }
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& runParam, const ConstInfo &constInfo,
    const AttenMaskInfo &attenMaskInfo,
    GlobalTensor<INPUT_T>& keyGm, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, attenMaskInfo,
        runParam.boIdx, keyGm, actualSeqQlenAddr, actualSeqKvlenAddr);
    GetValueCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqKvlenAddr);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo,
    bool lastBN, int64_t nextGs1Idx)
{
    constexpr int32_t s1BaseSize = static_cast<int32_t>(s1TemplateType);
    int32_t s1LoopTimes;
    if (constInfo.isGqa) {
        s1LoopTimes = CeilDiv(runParam.actualS1Size * constInfo.gSize, s1BaseSize);
    } else {
        s1LoopTimes = CeilDiv(runParam.actualS1Size, s1BaseSize);
    }
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.s1LoopTimes = s1LoopTimes;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.s1LoopTimes = nextGs1Idx == 0 ? s1LoopTimes : nextGs1Idx;
    }
    if (constInfo.isGqa && constInfo.gSize > NUM_128 && qLayout == LayOutTypeEnum::LAYOUT_TND) {
        runParam.s1LoopTimes = s1LoopTimes;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * (uint32_t)s1TemplateType;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
    } else {
        if (constInfo.isGqa) {
            runParam.s1RealSize = Min((uint32_t)s1TemplateType, runParam.actualS1Size * constInfo.gSize - cubeSOuterOffset);
        } else {
            runParam.s1RealSize = Min((uint32_t)s1TemplateType, runParam.actualS1Size - cubeSOuterOffset);
        }
    }

    cubeSOuterOffset += (runParam.nextTokensPerBatch < 0) ? -runParam.nextTokensPerBatch : 0;
   
    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    if (constInfo.s1Size > 1 && constInfo.isGqa && (qLayout == LayOutTypeEnum::LAYOUT_BSH ||
        qLayout == LayOutTypeEnum::LAYOUT_TND)) {
        runParam.halfS1RealSize = (runParam.halfS1RealSize + constInfo.gSize - 1) / constInfo.gSize *
            constInfo.gSize;
    }
    
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
    if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND || qLayout == LayOutTypeEnum::LAYOUT_NTD) {
        actualSeqLen = (bIdx == 0) ? actualSeqQlenAddr[0] : actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
        seqOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1];
    } else {
        actualSeqLen = constInfo.s1Size;
        seqOffset = bIdx * constInfo.s1Size;
    }
    if ASCEND_IS_AIC {
        if (constInfo.isGqa && constInfo.s1Size > 1) { // 合轴：antiquant PFA
            if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH || qLayout == LayOutTypeEnum::LAYOUT_TND) {
                runParam.tensorQOffset = runParam.qBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                    runParam.n2oIdx * constInfo.gD;
            } else {
                runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
                    runParam.cubeSOuterOffset * constInfo.dSize;
            }
        } else if (constInfo.isGqa) {
            runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gD * actualSeqLen +
                runParam.cubeSOuterOffset * constInfo.dSize;
        } else {
            if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH || qLayout == LayOutTypeEnum::LAYOUT_TND) {
                runParam.tensorQOffset = runParam.qBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                    runParam.n2oIdx * constInfo.gD + runParam.goIdx * constInfo.dSize;
            } else {
                runParam.tensorQOffset = runParam.qBOffset + runParam.n2oIdx * constInfo.gS1D +
                    runParam.goIdx * constInfo.s1D + runParam.cubeSOuterOffset * constInfo.dSize;
            }
        }
    } else {
        int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
    
        if (constInfo.isGqa && constInfo.s1Size > 1) { // PFA
            if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH || qLayout == LayOutTypeEnum::LAYOUT_TND) {
                runParam.attentionOutOffset = attentionOutSeqOffset + runParam.queryLeftPaddingSize * constInfo.n2GDv +
                    runParam.sOuterOffset / constInfo.gSize * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv;
            } else {
                runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
                    runParam.sOuterOffset * constInfo.dSizeV;
            }
        } else if (constInfo.isGqa) { // IFA
                runParam.attentionOutOffset = attentionOutSeqOffset + runParam.n2oIdx * constInfo.gDv * actualSeqLen +
                    runParam.sOuterOffset * constInfo.dSizeV;
        } else {
            runParam.attentionOutOffset = attentionOutSeqOffset + runParam.queryLeftPaddingSize * constInfo.n2GDv +
                runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                runParam.goIdx * constInfo.dSizeV;
        }

        int64_t softmaxLseSeqOffset = seqOffset * constInfo.n2G;
    
        if (constInfo.isGqa && constInfo.s1Size > 1) {
            if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH || qLayout == LayOutTypeEnum::LAYOUT_TND) {
                runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.queryLeftPaddingSize * constInfo.n2G +
                    runParam.sOuterOffset / constInfo.gSize * constInfo.n2G + runParam.n2oIdx * constInfo.gSize;
            } else {
                runParam.softmaxLseOffset = softmaxLseSeqOffset +
                    runParam.n2oIdx * constInfo.gSize * actualSeqLen + runParam.sOuterOffset;
            }
        } else if (constInfo.isGqa) {
            runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gSize * actualSeqLen +
                runParam.sOuterOffset;
        } else {
            if constexpr (qLayout == LayOutTypeEnum::LAYOUT_TND || qLayout == LayOutTypeEnum::LAYOUT_NTD) {
                runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.sOuterOffset * constInfo.n2G +
                    runParam.n2oIdx * constInfo.gSize + runParam.goIdx;
            } else {
                runParam.softmaxLseOffset = softmaxLseSeqOffset + runParam.n2oIdx * constInfo.gS1 +
                    runParam.goIdx * constInfo.s1Size + runParam.sOuterOffset + runParam.queryLeftPaddingSize;
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int64_t *actualSeqQlenAddr)
{
    // 后续的函数依赖 sOuterOffset
    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);

    // 使用转换后的左上角的pretoken nexttoken
    if (runParam.s1RealSize == 0 || (runParam.nextTokensPerBatch < 0 && runParam.sOuterOffset < ((runParam.nextTokensPerBatch * (-1)) /
        runParam.halfS1RealSize * runParam.halfS1RealSize))) {
        return true;
    }

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, actualSeqQlenAddr);
    return false;
}

TEMPLATE_INTF
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo)
{
    constexpr int32_t s2BaseSize = static_cast<int32_t>(s2TemplateType);

    int64_t sInnerFirstToken = 0;

    if (constInfo.isGqa) {
        sInnerFirstToken = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.s1oIdx - runParam.preTokensPerBatch,
            0, runParam.actualS2Size);
        runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.s1oIdx + runParam.nextTokensPerBatch +
            runParam.s1RealSize, 0, runParam.actualS2Size);
    } else {
        sInnerFirstToken = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset - runParam.preTokensPerBatch,
            0, runParam.actualS2Size);
        runParam.s2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(runParam.cubeSOuterOffset + runParam.nextTokensPerBatch +
            runParam.s1RealSize, 0, runParam.actualS2Size);
    }
    runParam.s2LoopEndIdx = (runParam.s2LineEndIdx + s2BaseSize - 1) / s2BaseSize - sInnerFirstToken / s2BaseSize;
    if (runParam.s2LoopEndIdx <= 0) {
        return true;
    }
    if constexpr (hasAtten) {
        runParam.s2LineStartIdx = sInnerFirstToken / s2BaseSize * s2BaseSize;
    } else {
        runParam.s2LineStartIdx = sInnerFirstToken;
    }
    return false;
}

TEMPLATE_INTF
__aicore__ inline void ComputeOffset(const RunParamStr& runParam,
    const ConstInfo &constInfo, uint32_t sInnerLoopIdx, RunInfo &runInfo)
{
    if ASCEND_IS_AIV {
        if (!constInfo.isGqa) {
            runInfo.vecCoreOffset = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
        } else {
            runInfo.vecCoreOffset = 0;
        }
    } else {
        if constexpr (qLayout == LayOutTypeEnum::LAYOUT_BSH || qLayout == LayOutTypeEnum::LAYOUT_TND) {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
                runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2D;
            } else {
                runInfo.keyOffset = runInfo.valueOffset;
            }
        } else if constexpr (qLayout == LayOutTypeEnum::LAYOUT_NTD) {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
                runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            } else {
                runInfo.keyOffset = runInfo.valueOffset;
            }

        } else {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
                runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseD;
            } else {
                runInfo.keyOffset = runInfo.valueOffset;
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr& runParam, RunInfo &runInfo)
{
    runInfo.keyOffset = runParam.keyOffset;
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatch = runParam.preTokensPerBatch;
    runInfo.nextTokensPerBatch = runParam.nextTokensPerBatch;
    runInfo.b1SSAttenMaskOffset = runParam.b1SSAttenMaskOffset;
    runInfo.b1SSOffset = runParam.b1SSOffset;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
    runInfo.s2InCurrentBatch = runParam.s2InCurrentBatch;
    runInfo.queryLeftPaddingSize = runParam.queryLeftPaddingSize;
    runInfo.kvLeftPaddingSize = runParam.kvLeftPaddingSize;
}

#endif  // INFER_FLASH_ATTENTION_KVCACHE_H