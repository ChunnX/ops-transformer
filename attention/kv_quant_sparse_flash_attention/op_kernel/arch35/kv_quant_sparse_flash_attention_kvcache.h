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
 * \file kv_quant_sparse_flash_attention_kvcache.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_KVCACHE_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kv_quant_sparse_flash_attention_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;

static constexpr uint32_t sparseModeZero = 0;
static constexpr uint32_t sparseModeThree = 3;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr& qsfaRunParam, const ConstInfo &qsfaConstInfo,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t * actualSeqKvlenAddr)
{
    int32_t actualS1Size = 0;
    int32_t actualS2Size = 0;
    int32_t actualSeqMin = 1;
    int32_t actualSeqKVMin = 1;
    int32_t sIdx = qsfaRunParam.qsfaBoIdx;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        // actual seq length first
        if (actualSeqQlenAddr != nullptr) {
            actualS1Size = (sIdx == 0) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1];
        } else {
            actualS1Size = qsfaConstInfo.qsfaS1Size;
        }
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? qsfaConstInfo.qsfaS1Size :
            actualSeqQlenAddr[sIdx];
    }

    if (qsfaConstInfo.qsfaIsActualLenDimsKVNull) {
        actualS2Size = qsfaConstInfo.qsfaS2Size;
    } else {
        if constexpr (isPa) {
            if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
                actualS2Size = actualSeqKvlenAddr[sIdx];
            } else {
                actualS2Size = (qsfaConstInfo.qsfaActualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        } else {
            if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
                actualS2Size = (sIdx == 0) ? actualSeqKvlenAddr[0] :
                    actualSeqKvlenAddr[sIdx] - actualSeqKvlenAddr[sIdx - 1];
            } else {
                actualS2Size = (qsfaConstInfo.qsfaActualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        }
    }

    qsfaRunParam.qsfaActualS1Size = actualS1Size;
    qsfaRunParam.qsfaActualS2Size = actualS2Size;
    if (qsfaConstInfo.qsfaSparseMode == sparseModeZero) {
        qsfaRunParam.qsfaNextTokensPerBatch = MAX_PRE_NEXT_TOKENS;
    } else {
        qsfaRunParam.qsfaNextTokensPerBatch = qsfaRunParam.qsfaActualS2Size - qsfaRunParam.qsfaActualS1Size;
    }
    qsfaRunParam.qsfaPreTokensPerBatch = qsfaRunParam.qsfaActualS1Size;
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& qsfaRunParam, const ConstInfo &qsfaConstInfo,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(qsfaRunParam, qsfaConstInfo, actualSeqQlenAddr, actualSeqKvlenAddr);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& qsfaRunParam, const ConstInfo &qsfaConstInfo, bool lastBN,
    int64_t nextGs1Idx, int64_t gS1StartIdx)
{
    qsfaRunParam.qSNumInOneBlock = 1; // 不切G轴, 计算每个基本块可以拷贝多少行s
    qsfaRunParam.gs1LoopStartIdx = gS1StartIdx;
    if (qsfaRunParam.qsfaNextTokensPerBatch < 0) {
        uint64_t invalidTokenCount = static_cast<uint64_t>(-(qsfaRunParam.qsfaNextTokensPerBatch + 1)) + 1ULL;
        int64_t gs1LoopStartIdx =
            invalidTokenCount / qsfaRunParam.qSNumInOneBlock * qsfaRunParam.qSNumInOneBlock;
        if (gs1LoopStartIdx > gS1StartIdx) {
            qsfaRunParam.gs1LoopStartIdx = gs1LoopStartIdx;
        }
    }

    int32_t gs1LoopEndIdx = qsfaRunParam.qsfaActualS1Size; // 对于QSFA, 不切G轴, 每次拷贝一行的topk，只算一行的qs

    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        qsfaRunParam.gs1LoopEndIdx = gs1LoopEndIdx;
    } else { // 最后一个bn, 从数组下一个元素取值
        qsfaRunParam.gs1LoopEndIdx = nextGs1Idx == 0 ? gs1LoopEndIdx : nextGs1Idx;
    }

    if (qsfaRunParam.gs1LoopStartIdx > qsfaRunParam.gs1LoopEndIdx) {
        qsfaRunParam.gs1LoopStartIdx = qsfaRunParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr& qsfaRunParam, const ConstInfo &qsfaConstInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t qsfaCubeSOuterOffset = sOuterLoopIdx * qsfaRunParam.qSNumInOneBlock;
    if (qsfaRunParam.qsfaActualS1Size == 0) {
        qsfaRunParam.qsfaS1RealSize = 0;
        qsfaRunParam.qsfaMRealSize = 0;
    } else {
        qsfaRunParam.qsfaS1RealSize = Min(qsfaRunParam.qSNumInOneBlock, qsfaRunParam.qsfaActualS1Size - qsfaCubeSOuterOffset);
        qsfaRunParam.qsfaMRealSize = qsfaRunParam.qsfaS1RealSize * qsfaConstInfo.qsfaGSize;
        if constexpr (IS_SPLIT_G) {
            qsfaRunParam.qsfaMRealSize = qsfaRunParam.qsfaMRealSize >> 1;
        }
    }

    qsfaRunParam.cubeMOuterOffset = qsfaCubeSOuterOffset * qsfaConstInfo.qsfaGSize;
    qsfaRunParam.qsfaHalfMRealSize = (qsfaRunParam.qsfaMRealSize + 1) >> 1;
    qsfaRunParam.qsfaFirstHalfMRealSize = qsfaRunParam.qsfaHalfMRealSize;
    if (qsfaConstInfo.qsfaSubBlockIdx == 1) {
        qsfaRunParam.qsfaHalfMRealSize = qsfaRunParam.qsfaMRealSize - qsfaRunParam.qsfaHalfMRealSize;
        qsfaRunParam.qsfaMOuterOffset = qsfaRunParam.cubeMOuterOffset + qsfaRunParam.qsfaFirstHalfMRealSize;
    } else {
        qsfaRunParam.qsfaMOuterOffset = qsfaRunParam.cubeMOuterOffset;
    }

    qsfaRunParam.qsfaHalfS1RealSize = (qsfaRunParam.qsfaS1RealSize + 1) >> 1;
    qsfaRunParam.qsfaFirstHalfS1RealSize = qsfaRunParam.qsfaHalfS1RealSize;
    if (qsfaConstInfo.qsfaSubBlockIdx == 1) {
        qsfaRunParam.qsfaHalfS1RealSize = qsfaRunParam.qsfaS1RealSize - qsfaRunParam.qsfaHalfS1RealSize;
        qsfaRunParam.qsfaSOuterOffset = qsfaCubeSOuterOffset + qsfaRunParam.qsfaHalfMRealSize / qsfaConstInfo.qsfaGSize;
    } else {
        qsfaRunParam.qsfaSOuterOffset = qsfaCubeSOuterOffset;
    }
    qsfaRunParam.qsfaCubeSOuterOffset = qsfaCubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr& qsfaRunParam, const ConstInfo &qsfaConstInfo,
    int32_t sIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            seqOffset = sIdx == 0 ? 0 : cuSeqlensQAddr[sIdx - 1];
        } else {
            seqOffset = sIdx * qsfaConstInfo.qsfaS1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * qsfaConstInfo.qsfaN2GDv;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND || LAYOUT_T == QSFA_LAYOUT::TND) {
            qsfaRunParam.qsfaAttentionOutOffset = attentionOutSeqOffset +
                qsfaRunParam.qsfaSOuterOffset * qsfaConstInfo.qsfaN2GDv + qsfaRunParam.qsfaN2oIdx * qsfaConstInfo.qsfaGDv +
                qsfaRunParam.qsfaGoIdx * qsfaConstInfo.qsfaDSizeV;
        }
        if (qsfaConstInfo.qsfaSubBlockIdx == 1) {
            qsfaRunParam.qsfaAttentionOutOffset += qsfaRunParam.qsfaFirstHalfMRealSize * qsfaConstInfo.qsfaDSizeV;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr& qsfaRunParam, const ConstInfo &qsfaConstInfo,
    uint32_t sOuterLoopIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if (qsfaRunParam.qsfaNextTokensPerBatch < 0) {
        uint64_t invalidTokenCount = static_cast<uint64_t>(-(qsfaRunParam.qsfaNextTokensPerBatch + 1)) + 1ULL;
        if (qsfaRunParam.qsfaS1oIdx <
            invalidTokenCount / qsfaRunParam.qSNumInOneBlock * qsfaRunParam.qSNumInOneBlock) {
            return true;
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(qsfaRunParam, qsfaConstInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(qsfaRunParam, qsfaConstInfo, qsfaRunParam.qsfaBoIdx, cuSeqlensQAddr);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr& qsfaRunParam, __gm__ int32_t *cuSeqlensQAddr)
{
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (qsfaRunParam.qsfaBoIdx > 0 && ((qsfaRunParam.qsfaBoIdx == 0 && cuSeqlensQAddr[qsfaRunParam.qsfaBoIdx] == 0) || (cuSeqlensQAddr[qsfaRunParam.qsfaBoIdx] - cuSeqlensQAddr[qsfaRunParam.qsfaBoIdx - 1] == 0))) {
            return true;
        }
    }
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
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr& qsfaRunParam, const ConstInfo &qsfaConstInfo)
{
    if (qsfaRunParam.qsfaActualS2Size == 0) {
        qsfaRunParam.kvLoopEndIdx = 0;
        qsfaRunParam.qsfaS2LoopEndIdx = 0;
        return true;
    }
    uint32_t qsfaS2BaseSize = qsfaConstInfo.qsfaS2BaseSize;

    if (qsfaConstInfo.qsfaSparseMode == sparseModeZero) {
        qsfaRunParam.qsfaS2LineStartIdx = 0;
        qsfaRunParam.qsfaS2LineEndIdx = Min(qsfaRunParam.qsfaActualS2Size, qsfaConstInfo.qsfaSparseBlockCount);
    } else if (qsfaConstInfo.qsfaSparseMode == sparseModeThree) {
        qsfaRunParam.qsfaS2LineStartIdx =
            ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(qsfaRunParam.qsfaCubeSOuterOffset - qsfaRunParam.qsfaPreTokensPerBatch,
            0, qsfaRunParam.qsfaActualS2Size);
        qsfaRunParam.qsfaS2LineEndIdx =
            ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(qsfaRunParam.qsfaCubeSOuterOffset + qsfaRunParam.qsfaNextTokensPerBatch +
            qsfaRunParam.qsfaS1RealSize, 0, qsfaRunParam.qsfaActualS2Size);
        qsfaRunParam.qsfaS2LineEndIdx = Min(qsfaRunParam.qsfaS2LineEndIdx, qsfaConstInfo.qsfaSparseBlockCount); // 当前LI输出的block size只可能是1
    }

    qsfaRunParam.kvLoopEndIdx = (qsfaRunParam.qsfaS2LineEndIdx + qsfaS2BaseSize - 1) / qsfaS2BaseSize;
    qsfaRunParam.qsfaS2LoopEndIdx = qsfaRunParam.kvLoopEndIdx;
    return false;
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr& qsfaRunParam, RunInfo &qsfaRunInfo)
{
    qsfaRunInfo.qsfaBoIdx = qsfaRunParam.qsfaBoIdx;
    qsfaRunInfo.qsfaPreTokensPerBatch = qsfaRunParam.qsfaPreTokensPerBatch;
    qsfaRunInfo.qsfaNextTokensPerBatch = qsfaRunParam.qsfaNextTokensPerBatch;
    qsfaRunInfo.qsfaActualS1Size = qsfaRunParam.qsfaActualS1Size;
    qsfaRunInfo.qsfaActualS2Size = qsfaRunParam.qsfaActualS2Size;
    qsfaRunInfo.softmaxLseOffset = qsfaRunParam.softmaxLseOffset;
    qsfaRunInfo.qSNumInOneBlock = qsfaRunParam.qSNumInOneBlock;
    qsfaRunInfo.kvLoopEndIdx = qsfaRunParam.kvLoopEndIdx;
}

#endif  // KV_QUANT_SPARSE_FLASH_ATTENTION_KVCACHE_H
