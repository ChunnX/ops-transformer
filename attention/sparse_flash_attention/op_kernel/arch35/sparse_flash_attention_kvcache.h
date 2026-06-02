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
 * \file sparse_attn_sharedkv_kvcache.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_KVCACHE_H
#define SPARSE_FLASH_ATTENTION_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "sparse_flash_attention_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;

static constexpr uint32_t sparseModeZero = 0;
static constexpr uint32_t sparseModeThree = 3;

TEMPLATE_INTF
__aicore__ inline void CalculateQueryOffset(RunParamStr& sfaRunParam,
    const ConstInfo &sfaConstInfo, int32_t bIdx,
    __gm__ int32_t* actualSeqQlenAddr)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        sfaRunParam.sfaQBOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1] * sfaConstInfo.sfaGSize * 512;
        sfaRunParam.sfaQRopeBOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1] * sfaConstInfo.sfaGSize * 64;
    }
}

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr& sfaRunParam, const ConstInfo &sfaConstInfo,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t * actualSeqKvlenAddr)
{
    int32_t actualS1Size = 0;
    int32_t actualS2Size = 0;
    int32_t actualSeqMin = 1;
    int32_t actualSeqKVMin = 1;
    int32_t sIdx = sfaRunParam.sfaBoIdx;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        // actual seq length first
        if (actualSeqQlenAddr != nullptr) {
            actualS1Size = (sIdx == 0) ? actualSeqQlenAddr[0] :
                actualSeqQlenAddr[sIdx] - actualSeqQlenAddr[sIdx - 1];
        } else {
            actualS1Size = sfaConstInfo.sfaS1Size;
        }
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? sfaConstInfo.sfaS1Size :
            actualSeqQlenAddr[sIdx];
    }

    if (sfaConstInfo.sfaIsActualLenDimsKVNull) {
        actualS2Size = sfaConstInfo.sfaS2Size;
    } else {
        if constexpr (isPa) {
            if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
                actualS2Size = actualSeqKvlenAddr[sIdx];
            } else {
                actualS2Size = (sfaConstInfo.sfaActualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        } else {
            if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
                actualS2Size = (sIdx == 0) ? actualSeqKvlenAddr[0] :
                    actualSeqKvlenAddr[sIdx] - actualSeqKvlenAddr[sIdx - 1];
            } else {
                actualS2Size = (sfaConstInfo.sfaActualSeqLenKVSize == actualSeqKVMin) ?
                    actualSeqKvlenAddr[0] : actualSeqKvlenAddr[sIdx];
            }
        }
    }

    sfaRunParam.sfaActualS1Size = actualS1Size;
    sfaRunParam.sfaActualS2Size = actualS2Size;
    if (sfaConstInfo.sfaSparseMode == sparseModeZero) {
        sfaRunParam.sfaNextTokensPerBatch = MAX_PRE_NEXT_TOKENS;
    } else {
        sfaRunParam.sfaNextTokensPerBatch = sfaRunParam.sfaActualS2Size - sfaRunParam.sfaActualS1Size;
    }
    sfaRunParam.sfaPreTokensPerBatch = sfaRunParam.sfaActualS1Size;
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& sfaRunParam, const ConstInfo &sfaConstInfo,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t *actualSeqKvlenAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(sfaRunParam, sfaConstInfo, actualSeqQlenAddr, actualSeqKvlenAddr);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& sfaRunParam, const ConstInfo &sfaConstInfo, bool lastBN,
    int64_t nextGs1Idx, int64_t gS1StartIdx)
{
    sfaRunParam.qSNumInOneBlock = 1; // 不切G轴, 计算每个基本块可以拷贝多少行s
    sfaRunParam.gs1LoopStartIdx = gS1StartIdx;
    if (sfaRunParam.sfaNextTokensPerBatch < 0) {
        int64_t gs1LoopStartIdx = sfaRunParam.sfaNextTokensPerBatch * (-1) / sfaRunParam.qSNumInOneBlock
                                    * sfaRunParam.qSNumInOneBlock;
        if (gs1LoopStartIdx > gS1StartIdx) {
            sfaRunParam.gs1LoopStartIdx = gs1LoopStartIdx;
        }
    }

    int32_t gs1LoopEndIdx = sfaRunParam.sfaActualS1Size; // 不切G轴, 每次拷贝一行的topk，只算一行的qs

    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        sfaRunParam.gs1LoopEndIdx = gs1LoopEndIdx;
    } else { // 最后一个bn, 从数组下一个元素取值
        sfaRunParam.gs1LoopEndIdx = nextGs1Idx == 0 ? gs1LoopEndIdx : nextGs1Idx;
    }

    if (sfaRunParam.gs1LoopStartIdx > sfaRunParam.gs1LoopEndIdx) {
        sfaRunParam.gs1LoopStartIdx = sfaRunParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr& sfaRunParam, const ConstInfo &sfaConstInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t sfaCubeSOuterOffset = sOuterLoopIdx * sfaRunParam.qSNumInOneBlock;
    if (sfaRunParam.sfaActualS1Size == 0) {
        sfaRunParam.sfaS1RealSize = 0;
        sfaRunParam.sfaMRealSize = 0;
    } else {
        sfaRunParam.sfaS1RealSize = Min(sfaRunParam.qSNumInOneBlock, sfaRunParam.sfaActualS1Size - sfaCubeSOuterOffset);
        sfaRunParam.sfaMRealSize = sfaRunParam.sfaS1RealSize * sfaConstInfo.sfaGSize;
        if constexpr (IS_SPLIT_G) {
            uint32_t firstHalfG = (sfaConstInfo.sfaGSize + 1) >> 1;
            uint32_t secondHalfG = sfaConstInfo.sfaGSize - firstHalfG;
            sfaRunParam.sfaMRealSize = sfaRunParam.sfaS1RealSize * 
                ((sfaConstInfo.sfaAicIdx % 2 == 0) ? firstHalfG : secondHalfG);
        }
    }

    sfaRunParam.cubeMOuterOffset = sfaCubeSOuterOffset * sfaConstInfo.sfaGSize;
    sfaRunParam.sfaHalfMRealSize = (sfaRunParam.sfaMRealSize + 1) >> 1;
    sfaRunParam.sfaFirstHalfMRealSize = sfaRunParam.sfaHalfMRealSize;
    if (sfaConstInfo.sfaSubBlockIdx == 1) {
        sfaRunParam.sfaHalfMRealSize = sfaRunParam.sfaMRealSize - sfaRunParam.sfaHalfMRealSize;
        sfaRunParam.sfaMOuterOffset = sfaRunParam.cubeMOuterOffset + sfaRunParam.sfaFirstHalfMRealSize;
    } else {
        sfaRunParam.sfaMOuterOffset = sfaRunParam.cubeMOuterOffset;
    }

    sfaRunParam.sfaHalfS1RealSize = (sfaRunParam.sfaS1RealSize + 1) >> 1;
    sfaRunParam.sfaFirstHalfS1RealSize = sfaRunParam.sfaHalfS1RealSize;
    if (sfaConstInfo.sfaSubBlockIdx == 1) {
        sfaRunParam.sfaHalfS1RealSize = sfaRunParam.sfaS1RealSize - sfaRunParam.sfaHalfS1RealSize;
        sfaRunParam.sfaSOuterOffset = sfaCubeSOuterOffset + sfaRunParam.sfaHalfMRealSize / sfaConstInfo.sfaGSize;
    } else {
        sfaRunParam.sfaSOuterOffset = sfaCubeSOuterOffset;
    }
    sfaRunParam.sfaCubeSOuterOffset = sfaCubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr& sfaRunParam, const ConstInfo &sfaConstInfo,
    int32_t sIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            seqOffset = sIdx == 0 ? 0 : cuSeqlensQAddr[sIdx - 1];
        } else {
            seqOffset = sIdx * sfaConstInfo.sfaS1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * sfaConstInfo.sfaN2GDv;
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND || LAYOUT_T == SFA_LAYOUT::TND) {
            sfaRunParam.sfaAttentionOutOffset = attentionOutSeqOffset +
                sfaRunParam.sfaSOuterOffset * sfaConstInfo.sfaN2GDv + sfaRunParam.sfaN2oIdx * sfaConstInfo.sfaGDv +
                sfaRunParam.sfaGoIdx * sfaConstInfo.sfaDSizeV;
        }
        if (sfaConstInfo.sfaSubBlockIdx == 1) {
            sfaRunParam.sfaAttentionOutOffset += sfaRunParam.sfaFirstHalfMRealSize * sfaConstInfo.sfaDSizeV;
        }
        if (sfaConstInfo.sfaReturnSoftmaxLse) {
            if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
                // [N2, T, G] (TND)
                sfaRunParam.softmaxLseOffset = sfaRunParam.sfaN2oIdx * sfaConstInfo.sfaS1Size * sfaConstInfo.sfaGSize +
                    (seqOffset + sfaRunParam.sfaSOuterOffset) * sfaConstInfo.sfaGSize;
            } else {
                sfaRunParam.softmaxLseOffset = sIdx * sfaConstInfo.sfaN2Size * sfaConstInfo.sfaS1Size * 
                    sfaConstInfo.sfaGSize +
                sfaRunParam.softmaxLseOffset = sIdx * sfaConstInfo.sfaN2Size * sfaConstInfo.sfaS1Size * 
                    sfaConstInfo.sfaGSize +
                    sfaRunParam.sfaN2oIdx * sfaConstInfo.sfaS1Size * sfaConstInfo.sfaGSize +
                    sfaRunParam.sfaSOuterOffset * sfaConstInfo.sfaGSize;
            }
            if (IS_SPLIT_G && sfaConstInfo.sfaAicIdx % 2U != 0) {
                sfaRunParam.softmaxLseOffset += (sfaConstInfo.sfaGSize + 1U) >> 1U; // splitG时，需要偏移前一半G
            }
            if (sfaConstInfo.sfaSubBlockIdx == 1) {
                sfaRunParam.softmaxLseOffset += sfaRunParam.sfaFirstHalfMRealSize;
                }
            }
            sfaRunParam.sfaTensorQOffset = sfaRunParam.sfaQBOffset + sfaRunParam.sfaCubeSOuterOffset * 
                sfaConstInfo.sfaN2GD +
            sfaRunParam.sfaTensorQRopeOffset = sfaRunParam.sfaQRopeBOffset + sfaRunParam.sfaCubeSOuterOffset * 
                sfaConstInfo.sfaN2GD +
            sfaRunParam.sfaTensorQOffset = sfaRunParam.sfaQBOffset + sfaRunParam.sfaCubeSOuterOffset * 
                sfaConstInfo.sfaN2GD +
                sfaRunParam.sfaN2oIdx * sfaConstInfo.sfaGD + sfaRunParam.sfaGoIdx * sfaConstInfo.sfaDSize;
            sfaRunParam.sfaTensorQRopeOffset = sfaRunParam.sfaQRopeBOffset + sfaRunParam.sfaCubeSOuterOffset * 
                sfaConstInfo.sfaN2GD +
            sfaRunParam.sfaTensorQRopeOffset = sfaRunParam.sfaQRopeBOffset + sfaRunParam.sfaN2oIdx * 
                sfaConstInfo.sfaGS1D +
        } else {
            sfaRunParam.sfaTensorQOffset = sfaRunParam.sfaQBOffset + sfaRunParam.sfaN2oIdx * sfaConstInfo.sfaGS1D +
                sfaRunParam.sfaGoIdx * sfaConstInfo.sfaS1D + sfaRunParam.sfaCubeSOuterOffset * sfaConstInfo.sfaDSize;
            sfaRunParam.sfaTensorQRopeOffset = sfaRunParam.sfaQRopeBOffset + sfaRunParam.sfaN2oIdx * 
                sfaConstInfo.sfaGS1D +
                sfaRunParam.sfaGoIdx * sfaConstInfo.sfaS1D +
                    sfaRunParam.sfaCubeSOuterOffset * sfaConstInfo.sfaDSizeRope;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr& sfaRunParam, const ConstInfo &sfaConstInfo,
    uint32_t sOuterLoopIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if (sfaRunParam.sfaNextTokensPerBatch < 0) {
        if (sfaRunParam.sfaS1oIdx < (sfaRunParam.sfaNextTokensPerBatch * (-1)) \
            / sfaRunParam.qSNumInOneBlock * sfaRunParam.qSNumInOneBlock) {
            return true;
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(sfaRunParam, sfaConstInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(sfaRunParam, sfaConstInfo, sfaRunParam.sfaBoIdx, cuSeqlensQAddr);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr& sfaRunParam, __gm__ int32_t *cuSeqlensQAddr)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (sfaRunParam.sfaBoIdx > 0 && ((sfaRunParam.sfaBoIdx == 0 && cuSeqlensQAddr[sfaRunParam.sfaBoIdx] == 0) ||
            (cuSeqlensQAddr[sfaRunParam.sfaBoIdx] - cuSeqlensQAddr[sfaRunParam.sfaBoIdx - 1] == 0))) {
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
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr& sfaRunParam, const ConstInfo &sfaConstInfo)
{
    if (sfaRunParam.sfaActualS2Size == 0) {
        sfaRunParam.kvLoopEndIdx = 0;
        sfaRunParam.sfaS2LoopEndIdx = 0;
        return true;
    }
    uint32_t sfaS2BaseSize = sfaConstInfo.sfaS2BaseSize;

    if (sfaConstInfo.sfaSparseMode == sparseModeZero) {
        sfaRunParam.sfaS2LineStartIdx = 0;
        sfaRunParam.sfaS2LineEndIdx = Min(sfaRunParam.sfaActualS2Size, sfaConstInfo.sfaSparseBlockCount);
    } else if (sfaConstInfo.sfaSparseMode == sparseModeThree) {
        // 当前LI输出的block size只可能是1
        sfaRunParam.sfaS2LineEndIdx = Min(sfaRunParam.sfaS2LineEndIdx, sfaConstInfo.sfaSparseBlockCount);
            sfaRunParam.sfaCubeSOuterOffset - sfaRunParam.sfaPreTokensPerBatch, 0, sfaRunParam.sfaActualS2Size);
        sfaRunParam.sfaS2LineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            sfaRunParam.sfaCubeSOuterOffset + sfaRunParam.sfaNextTokensPerBatch +
            sfaRunParam.sfaS1RealSize, 0, sfaRunParam.sfaActualS2Size);
        // 当前LI输出的block size只可能是1
        sfaRunParam.sfaS2LineEndIdx = Min(sfaRunParam.sfaS2LineEndIdx, sfaConstInfo.sfaSparseBlockCount);
    }

    sfaRunParam.kvLoopEndIdx = (sfaRunParam.sfaS2LineEndIdx + sfaS2BaseSize - 1) / sfaS2BaseSize;
    sfaRunParam.sfaS2LoopEndIdx = sfaRunParam.kvLoopEndIdx;
    return false;
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr& sfaRunParam, RunInfo &sfaRunInfo)
{
    sfaRunInfo.sfaBoIdx = sfaRunParam.sfaBoIdx;
    sfaRunInfo.sfaPreTokensPerBatch = sfaRunParam.sfaPreTokensPerBatch;
    sfaRunInfo.sfaNextTokensPerBatch = sfaRunParam.sfaNextTokensPerBatch;
    sfaRunInfo.sfaActualS1Size = sfaRunParam.sfaActualS1Size;
    sfaRunInfo.sfaActualS2Size = sfaRunParam.sfaActualS2Size;
    sfaRunInfo.softmaxLseOffset = sfaRunParam.softmaxLseOffset;
    sfaRunInfo.qSNumInOneBlock = sfaRunParam.qSNumInOneBlock;
    sfaRunInfo.kvLoopEndIdx = sfaRunParam.kvLoopEndIdx;
}

#endif  // SPARSE_FLASH_ATTENTION_KVCACHE_H
