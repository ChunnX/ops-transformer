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
 * \file sparse_flash_mla_kvcache.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_KVCACHE_H
#define SPARSE_FLASH_MLA_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "sparse_flash_mla_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace SMLAKernel;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr& runParam, const ConstInfo &constInfo,
    __gm__ int32_t *cuSeqlensQAddr, __gm__ int32_t *cuSeqlensOriKvAddr, __gm__ int32_t *cuSeqlensCmpKvAddr,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t *actualSeqOriKvlenAddr,
    __gm__ int32_t *actualSeqCmpKvlenAddr, __gm__ int32_t *cmpResidualKvAddr)
{
    int32_t actualS1Size = 0;
    //int32_t actualS2Size = 0; // del
    int32_t actualS2OriSize = 0;
    int32_t actualS2CmpSize = 0;
    int32_t actualSeqMin = 1;
    int32_t actualSeqKVMin = 1;
    int32_t bIdx = runParam.boIdx;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // actual seq length first
        actualS1Size = (actualSeqQlenAddr == nullptr) ? (cuSeqlensQAddr[bIdx + 1] - cuSeqlensQAddr[bIdx]) :
            actualSeqQlenAddr[bIdx];
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size :
            actualSeqQlenAddr[bIdx];
    }

    // if (constInfo.isActualLenDimsOriKVNull) {  补回来
    //     actualS2OriSize = constInfo.s2Size;
    // } else {
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            if constexpr (IS_PA) {
                actualS2OriSize = actualSeqOriKvlenAddr[bIdx];
            } else {
                actualS2OriSize = cuSeqlensOriKvAddr[bIdx + 1] - cuSeqlensOriKvAddr[bIdx];
            }
        } else {
            // actualS2OriSize = (constInfo.actualLenDimsOriKV == actualSeqKVMin) ?
            //     actualSeqOriKvlenAddr[0] : actualSeqOriKvlenAddr[bIdx];
            actualS2OriSize = actualSeqOriKvlenAddr[bIdx];
        }
    //}

    // 处理cmp的actualSeq
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            if (actualSeqCmpKvlenAddr != nullptr) {    // 有actualSeq先用，没有再看cuSeqlens，什么都没有就做除法
                actualS2CmpSize = actualSeqCmpKvlenAddr[bIdx];
            } else if (cuSeqlensCmpKvAddr != nullptr) {
                actualS2CmpSize = cuSeqlensCmpKvAddr[bIdx + 1] - cuSeqlensCmpKvAddr[bIdx];
            } else {
                actualS2CmpSize = actualS2OriSize / constInfo.cmpRatio;
            }
        } else {
            actualS2CmpSize = (actualSeqCmpKvlenAddr == nullptr) ? (actualS2OriSize / constInfo.cmpRatio) :
                (constInfo.actualLenDimsCmpKV == actualSeqKVMin) ? actualSeqCmpKvlenAddr[0] : actualSeqCmpKvlenAddr[bIdx];
        }
    }
    runParam.cmpSparseBlockCount = constInfo.cmpSparseBlockCount;


    runParam.actualS1Size = actualS1Size;
    runParam.actualS2OriSize = actualS2OriSize;
    runParam.actualS2CmpSize = actualS2CmpSize;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        runParam.cmpResidual = (cmpResidualKvAddr != nullptr) ? cmpResidualKvAddr[bIdx] : 0;
        runParam.nextTokensPerBatchCmp = (int64_t)runParam.actualS2CmpSize * constInfo.cmpRatio + runParam.cmpResidual - runParam.actualS1Size;
    }
    runParam.nextTokensPerBatchOri = runParam.actualS2OriSize - runParam.actualS1Size;

    // 只包含了ori_mask_mode=4的部分
    if (constInfo.oriWinLeft == -1) {
        runParam.preTokensPerBatchOri = runParam.actualS1Size;
    } else {
        runParam.preTokensPerBatchOri = -(runParam.actualS2OriSize - runParam.actualS1Size - constInfo.oriWinLeft);
    }
    runParam.preTokensPerBatchOri = Min(runParam.preTokensPerBatchOri, runParam.actualS1Size);
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr& runParam, const ConstInfo &constInfo,
    __gm__ int32_t *cuSeqlensQAddr, __gm__ int32_t *cuSeqlensOriKvAddr, __gm__ int32_t *cuSeqlensCmpKvAddr,
    __gm__ int32_t *actualSeqQlenAddr, __gm__ int32_t *actualSeqOriKvlenAddr,
    __gm__ int32_t *actualSeqCmpKvlenAddr, __gm__ int32_t *cmpResidualKvAddr)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, cuSeqlensQAddr, cuSeqlensOriKvAddr, cuSeqlensCmpKvAddr,
        actualSeqQlenAddr, actualSeqOriKvlenAddr, actualSeqCmpKvlenAddr, cmpResidualKvAddr);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr& runParam, const ConstInfo &constInfo, bool lastBN,
    int64_t nextGs1Idx, int64_t gS1StartIdx)
{
    runParam.qSNumInOneBlock = (constInfo.gSize <= 64) ? \
        (constInfo.s1BaseSize / constInfo.gSize) : 1;
    runParam.gs1LoopStartIdx = gS1StartIdx;
    if (runParam.nextTokensPerBatchOri < 0) {
        int64_t gs1LoopStartIdx = runParam.nextTokensPerBatchOri * \
            (-1) / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock;
        if (gs1LoopStartIdx > gS1StartIdx) {
            runParam.gs1LoopStartIdx = gs1LoopStartIdx;
        }
    }

    int32_t gs1LoopEndIdx = 0;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        gs1LoopEndIdx = runParam.actualS1Size; // 对于SCFA, 不切G轴, 每次拷贝一行的topk，只算一行的qs
    } else { // SWA/CFA
        // 不需要取topk, 每次计算gSize行, 循环qs次
        gs1LoopEndIdx = (runParam.actualS1Size + runParam.qSNumInOneBlock - 1) / runParam.qSNumInOneBlock;
    }
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.gs1LoopEndIdx = gs1LoopEndIdx;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.gs1LoopEndIdx = nextGs1Idx == 0 ? gs1LoopEndIdx : nextGs1Idx;
    }

    if (runParam.gs1LoopStartIdx > runParam.gs1LoopEndIdx) {
        runParam.gs1LoopStartIdx = runParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * runParam.qSNumInOneBlock;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
        runParam.mRealSize = 0;
    } else {
        runParam.s1RealSize = Min(runParam.qSNumInOneBlock, runParam.actualS1Size - cubeSOuterOffset);
        runParam.mRealSize = runParam.s1RealSize * constInfo.gSize;
        if constexpr (IS_SPLIT_G) {
            runParam.mRealSize = runParam.mRealSize >> 1;
        }
    }

    runParam.cubeMOuterOffset = cubeSOuterOffset * constInfo.gSize;
    runParam.halfMRealSize = (runParam.mRealSize + 1) >> 1;
    runParam.firstHalfMRealSize = runParam.halfMRealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfMRealSize = runParam.mRealSize - runParam.halfMRealSize;
        runParam.mOuterOffset = runParam.cubeMOuterOffset + runParam.firstHalfMRealSize;
    } else {
        runParam.mOuterOffset = runParam.cubeMOuterOffset;
    }

    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
        runParam.sOuterOffset = cubeSOuterOffset + runParam.halfMRealSize / constInfo.gSize;
    } else {
        runParam.sOuterOffset = cubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr& runParam, const ConstInfo &constInfo,
    int32_t sIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            seqOffset = cuSeqlensQAddr[sIdx];
        } else {
            seqOffset = sIdx * constInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * constInfo.n2GDv;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND || LAYOUT_T == SMLA_LAYOUT::TND) {
            runParam.attentionOutOffset = attentionOutSeqOffset +
                runParam.sOuterOffset * constInfo.n2GDv + runParam.n2oIdx * constInfo.gDv +
                runParam.goIdx * constInfo.dSizeV;
        }
        if (constInfo.subBlockIdx == 1) {
            runParam.attentionOutOffset += runParam.halfMRealSize * constInfo.dSizeV;
        }
        if (constInfo.returnSoftmaxLse) {
            if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
                // [N2, T, G] (TND)
                runParam.softmaxLseOffset = runParam.n2oIdx * constInfo.s1Size * constInfo.gSize +
                    (seqOffset + runParam.sOuterOffset) * constInfo.gSize;
            } else {
                // [B, N2, S1, G] (BSND)
                runParam.softmaxLseOffset = sIdx * constInfo.n2Size * constInfo.s1Size * constInfo.gSize +
                    runParam.n2oIdx * constInfo.s1Size * constInfo.gSize +
                    runParam.sOuterOffset * constInfo.gSize;
            }
            uint32_t aicIdx = constInfo.aivIdx >> 1U;
            if (IS_SPLIT_G && aicIdx % 2U != 0) {
                runParam.softmaxLseOffset += 64; // splitG时，需要偏移64
            }
            if (constInfo.subBlockIdx == 1) {
                runParam.softmaxLseOffset += runParam.firstHalfMRealSize;
                }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr& runParam, const ConstInfo &constInfo,
    uint32_t sOuterLoopIdx, __gm__ int32_t *cuSeqlensQAddr)
{
    if (runParam.nextTokensPerBatchOri < 0) {
        if (runParam.s1oIdx < (runParam.nextTokensPerBatchOri * (-1)) \
            / runParam.qSNumInOneBlock * runParam.qSNumInOneBlock) {
            return true;
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, cuSeqlensQAddr);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr& runParam, __gm__ int32_t *cuSeqlensQAddr)
{
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (runParam.boIdx > 0 && cuSeqlensQAddr[runParam.boIdx + 1] - cuSeqlensQAddr[runParam.boIdx] == 0) {
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
__aicore__ inline bool ComputeS2LoopInfo(int64_t bnIndex, int64_t gS1Index, __gm__ int32_t *cuSeqlensQAddr,
    GlobalTensor<int32_t>& oriTopkLengthGm, bool hasOriTopkLength, GlobalTensor<int32_t>& cmpTopkLengthGm,
    bool hasCmpTopkLength, RunParamStr& runParam, const ConstInfo &constInfo)
{
    if (runParam.actualS2OriSize == 0) {
        runParam.oriKvLoopEndIdx = 0;
        runParam.cmpKvLoopEndIdx = 0;
        runParam.s2LoopEndIdx = 0;
        return true;
    }
    uint32_t s2BaseSize = constInfo.s2BaseSize;

    runParam.s2OriLineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        runParam.cubeSOuterOffset - runParam.preTokensPerBatchOri, 0, runParam.actualS2OriSize);
    runParam.s2OriLineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        runParam.cubeSOuterOffset + runParam.nextTokensPerBatchOri + runParam.s1RealSize, 0, runParam.actualS2OriSize);
    runParam.oriKvLoopEndIdx = (runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx + s2BaseSize - 1) / s2BaseSize;

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        runParam.s2CmpLineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            (runParam.cubeSOuterOffset + runParam.s1RealSize + runParam.nextTokensPerBatchCmp) / constInfo.cmpRatio,
            0, runParam.actualS2CmpSize);
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::CFA_TEMPLATE_MODE) {
            runParam.s2CmpLineEndIdx = Min(runParam.s2CmpLineEndIdx, runParam.actualS2CmpSize);
            runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
        } else { // SCFA_TEMPLATE_MODE
            runParam.s2CmpLineEndIdx = Min(runParam.s2CmpLineEndIdx, constInfo.cmpSparseBlockCount);
            runParam.s2CmpLineEndIdx = Min(runParam.s2CmpLineEndIdx, runParam.actualS2CmpSize);
            runParam.cmpKvLoopEndIdx = (runParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
        }
    } else {
        runParam.cmpKvLoopEndIdx = 0;
        runParam.s2CmpLineEndIdx = 0;
    }

    runParam.oriKvLoopEndIdx = (runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx + s2BaseSize - 1) / s2BaseSize;

    runParam.s2LoopEndIdx = runParam.oriKvLoopEndIdx + runParam.cmpKvLoopEndIdx;
    return (runParam.s2LoopEndIdx == 0);

}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr& runParam, RunInfo &runInfo, const ConstInfo &constInfo)
{
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatchOri = runParam.preTokensPerBatchOri;
    runInfo.nextTokensPerBatchOri = runParam.nextTokensPerBatchOri;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2OriSize = runParam.actualS2OriSize;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        runInfo.actualS2CmpSize = runParam.actualS2CmpSize;
        runInfo.cmpResidual = runParam.cmpResidual;
    }
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
    runInfo.qSNumInOneBlock = runParam.qSNumInOneBlock;
    runInfo.oriKvLoopEndIdx = runParam.oriKvLoopEndIdx;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        runInfo.cmpKvLoopEndIdx = runParam.cmpKvLoopEndIdx;
        runInfo.isCmp = runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx;
    }
    runInfo.oriSparseBlockCount = runParam.oriSparseBlockCount;
    runInfo.cmpSparseBlockCount = runParam.cmpSparseBlockCount;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE) {
        if ((runInfo.isCmp) && (!constInfo.hasCmpTopkLength)) {
            runInfo.isS2Gather = true;
        } else {
            runInfo.isS2Gather = false;
        }
    } else {
        runInfo.isS2Gather = false;
    }
}

#endif  // SPARSE_FLASH_MLA_KVCACHE_H