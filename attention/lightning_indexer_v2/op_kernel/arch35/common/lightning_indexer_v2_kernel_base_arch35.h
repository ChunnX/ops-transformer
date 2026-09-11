/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H

#include "lightning_indexer_v2_base_arch35.h"

namespace LIV2Common {

__aicore__ inline uint32_t GetActualSeqLen(uint32_t batchIdx, bool hasCuSeqlens, bool hasSeqused,
    AscendC::GlobalTensor<uint32_t> &cuSeqlensGm, AscendC::GlobalTensor<uint32_t> &sequsedGm,
    uint32_t defaultSeqLen)
{
    if (hasSeqused) {
        return sequsedGm.GetValue(batchIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(batchIdx + 1) - cuSeqlensGm.GetValue(batchIdx);
    } else {
        return defaultSeqLen;
    }
}

template <typename RatioT>
__aicore__ inline uint32_t GetMaskedS2BaseBlockNum(uint32_t s1gIdx, uint32_t actS1Size,
    uint32_t actS2SizeOrig, uint32_t s1BaseSize, RatioT cmpRatio, uint32_t s2BaseSize,
    uint32_t *validLength = nullptr)
{
    if (actS2SizeOrig / cmpRatio == 0) {
        return 0;
    }
    uint32_t s1Offset = s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase +
                         static_cast<int32_t>(s1BaseSize)) / static_cast<int32_t>(cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / cmpRatio);
    validS2Len = Max(validS2Len, 1);
    if (validLength != nullptr) {
        *validLength = validS2Len;
    }
    return (validS2Len + s2BaseSize - 1) / s2BaseSize;
}

// The derived kernel owns operator-specific state; the base adds no data members.
template <typename Derived, typename RunInfoT, typename ConstInfoT>
class LightningIndexerV2KernelBase {
protected:
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx)
    {
        auto &kernel = static_cast<Derived &>(*this);
        kernel.tempLoopInfo.bN2Idx = bN2Idx;
        kernel.tempLoopInfo.bIdx = bN2Idx / kernel.constInfo.kHeadNum;
        kernel.tempLoopInfo.n2Idx = bN2Idx % kernel.constInfo.kHeadNum;
    }

    __aicore__ inline void InitBuffers()
    {
        auto &kernel = static_cast<Derived &>(*this);
        if ASCEND_IS_AIV {
            kernel.vectorService.InitBuffers(kernel.pipe);
        } else {
            kernel.matmulService.InitBuffers(kernel.pipe);
        }
    }

    __aicore__ inline void InitLoopState(RunInfoT &) {}
    __aicore__ inline void BeginS2Loop(RunInfoT &) {}
    __aicore__ inline void EndS2Loop(RunInfoT &) {}

    __aicore__ inline void ProcessMain()
    {
        auto &kernel = static_cast<Derived &>(*this);
        auto &splitCoreInfo = kernel.splitCoreInfo;
        auto &tempLoopInfo = kernel.tempLoopInfo;
        if (!splitCoreInfo.isCoreEnable) {
            return;
        }

        if ASCEND_IS_AIV {
            kernel.vectorService.AllocEventID();
            AscendC::CrossCoreSetFlag<ConstInfoT::SYNC_MODE4, PIPE_V>(ConstInfoT::CROSS_VC_EVENT + 0);
            AscendC::CrossCoreSetFlag<ConstInfoT::SYNC_MODE4, PIPE_V>(ConstInfoT::CROSS_VC_EVENT + 1);
        } else {
            kernel.matmulService.AllocEventID();
        }

        RunInfoT runInfo;
        kernel.InitLoopState(runInfo);
        uint32_t gloop = 0;
        for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
            kernel.CalcGS1LoopParams(bN2LoopIdx);
            if (tempLoopInfo.curActSeqLenIsZero) {
                kernel.DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
                continue;
            }
            for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
                kernel.CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
                kernel.BeginS2Loop(runInfo);
                runInfo.s2Start = splitCoreInfo.s2Start;
                runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;
                for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                    kernel.ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                    ++gloop;
                }
                kernel.EndS2Loop(runInfo);
                splitCoreInfo.s2Start = 0;
            }
            if (tempLoopInfo.needDealActS1LessThanS1) {
                kernel.DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
            }
            splitCoreInfo.gS1Start = 0;
        }

        if ASCEND_IS_AIV {
            kernel.vectorService.FreeEventID();
        } else {
            kernel.matmulService.FreeEventID();
            AscendC::CrossCoreWaitFlag<ConstInfoT::SYNC_MODE4, PIPE_FIX>(ConstInfoT::CROSS_VC_EVENT + 0);
            AscendC::CrossCoreWaitFlag<ConstInfoT::SYNC_MODE4, PIPE_FIX>(ConstInfoT::CROSS_VC_EVENT + 1);
            AscendC::CrossCoreWaitFlag<ConstInfoT::SYNC_MODE4, PIPE_FIX>(
                ConstInfoT::CROSS_VC_EVENT + 0 + ConstInfoT::AIV0_AIV1_OFFSET);
            AscendC::CrossCoreWaitFlag<ConstInfoT::SYNC_MODE4, PIPE_FIX>(
                ConstInfoT::CROSS_VC_EVENT + 1 + ConstInfoT::AIV0_AIV1_OFFSET);
        }
    }

    template <typename MetadataTraits>
    __aicore__ inline void LoadSplitCoreInfo(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
        AscendC::GlobalTensor<uint32_t> &metadataGm)
    {
        auto &kernel = static_cast<Derived &>(*this);
        auto &constInfo = kernel.constInfo;
        auto &splitCoreInfo = kernel.splitCoreInfo;
        auto &ldInfo = kernel.ldInfo;
        uint32_t liCoreEnableIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kCoreEnableIndex);
        uint32_t bN2StartIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kBn2StartIndex);
        uint32_t mStartIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kMStartIndex);
        uint32_t s2StartIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kS2StartIndex);
        uint32_t bN2EndIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kBn2EndIndex);
        uint32_t mEndIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kMEndIndex);
        uint32_t s2EndIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kS2EndIndex);

        uint32_t liZeroCoreEnableIndex = MetadataTraits::GetIndex(0, MetadataTraits::kCoreEnableIndex);
        if (metadataGm.GetValue(liZeroCoreEnableIndex) == 0) {
            kernel.isUsedCoreEqZero = true;
        }
        if (metadataGm.GetValue(liCoreEnableIndex) == 0) {
            splitCoreInfo.isCoreEnable = false;
            return;
        } else {
            splitCoreInfo.isCoreEnable = true;
        }

        splitCoreInfo.bN2Start = metadataGm.GetValue(bN2StartIndex);
        splitCoreInfo.gS1Start = metadataGm.GetValue(mStartIndex);
        splitCoreInfo.s2Start = metadataGm.GetValue(s2StartIndex);
        splitCoreInfo.bN2End = metadataGm.GetValue(bN2EndIndex);
        splitCoreInfo.gS1End = metadataGm.GetValue(mEndIndex);
        splitCoreInfo.s2End = metadataGm.GetValue(s2EndIndex);

        if (splitCoreInfo.s2End != 0) {
            // 此时只需要s2End往前退一格，bN2End和gS1End都不变
            splitCoreInfo.s2End = splitCoreInfo.s2End - 1;
        } else {
            if (splitCoreInfo.gS1End != 0) {
                // splitCoreInfo.gS1End != 0 splitCoreInfo.s2End == 0 时，gS1End需要往前退一格, bN2End不变
                // 此时需要使用bIdx获取实际Actal S2来计算出 s2End
                splitCoreInfo.gS1End = splitCoreInfo.gS1End - 1;
                // 需要获取当前的Actaul S2
                uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kHeadNum;
                uint32_t actS1Size, actS2Size, actS2SizeOrig;
                kernel.GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
                // s2的切块数量
                uint32_t s2BaseNum;
                if (constInfo.attenMaskFlag) {
                    s2BaseNum = kernel.GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2SizeOrig);
                } else {
                    s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
                }
                splitCoreInfo.s2End = s2BaseNum - 1;
            } else {
                // splitCoreInfo.gS1End == 0 splitCoreInfo.s2End == 0 时，bN2End需要往前退一格
                // 此时需要使用bIdx获取实际Actal S1和S2来计算出 gS1End 和 s2End
                splitCoreInfo.bN2End = splitCoreInfo.bN2End - 1;

                // 需要获取当前的Actaul S1 S2
                uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kHeadNum;
                uint32_t actS1Size, actS2Size, actS2SizeOrig;
                kernel.GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);

                // s1的切块数量
                uint32_t s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
                splitCoreInfo.gS1End = s1GBaseNum - 1;

                // s2的切块数量
                uint32_t s2BaseNum;
                if (constInfo.attenMaskFlag) {
                    s2BaseNum = kernel.GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2SizeOrig);
                } else {
                    s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
                }
                splitCoreInfo.s2End = s2BaseNum - 1;
            }
        }

        uint32_t ldFirstWorkSpaceIndex = MetadataTraits::GetIndex(cubeCoreIdx, MetadataTraits::kFirstWorkspaceIndex, false);
        ldInfo.saveWorkSpaceIdx = metadataGm.GetValue(ldFirstWorkSpaceIndex);
        if ASCEND_IS_AIV {
            uint32_t ldCoreEnableIndex = MetadataTraits::GetIndex(vecCoreIdx, MetadataTraits::kLdCoreEnableIndex, true);
            ldInfo.isLdCoreEnable = metadataGm.GetValue(ldCoreEnableIndex);

            if (!ldInfo.isLdCoreEnable) {
                return;
            }

            uint32_t ldBn2IdxIndex = MetadataTraits::GetIndex(vecCoreIdx, MetadataTraits::kLdBn2Index, true);
            uint32_t ldMIdxIndex = MetadataTraits::GetIndex(vecCoreIdx, MetadataTraits::kLdMIndex, true);
            uint32_t ldWorkspaceIdxIndex = MetadataTraits::GetIndex(vecCoreIdx, MetadataTraits::kLdWorkspaceIndex, true);
            uint32_t ldWorkspaceNumIndex = MetadataTraits::GetIndex(vecCoreIdx, MetadataTraits::kLdWorkspaceNumIndex, true);
            uint32_t ldMstartIndex = MetadataTraits::GetIndex(vecCoreIdx, MetadataTraits::kLdMStartIndex, true);
            uint32_t ldMNumIndex = MetadataTraits::GetIndex(vecCoreIdx, MetadataTraits::kLdMNumIndex, true);

            ldInfo.bn2Idx = metadataGm.GetValue(ldBn2IdxIndex);
            ldInfo.bIdx = ldInfo.bn2Idx / constInfo.kHeadNum;
            ldInfo.n2Idx = ldInfo.bn2Idx % constInfo.kHeadNum;
            ldInfo.mIdx = metadataGm.GetValue(ldMIdxIndex);
            ldInfo.workspaceIdx = metadataGm.GetValue(ldWorkspaceIdxIndex);
            ldInfo.workspaceNum = metadataGm.GetValue(ldWorkspaceNumIndex);
            ldInfo.mStart = metadataGm.GetValue(ldMstartIndex);
            ldInfo.mNum = metadataGm.GetValue(ldMNumIndex);
            kernel.UpdateLdOutputOffset();
        }
    }

    __aicore__ inline void ProcessDecode()
    {
        auto &kernel = static_cast<Derived &>(*this);
        if ASCEND_IS_AIV {
            kernel.vectorService.InitLDBuffers(kernel.pipe, kernel.ldInfo);
            AscendC::ICachePreLoad(Derived::LD_PREFETCH_LEN);
            AscendC::SyncAll();
            if (kernel.ldInfo.isLdCoreEnable) {
                kernel.vectorService.ProcessLD();
            }
        }
    }
};

} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_KERNEL_BASE_ARCH35_H
