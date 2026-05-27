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
 * \file flash_attn_kernel_base_gqa_vector.h
 * \brief Flash Attention GQA vector processing methods
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_VECTOR_H_
#define FLASH_ATTENTION_NOQUANT_GQA_VECTOR_H_

#include "flash_attn_kernel_base_gqa_common.h"

namespace BaseApi {

template <typename VectorData>
class FlashAttentionVectorProcessor {
public:
    VectorData* vectorData;

    __aicore__ inline FlashAttentionVectorProcessor(VectorData* data) : vectorData(data){};

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            using VecMm2ResPos = typename VectorData::VecFaBlockTypeAlias::mm2ResPos;
            using UbBufferType = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
            if constexpr (std::is_same_v<VecMm2ResPos, UbBufferType>) {
                vectorData->ctx->bmm2Buffers.Get().SetCrossCore();
                vectorData->ctx->bmm2Buffers.Get().SetCrossCore();
            }
            vectorData->ctx->bmm1Buffers.Get().SetCrossCore();
            vectorData->ctx->bmm1Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void ProcessVector(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIV {
            vectorData->vecFaBlock.InitBuffers(vectorData->SharedBuffer1, vectorData->SharedBuffer2, vectorData->SharedBuffer3);
            vectorData->vecFaBlock.AllocEventID();
            if (vectorData->constInfo.aicIdx < vectorData->ctx->tilingData->flashAttnBaseParams.coreNum) {
                FlashAttentionVector(sectionIdx);
            }
            FlashDecodeVector(sectionIdx);
            vectorData->vecFaBlock.FreeEventID();
        }
    }

private:
    __aicore__ inline void FlashAttentionVector(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIV {
            if (vectorData->constInfo.aicIdx >= vectorData->ctx->tilingData->flashAttnBaseParams.coreNum) {
                return;
            }

            vectorData->GetFASectionInfo(sectionIdx);
            RunInfoXVec taskRunInfoVec[VectorData::PRELOAD_TASK_CACHE_SIZE] = {};

            vectorData->createdTaskCount = 0;
            vectorData->executedTaskCount = 0;
            vectorData->mloop = 0;
            vectorData->accGS1Loops = 0;
            vectorData->headS2Split = false;
            vectorData->tailS2Split = false;

            uint32_t bN2Cur = vectorData->bN2Start_;
            uint32_t gS1Cur = vectorData->gS1OStart_;
            uint32_t s2Cur = vectorData->s2OStart_;
            vectorData->prevBN2Idx = bN2Cur;
            vectorData->prevGS1Idx = gS1Cur;
            
            vectorData->coreFirstTmpOutWsPos_ = vectorData->ctx->faMetaDataGm.GetValue(
                vectorData->ctx->GetFAMetaDataIndex(vectorData->constInfo.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));

            bool shouldDispatchTask = true;
            bool shouldExecuteTask = false;
            while (shouldDispatchTask || shouldExecuteTask) {
                shouldDispatchTask = ShouldDispatchTaskVector(bN2Cur, gS1Cur, s2Cur);
                if (shouldDispatchTask) {
                    TASK_DEAL_MODE taskDealMode = GetTaskDealModeVector(bN2Cur, gS1Cur, s2Cur);
                    if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                        CreateTaskVector(vectorData->createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfoVec);
                        vectorData->createdTaskCount++;
                        UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                        vectorData->vecFaBlock.DealZeroActSeqLen(bN2Cur);
                        UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                        continue;
                    } else {
                        UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                        continue;
                    }
                }
                shouldExecuteTask = ShouldExecuteTaskVector(taskRunInfoVec);
                if (shouldExecuteTask) {
                    ExecuteTaskVector(vectorData->executedTaskCount, taskRunInfoVec);
                    vectorData->executedTaskCount++;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTaskVector(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if ASCEND_IS_AIV {
            return ((bN2Cur != vectorData->ctx->bN2End_) || (gS1Cur != vectorData->ctx->gS1OEnd_) || (s2Cur != vectorData->ctx->s2OEnd_));
        }
        return false;
    }

    __aicore__ inline bool ShouldExecuteTaskVector(RunInfoXVec taskRunInfoVec[VectorData::PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < VectorData::PRELOAD_TASK_CACHE_SIZE; i++) {
                if (taskRunInfoVec[i].isValid) {
                    return true;
                }
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealModeVector(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if ASCEND_IS_AIV {
            bool isFirstTask = (bN2Cur == vectorData->bN2Start_) && (gS1Cur == vectorData->gS1OStart_) && (s2Cur == vectorData->s2OStart_);
            uint32_t bIdx = bN2Cur / vectorData->ctx->tilingData->flashAttnBaseParams.n2Size;
            if (isFirstTask || vectorData->ctx->prevBIdx != bIdx) {
                vectorData->ctx->prevBIdx = bIdx;
                if (vectorData->ctx->tilingData->flashAttnBaseParams.cuSeqLensKVSize == 0 && vectorData->ctx->tilingData->flashAttnBaseParams.seqUsedKvSize == 0 &&
                    !vectorData->ctx->tilingData->flashAttnBaseParams.isKvContinuous) {
                    vectorData->ctx->actSeqLensKv = SeqLenFromTensorList<VectorData::LAYOUT_KV>(vectorData->ctx->keyPtr, bIdx);
                } else {
                    if constexpr (VectorData::LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                        vectorData->ctx->actSeqLensKv = vectorData->kvCuSeqLensParser.GetActualSeqLength(bIdx);
                    } else {
                        vectorData->ctx->actSeqLensKv = vectorData->kvSeqUsedParser.GetActualSeqLength(bIdx);
                    }
                }
                if constexpr (VectorData::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    vectorData->ctx->actSeqLensQ = vectorData->qCuSeqLensParser.GetActualSeqLength(bIdx);
                } else {
                    vectorData->ctx->actSeqLensQ = vectorData->qSeqUsedParser.GetActualSeqLength(bIdx);
                }
            }
            uint64_t s2LoopTimes = (vectorData->ctx->actSeqLensKv + VectorData::s2BaseSize - 1) / VectorData::s2BaseSize;
            uint64_t gS1Size = vectorData->ctx->actSeqLensQ * vectorData->ctx->tilingData->flashAttnBaseParams.gSize;
            uint64_t gS1LoopTimes = (gS1Size + VectorData::mBaseSize - 1) / VectorData::mBaseSize;
            if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
                if (gS1Cur == 0 && s2Cur == 0) {
                    return TASK_DEAL_MODE::DEAL_ZERO;
                }
                return TASK_DEAL_MODE::SKIP_ZERO;
            }

            if (isFirstTask || bN2Cur != vectorData->prevBN2Idx || gS1Cur != vectorData->prevGS1Idx) {
                if (vectorData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == 0) {
                    CalcCurS2StartEndNoSparseVector(bN2Cur, gS1Cur);
                } else {
                    CalcCurS2StartEndWithSparseVector(bN2Cur, gS1Cur);
                }
                vectorData->prevBN2Idx = bN2Cur;
                vectorData->prevGS1Idx = gS1Cur;
            }

            if (vectorData->ctx->curS2Start < vectorData->ctx->curS2End && s2Cur >= vectorData->ctx->curS2End) {
                return TASK_DEAL_MODE::S2_END;
            }

            if (s2Cur < vectorData->ctx->curS2Start || s2Cur >= vectorData->ctx->curS2End) {
                return TASK_DEAL_MODE::SKIP;
            }

            if (s2Cur == vectorData->ctx->curS2Start) {
                vectorData->mloop++;
            }

            return TASK_DEAL_MODE::CREATE_TASK;
        }
        return TASK_DEAL_MODE::SKIP;
    }

    __aicore__ inline void GetPreNextTokenLeftUpVector(int64_t actSeqLensQ, int64_t actSeqLensKv,
                                                        int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp)
    {
        if ASCEND_IS_AIV {
            preTokenLeftUp = vectorData->ctx->tilingData->flashAttnAttenMaskParams.winLefts;
            nextTokenLeftUp = vectorData->ctx->tilingData->flashAttnAttenMaskParams.winRights;
            fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                             vectorData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode);

            if (vectorData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::BAND) {
                preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
            }

            if (vectorData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
                nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
            } else if (vectorData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::BAND) {
                nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
            }
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparseVector(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIV {
            vectorData->ctx->curS2Start = 0U;
            vectorData->ctx->curS2End = (static_cast<uint32_t>(vectorData->ctx->actSeqLensKv) + VectorData::s2BaseSize - 1) / VectorData::s2BaseSize;

            if ((bN2Cur == vectorData->bN2Start_) && (gS1Cur == vectorData->gS1OStart_)) {
                vectorData->headS2Split = vectorData->s2OStart_ != 0U;
                vectorData->ctx->curS2Start = vectorData->s2OStart_;
            }

            if ((bN2Cur == vectorData->ctx->bN2End_) && (gS1Cur == vectorData->ctx->gS1OEnd_)) {
                vectorData->tailS2Split = vectorData->ctx->s2OEnd_ != 0U;
                vectorData->ctx->curS2End = vectorData->ctx->s2OEnd_;
            }
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparseVector(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIV {
            int64_t preTokenLeftUp = 0;
            int64_t nextTokenLeftUp = 0;
            GetPreNextTokenLeftUpVector(vectorData->ctx->actSeqLensQ, vectorData->ctx->actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

            int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(VectorData::mBaseSize);
            int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(VectorData::mBaseSize),
                                        static_cast<int64_t>(vectorData->ctx->actSeqLensQ) * static_cast<int64_t>(vectorData->ctx->tilingData->flashAttnBaseParams.gSize)) -
                                    1;
            int64_t s1FirstToken = 0;
            int64_t s1LastToken = 0;
            if constexpr (GetOutUbFormat<VectorData::LAYOUT_Q>() == UbFormat::S1G) {
                s1FirstToken = static_cast<int64_t>(s1GFirstToken / vectorData->ctx->tilingData->flashAttnBaseParams.gSize);
                s1LastToken = static_cast<int64_t>(s1GLastToken / vectorData->ctx->tilingData->flashAttnBaseParams.gSize);
            } else {
                if (s1GFirstToken / static_cast<int64_t>(vectorData->ctx->actSeqLensQ) == s1GLastToken / static_cast<int64_t>(vectorData->ctx->actSeqLensQ)) {
                    s1FirstToken = s1GFirstToken % static_cast<int64_t>(vectorData->ctx->actSeqLensQ);
                    s1LastToken = s1GLastToken % static_cast<int64_t>(vectorData->ctx->actSeqLensQ);
                } else {
                    s1FirstToken = 0;
                    s1LastToken = static_cast<int64_t>(vectorData->ctx->actSeqLensQ);
                }
            }

            uint32_t s2StartWithSparse = 0U;
            uint32_t s2EndWithSparse = 0U;
            int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
            int64_t s2LastToken = s1LastToken + nextTokenLeftUp;

            if (s2FirstToken >= static_cast<int64_t>(vectorData->ctx->actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
                vectorData->ctx->curS2Start = 0U;
                vectorData->ctx->curS2End = 0U;
                return;
            }

            s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(vectorData->ctx->actSeqLensKv - 1));
            s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(vectorData->ctx->actSeqLensKv - 1));

            s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / VectorData::s2BaseSize;
            s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / VectorData::s2BaseSize + 1U;

            vectorData->ctx->curS2Start = s2StartWithSparse;
            vectorData->ctx->curS2End = s2EndWithSparse;

            if (bN2Cur == vectorData->bN2Start_ && gS1Cur == vectorData->gS1OStart_) {
                vectorData->headS2Split = vectorData->s2OStart_ > s2StartWithSparse ? true : false;
                vectorData->ctx->curS2Start = Max(s2StartWithSparse, vectorData->s2OStart_);
            }
            if (bN2Cur == vectorData->ctx->bN2End_ && gS1Cur == vectorData->ctx->gS1OEnd_) {
                vectorData->tailS2Split = vectorData->ctx->s2OEnd_ > 0U ? true : false;
                vectorData->ctx->curS2End = vectorData->ctx->s2OEnd_ > 0U ? Min(s2EndWithSparse, vectorData->ctx->s2OEnd_) : s2EndWithSparse;
            }
        }
    }

    __aicore__ inline void ExecuteTaskVector(uint64_t loop, RunInfoXVec taskRunInfoVec[VectorData::PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIV {
            RunInfoXVec &runInfo0 = taskRunInfoVec[loop % VectorData::PRELOAD_TASK_CACHE_SIZE];
            RunInfoXVec &runInfoNegN = taskRunInfoVec[(loop - VectorData::PRELOAD_N) % VectorData::PRELOAD_TASK_CACHE_SIZE];
            if (runInfo0.isValid) {
                ComputeVec1Vector(runInfo0);
            }
            if (loop >= VectorData::PRELOAD_N) {
                if (runInfoNegN.isValid) {
                    ComputeVec2Vector(runInfoNegN);
                    runInfoNegN.isValid = false;
                }
            }
        }
    }

    __aicore__ inline void ComputeVec1Vector(RunInfoXVec &runInfoVec)
    {
        if ASCEND_IS_AIV {
            vectorData->vecFaBlock.ProcessVec1(vectorData->ctx->l1PBuffers.Get(), vectorData->ctx->bmm1Buffers.Get(), runInfoVec);
        }
    }

    __aicore__ inline void ComputeVec2Vector(RunInfoXVec &runInfoVec)
    {
        if ASCEND_IS_AIV {
            using VecMm2ResPos = typename VectorData::VecFaBlockTypeAlias::mm2ResPos;
            using UbBufferType = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
            if constexpr (std::is_same_v<VecMm2ResPos, UbBufferType>) {
                vectorData->vecFaBlock.ProcessVec2(vectorData->ctx->bmm2Buffers.Get(), runInfoVec);
            } else {
                vectorData->vecFaBlock.ProcessVec2(vectorData->ctx->bmm2ResGmBuffers.Get(), runInfoVec);
            }
        }
    }

    __aicore__ inline void CreateTaskVector(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                            RunInfoXVec taskRunInfoVec[VectorData::PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIV {
            RunInfoXVec &runInfo = taskRunInfoVec[loop % VectorData::PRELOAD_TASK_CACHE_SIZE];
            CalcParamsVector(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
            runInfo.isValid = true;
        }
    }

    __aicore__ inline void CalcParamsVector(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoXVec &info)
    {
        if ASCEND_IS_AIV {
            info.loop = loop;
            info.mloop = vectorData->mloop;
            info.bIdx = bN2Cur / vectorData->ctx->tilingData->flashAttnBaseParams.n2Size;
            info.n2Idx = bN2Cur % vectorData->ctx->tilingData->flashAttnBaseParams.n2Size;
            info.gS1Idx = gS1Cur * VectorData::mBaseSize;
            info.s2Idx = s2Cur * VectorData::s2BaseSize;
            info.actS1Size = vectorData->ctx->actSeqLensQ;
            info.actS2Size = vectorData->ctx->actSeqLensKv;

            info.actMSize = VectorData::mBaseSize;
            uint64_t gS1Size = info.actS1Size * vectorData->ctx->tilingData->flashAttnBaseParams.gSize;
            if (((gS1Cur + 1) * VectorData::mBaseSize) > gS1Size) {
                info.actMSize = gS1Size - gS1Cur * VectorData::mBaseSize;
            }
            info.actSingleLoopS2Size = VectorData::s2BaseSize;
            if (((s2Cur + 1) * VectorData::s2BaseSize) > info.actS2Size) {
                info.actSingleLoopS2Size = info.actS2Size - s2Cur * VectorData::s2BaseSize;
            }
            info.actSingleLoopS2SizeAlign =
                Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(typename VectorData::INPUT_T)));

            GetPreNextTokenLeftUpVector(vectorData->ctx->actSeqLensQ, vectorData->ctx->actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

            info.isFirstS2Loop = ((loop == 0) || (s2Cur == vectorData->ctx->curS2Start));
            info.isS2SplitCore = false;
            info.faTmpOutWsPos = vectorData->coreFirstTmpOutWsPos_;
            info.isLastS2Loop = (s2Cur + 1 == vectorData->ctx->curS2End);

            info.actVecMSize = (info.actMSize + 1) >> 1;
            info.vecMbaseIdx = 0;
            if (vectorData->ctx->constInfo.subBlockIdx == 1) {
                info.vecMbaseIdx = info.actVecMSize;
                info.actVecMSize = info.actMSize - info.actVecMSize;
            }

            info.gIdx = 0;
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;

            if (vectorData->bN2Start_ == vectorData->ctx->bN2End_ && vectorData->gS1OStart_ == vectorData->ctx->gS1OEnd_) {
                info.isS2SplitCore = true;
            } else {
                if (vectorData->headS2Split && (bN2Cur == vectorData->bN2Start_) && (gS1Cur == vectorData->gS1OStart_)) {
                    info.isS2SplitCore = true;
                } else if (vectorData->tailS2Split && (bN2Cur == vectorData->ctx->bN2End_) && (gS1Cur == vectorData->ctx->gS1OEnd_)) {
                    info.isS2SplitCore = true;
                    info.faTmpOutWsPos = vectorData->headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
                }
            }
        }
    }

    __aicore__ inline void UpdateAxisInfoVector(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                                uint32_t &s2Cur)
    {
        if ASCEND_IS_AIV {
            uint64_t s2LoopTimes = (vectorData->ctx->actSeqLensKv + VectorData::s2BaseSize - 1) / VectorData::s2BaseSize;
            uint64_t gS1Size = vectorData->ctx->actSeqLensQ * vectorData->ctx->tilingData->flashAttnBaseParams.gSize;
            uint64_t gS1LoopTimes = (gS1Size + VectorData::mBaseSize - 1) / VectorData::mBaseSize;

            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }

            if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK || taskDealMode == TASK_DEAL_MODE::S2_END ||
                taskDealMode == TASK_DEAL_MODE::SKIP_S1OUT) {
                vectorData->accGS1Loops++;
            }

            s2Cur = 0;
            if (gS1Cur + 1 < gS1LoopTimes) {
                gS1Cur++;
                return;
            }

            if ((taskDealMode == TASK_DEAL_MODE::DEAL_ZERO || taskDealMode == TASK_DEAL_MODE::SKIP_ZERO)) {
                vectorData->accGS1Loops++;
            }

            gS1Cur = 0;
            bN2Cur++;
        }
    }

    __aicore__ inline void FlashDecodeVector(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIV {
            if (!vectorData->isFd) {
                return;
            }
            vectorData->vecFdBlock.GetFDSectionInfo(sectionIdx);
            vectorData->vecFdBlock.InitBuffers(vectorData->ctx->pipe, vectorData->SharedBuffer1, vectorData->SharedBuffer2, vectorData->SharedBuffer3);
            AscendC::ICachePreLoad(2);
            vectorData->vecFdBlock.AllocEventID();
            vectorData->vecFdBlock.InitDecodeParams();
            SyncAll();
            vectorData->vecFdBlock.FlashDecode();
            SyncAll();
            vectorData->vecFdBlock.FreeBuffers(vectorData->SharedBuffer1, vectorData->SharedBuffer2);
            vectorData->vecFdBlock.FreeEventID();
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_VECTOR_H_