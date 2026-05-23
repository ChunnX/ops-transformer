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

template <typename KernelBase>
class FlashAttentionVectorProcessor : public KernelBase {
public:
    using KernelBase::KernelBase;

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            if constexpr (KernelBase::BMM2_TOUB) {
                this->bmm2Buffers.Get().SetCrossCore();
                this->bmm2Buffers.Get().SetCrossCore();
            }
            this->bmm1Buffers.Get().SetCrossCore();
            this->bmm1Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void InitVector(__gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                      __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                      __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                      __gm__ uint8_t *workspace, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv)
    {
        if ASCEND_IS_AIV {
            this->vecFaBlock.isFd = this->isFd;
            if constexpr (KernelBase::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                this->vecFaBlock.InitVecBlock(this->pipe, cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse,
                                              attentionOut, workspace);
                this->vecFaBlock.SetCuSeqLensParser(this->qCuSeqLensParser);
            } else {
                this->vecFaBlock.InitVecBlock(this->pipe, seqUsedQ, seqUsedKv, attenMask, learnableSink, softmaxLse,
                                              attentionOut, workspace);
                this->vecFaBlock.SetCuSeqLensParser(this->qSeqUsedParser);
            }
            this->vecFaBlock.ClearOutput();
        }

        if (this->isFd) {
            if ASCEND_IS_AIV {
                this->vecFdBlock.InitParams();
                this->vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                                  this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm,
                                                  this->keyPtr);
                if (learnableSink != nullptr) {
                    this->sinkGm.SetGlobalBuffer((__gm__ typename KernelBase::INPUT_T *)learnableSink);
                    this->constInfo.learnableSinkFlag = true;
                    this->vecFdBlock.InitLearnableSinkGm(this->sinkGm);
                }
                if (this->constInfo.isSoftmaxLseEnable) {
                    this->softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    this->vecFdBlock.InitSoftmaxLseGm(this->softmaxLseGm);
                }
                if constexpr (KernelBase::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    this->vecFdBlock.SetCuSeqLensParsers(this->qCuSeqLensParser, this->kvCuSeqLensParser);
                } else {
                    this->vecFdBlock.SetCuSeqLensParsers(this->qSeqUsedParser, this->kvSeqUsedParser);
                }
            }
        }
    }

    __aicore__ inline void ProcessVector(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIV {
            this->vecFaBlock.InitBuffers(this->SharedBuffer1, this->SharedBuffer2, this->SharedBuffer3);
            this->vecFaBlock.AllocEventID();
        }

        if (this->constInfo.aicIdx < this->constInfo.coreNum) {
            FlashAttentionVector(sectionIdx);
        }

        if ASCEND_IS_AIV {
            FlashDecodeVector(sectionIdx);
        }

        if ASCEND_IS_AIV {
            this->vecFaBlock.FreeEventID();
        }
    }

private:
    __aicore__ inline void FlashAttentionVector(uint32_t sectionIdx)
    {
        if (this->constInfo.aicIdx >= this->constInfo.coreNum) {
            return;
        }

        this->GetFASectionInfo(sectionIdx);
        RunInfoVector taskRunInfoXVec[KernelBase::PRELOAD_TASK_CACHE_SIZE] = {};

        this->createdTaskCount = 0;
        this->executedTaskCount = 0;
        this->mloop = 0;
        this->accGS1Loops = 0;
        this->headS2Split = false;
        this->tailS2Split = false;

        uint32_t bN2Cur = this->bN2Start_;
        uint32_t gS1Cur = this->gS1OStart_;
        uint32_t s2Cur = this->s2OStart_;
        this->prevBN2Idx = bN2Cur;
        this->prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        while (shouldDispatchTask || shouldExecuteTask) {
            shouldDispatchTask = ShouldDispatchTaskVector(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealModeVector(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTaskVector(this->createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfoXVec);
                    this->createdTaskCount++;
                    UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        this->vecFaBlock.DealZeroActSeqLen(bN2Cur);
                    }
                    UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            shouldExecuteTask = ShouldExecuteTaskVector(taskRunInfoXVec);
            if (shouldExecuteTask) {
                ExecuteTaskVector(this->executedTaskCount, taskRunInfoXVec);
                this->executedTaskCount++;
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTaskVector(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != this->bN2End_) || (gS1Cur != this->gS1OEnd_) || (s2Cur != this->s2OEnd_));
    }

    __aicore__ inline bool ShouldExecuteTaskVector(RunInfoVector taskRunInfoXVec[KernelBase::PRELOAD_TASK_CACHE_SIZE])
    {
        for (uint32_t i = 0; i < KernelBase::PRELOAD_TASK_CACHE_SIZE; i++) {
            if (taskRunInfoXVec[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealModeVector(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == this->bN2Start_) && (gS1Cur == this->gS1OStart_) && (s2Cur == this->s2OStart_);
        uint32_t bIdx = bN2Cur / this->constInfo.n2Size;
        if (isFirstTask || this->prevBIdx != bIdx) {
            this->prevBIdx = bIdx;
            if (this->constInfo.cuSeqLensKVSize == 0 && this->constInfo.seqUsedKvSize == 0 &&
                !this->constInfo.isKvContinuous) {
                this->actSeqLensKv = SeqLenFromTensorList<KernelBase::LAYOUT_KV>(this->keyPtr, bIdx);
            } else {
                if constexpr (KernelBase::LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                    this->actSeqLensKv = this->kvCuSeqLensParser.GetActualSeqLength(bIdx);
                } else {
                    this->actSeqLensKv = this->kvSeqUsedParser.GetActualSeqLength(bIdx);
                }
            }
            if constexpr (KernelBase::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                this->actSeqLensQ = this->qCuSeqLensParser.GetActualSeqLength(bIdx);
            } else {
                this->actSeqLensQ = this->qSeqUsedParser.GetActualSeqLength(bIdx);
            }
        }
        uint64_t s2LoopTimes = (this->actSeqLensKv + KernelBase::s2BaseSize - 1) / KernelBase::s2BaseSize;
        uint64_t gS1Size = this->actSeqLensQ * this->constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + KernelBase::mBaseSize - 1) / KernelBase::mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }

        if (isFirstTask || bN2Cur != this->prevBN2Idx || gS1Cur != this->prevGS1Idx) {
            if constexpr (!KernelBase::HAS_MASK) {
                CalcCurS2StartEndNoSparseVector(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparseVector(bN2Cur, gS1Cur);
            }
            this->prevBN2Idx = bN2Cur;
            this->prevGS1Idx = gS1Cur;
        }

        if (this->curS2Start < this->curS2End && s2Cur >= this->curS2End) {
            return TASK_DEAL_MODE::S2_END;
        }

        if (s2Cur < this->curS2Start || s2Cur >= this->curS2End) {
            return TASK_DEAL_MODE::SKIP;
        }

        if (s2Cur == this->curS2Start) {
            this->mloop++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUpVector(int64_t actSeqLensQ, int64_t actSeqLensKv,
                                                       int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = this->constInfo.preTokens;
        nextTokenLeftUp = this->constInfo.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        this->constInfo.sparseMode);

        if (this->constInfo.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (this->constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (this->constInfo.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparseVector(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        this->curS2Start = 0U;
        this->curS2End = (static_cast<uint32_t>(this->actSeqLensKv) + KernelBase::s2BaseSize - 1) / KernelBase::s2BaseSize;

        if ((bN2Cur == this->bN2Start_) && (gS1Cur == this->gS1OStart_)) {
            this->headS2Split = this->s2OStart_ != 0U;
            this->curS2Start = this->s2OStart_;
        }

        if ((bN2Cur == this->bN2End_) && (gS1Cur == this->gS1OEnd_)) {
            this->tailS2Split = this->s2OEnd_ != 0U;
            this->curS2End = this->s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparseVector(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUpVector(this->actSeqLensQ, this->actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(KernelBase::mBaseSize);
        int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(KernelBase::mBaseSize),
                                   static_cast<int64_t>(this->actSeqLensQ) * static_cast<int64_t>(this->constInfo.gSize)) -
                               1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<KernelBase::LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / this->constInfo.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / this->constInfo.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(this->actSeqLensQ) == s1GLastToken / static_cast<int64_t>(this->actSeqLensQ)) {
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(this->actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(this->actSeqLensQ);
            } else {
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(this->actSeqLensQ);
            }
        }

        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;

        if (s2FirstToken >= static_cast<int64_t>(this->actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            this->curS2Start = 0U;
            this->curS2End = 0U;
            return;
        }

        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(this->actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(this->actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / KernelBase::s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / KernelBase::s2BaseSize + 1U;

        this->curS2Start = s2StartWithSparse;
        this->curS2End = s2EndWithSparse;

        if (bN2Cur == this->bN2Start_ && gS1Cur == this->gS1OStart_) {
            this->headS2Split = this->s2OStart_ > s2StartWithSparse ? true : false;
            this->curS2Start = Max(s2StartWithSparse, this->s2OStart_);
        }
        if (bN2Cur == this->bN2End_ && gS1Cur == this->gS1OEnd_) {
            this->tailS2Split = this->s2OEnd_ > 0U ? true : false;
            this->curS2End = this->s2OEnd_ > 0U ? Min(s2EndWithSparse, this->s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTaskVector(uint64_t loop, RunInfoVector taskRunInfoXVec[KernelBase::PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoVector &runInfoXVec0 = taskRunInfoXVec[loop % KernelBase::PRELOAD_TASK_CACHE_SIZE];
        RunInfoVector &runInfoXVecNegN = taskRunInfoXVec[(loop - KernelBase::PRELOAD_N) % KernelBase::PRELOAD_TASK_CACHE_SIZE];
if (runInfoXVec0.isValid) {
            if ASCEND_IS_AIV {
                ComputeVec1Vector(runInfoXVec0);
            }
        }
        if (loop >= KernelBase::PRELOAD_N) {
            if (runInfoXVecNegN.isValid) {
                if ASCEND_IS_AIV {
                    ComputeVec2Vector(runInfoXVecNegN);
                }
                runInfoXVecNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeVec1Vector(RunInfoVector &runInfoXVec)
    {
        this->vecFaBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfoXVec);
    }

    __aicore__ inline void ComputeVec2Vector(RunInfoVector &runInfoXVec)
    {
        if constexpr (KernelBase::BMM2_TOUB) {
            this->vecFaBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfoXVec);
        } else {
            this->vecFaBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfoXVec);
        }
    }

    __aicore__ inline void CreateTaskVector(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                            RunInfoVector taskRunInfoXVec[KernelBase::PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoVector &runInfoXVec = taskRunInfoXVec[loop % KernelBase::PRELOAD_TASK_CACHE_SIZE];
        CalcParamsVector(loop, bN2Cur, gS1Cur, s2Cur, runInfoXVec);
        runInfoXVec.isValid = true;
    }

    __aicore__ inline void CalcParamsVector(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                            RunInfoVector &runInfoXVec)
    {
        if ASCEND_IS_AIV {
            runInfoXVec.loop = loop;
            runInfoXVec.mloop = this->mloop;
            runInfoXVec.bIdx = bN2Cur / this->constInfo.n2Size;
            runInfoXVec.n2Idx = bN2Cur % this->constInfo.n2Size;
            runInfoXVec.gS1Idx = gS1Cur * KernelBase::mBaseSize;
            runInfoXVec.gIdx = 0;
            if constexpr (KernelBase::LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH ||
                          KernelBase::LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                          KernelBase::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                runInfoXVec.s1Idx = runInfoXVec.gS1Idx / this->constInfo.gSize;
            } else {
                runInfoXVec.s1Idx = runInfoXVec.gS1Idx % this->actSeqLensQ;
            }
            runInfoXVec.s2Idx = s2Cur * KernelBase::s2BaseSize;

            runInfoXVec.actS1Size = this->actSeqLensQ;
            runInfoXVec.actS2Size = this->actSeqLensKv;
            runInfoXVec.actMSize = KernelBase::mBaseSize;
            uint64_t gS1Size = runInfoXVec.actS1Size * this->constInfo.gSize;
            if (((gS1Cur + 1) * KernelBase::mBaseSize) > gS1Size) {
                runInfoXVec.actMSize = gS1Size - gS1Cur * KernelBase::mBaseSize;
            }
            runInfoXVec.actSingleLoopS2Size = KernelBase::s2BaseSize;
            if (((s2Cur + 1) * KernelBase::s2BaseSize) > runInfoXVec.actS2Size) {
                runInfoXVec.actSingleLoopS2Size = runInfoXVec.actS2Size - s2Cur * KernelBase::s2BaseSize;
            }
            runInfoXVec.actSingleLoopS2SizeAlign =
                Align((uint32_t)runInfoXVec.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(typename KernelBase::INPUT_T)));

            runInfoXVec.isFirstS2Loop = ((loop == 0) || (s2Cur == this->curS2Start));
            runInfoXVec.isLastS2Loop = (s2Cur + 1 == this->curS2End);

            GetPreNextTokenLeftUpVector(this->actSeqLensQ, this->actSeqLensKv, runInfoXVec.preTokensLeftUp, runInfoXVec.nextTokensLeftUp);

            if constexpr (KernelBase::USE_DN) {
                runInfoXVec.actMSizeAlign32 = (runInfoXVec.actMSize + 31) >> 5 << 5;
                runInfoXVec.actVecMSize = runInfoXVec.actMSize <= 16 ? runInfoXVec.actMSize : (runInfoXVec.actMSizeAlign32 >> 1);
            } else {
                runInfoXVec.actVecMSize = (runInfoXVec.actMSize + 1) >> 1;
            }
            runInfoXVec.vecMbaseIdx = 0;
            if (this->constInfo.subBlockIdx == 1) {
                runInfoXVec.vecMbaseIdx = runInfoXVec.actVecMSize;
                runInfoXVec.actVecMSize = runInfoXVec.actMSize - runInfoXVec.actVecMSize;
            }

            runInfoXVec.isS2SplitCore = false;
            runInfoXVec.faTmpOutWsPos = this->coreFirstTmpOutWsPos_;
            if (this->bN2Start_ == this->bN2End_ && this->gS1OStart_ == this->gS1OEnd_) {
                runInfoXVec.isS2SplitCore = true;
            } else {
                if (this->headS2Split && (bN2Cur == this->bN2Start_) && (gS1Cur == this->gS1OStart_)) {
                    runInfoXVec.isS2SplitCore = true;
                } else if (this->tailS2Split && (bN2Cur == this->bN2End_) && (gS1Cur == this->gS1OEnd_)) {
                    runInfoXVec.isS2SplitCore = true;
                    runInfoXVec.faTmpOutWsPos = this->headS2Split ? (runInfoXVec.faTmpOutWsPos + 1) : runInfoXVec.faTmpOutWsPos;
                }
            }
        }
    }

    __aicore__ inline void UpdateAxisInfoVector(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                                uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (this->actSeqLensKv + KernelBase::s2BaseSize - 1) / KernelBase::s2BaseSize;
        uint64_t gS1Size = this->actSeqLensQ * this->constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + KernelBase::mBaseSize - 1) / KernelBase::mBaseSize;

        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }

        if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK || taskDealMode == TASK_DEAL_MODE::S2_END ||
            taskDealMode == TASK_DEAL_MODE::SKIP_S1OUT) {
            this->accGS1Loops++;
        }

        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        if ((taskDealMode == TASK_DEAL_MODE::DEAL_ZERO || taskDealMode == TASK_DEAL_MODE::SKIP_ZERO)) {
            this->accGS1Loops++;
        }

        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecodeVector(uint32_t sectionIdx)
    {
        if (!this->isFd) {
            return;
        }
        this->GetFDSectionInfo(sectionIdx);
        this->vecFdBlock.InitBuffers(this->pipe, this->SharedBuffer1, this->SharedBuffer2, this->SharedBuffer3);
        AscendC::ICachePreLoad(2);
        this->vecFdBlock.AllocEventID();
        this->vecFdBlock.InitDecodeParams();
        SyncAll();
        this->vecFdBlock.FlashDecode(this->fdParams_);
        SyncAll();
        this->vecFdBlock.FreeBuffers(this->SharedBuffer1, this->SharedBuffer2);
        this->vecFdBlock.FreeEventID();
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_VECTOR_H_