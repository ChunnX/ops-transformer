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
 * \file flash_attn_kernel_base_gqa_cube.h
 * \brief Flash Attention GQA cube processing methods
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_CUBE_H_
#define FLASH_ATTENTION_NOQUANT_GQA_CUBE_H_

#include "flash_attn_kernel_base_gqa_common.h"

namespace BaseApi {

template <typename KernelBase>
class FlashAttentionCubeProcessor : public KernelBase {
public:
    using KernelBase::KernelBase;

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            this->bmm1Buffers.Get().WaitCrossCore();
            this->bmm1Buffers.Get().WaitCrossCore();
            if constexpr (KernelBase::BMM2_TOUB) {
                this->bmm2Buffers.Get().WaitCrossCore();
                this->bmm2Buffers.Get().WaitCrossCore();
            }
        }
    }

    __aicore__ inline void InitCube(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                    __gm__ uint8_t *blockTable)
    {
        if ASCEND_IS_AIC {
            if constexpr (KernelBase::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                this->cubeBlock.InitCubeBlock(this->pipe, &this->l1BufferManager, query, key, value, blockTable,
                                              this->qCuSeqLensParser, this->kvCuSeqLensParser);
            } else {
                this->cubeBlock.InitCubeBlock(this->pipe, &this->l1BufferManager, query, key, value, blockTable,
                                              this->qSeqUsedParser, this->kvSeqUsedParser);
            }
        }
    }

    __aicore__ inline void ProcessCube(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIC {
            this->cubeBlock.InitBuffers();
            this->cubeBlock.AllocEventID();
        }

        if (this->constInfo.aicIdx < this->constInfo.coreNum) {
            FlashAttentionCube(sectionIdx);
        }

        if ASCEND_IS_AIC {
            this->cubeBlock.FreeEventID();
        }
    }

private:
    __aicore__ inline void FlashAttentionCube(uint32_t sectionIdx)
    {
        if (this->constInfo.aicIdx >= this->constInfo.coreNum) {
            return;
        }

        this->GetFASectionInfo(sectionIdx);
        RunInfoCube taskRunInfoXCube[KernelBase::PRELOAD_TASK_CACHE_SIZE] = {};

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
            shouldDispatchTask = ShouldDispatchTaskCube(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealModeCube(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTaskCube(this->createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfoXCube);
                    this->createdTaskCount++;
                    UpdateAxisInfoCube(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    UpdateAxisInfoCube(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfoCube(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            shouldExecuteTask = ShouldExecuteTaskCube(taskRunInfoXCube);
            if (shouldExecuteTask) {
                ExecuteTaskCube(this->executedTaskCount, taskRunInfoXCube);
                this->executedTaskCount++;
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTaskCube(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != this->bN2End_) || (gS1Cur != this->gS1OEnd_) || (s2Cur != this->s2OEnd_));
    }

    __aicore__ inline bool ShouldExecuteTaskCube(RunInfoCube taskRunInfoXCube[KernelBase::PRELOAD_TASK_CACHE_SIZE])
    {
        for (uint32_t i = 0; i < KernelBase::PRELOAD_TASK_CACHE_SIZE; i++) {
            if (taskRunInfoXCube[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealModeCube(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
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
                CalcCurS2StartEndNoSparseCube(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparseCube(bN2Cur, gS1Cur);
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

    __aicore__ inline void GetPreNextTokenLeftUpCube(int64_t actSeqLensQ, int64_t actSeqLensKv,
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

    __aicore__ inline void CalcCurS2StartEndNoSparseCube(uint32_t bN2Cur, uint32_t gS1Cur)
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

    __aicore__ inline void CalcCurS2StartEndWithSparseCube(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUpCube(this->actSeqLensQ, this->actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

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

    __aicore__ inline void ExecuteTaskCube(uint64_t loop, RunInfoCube taskRunInfoXCube[KernelBase::PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoCube &runInfoXCube0 = taskRunInfoXCube[loop % KernelBase::PRELOAD_TASK_CACHE_SIZE];
        RunInfoCube &runInfoXCubeNegN = taskRunInfoXCube[(loop - KernelBase::PRELOAD_N) % KernelBase::PRELOAD_TASK_CACHE_SIZE];
if (runInfoXCube0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1Cube(runInfoXCube0);
            }
        }
        if (loop >= KernelBase::PRELOAD_N) {
            if (runInfoXCubeNegN.isValid) {
                if ASCEND_IS_AIC {
                    ComputeMm2Cube(runInfoXCubeNegN);
                }
                runInfoXCubeNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1Cube(RunInfoCube &runInfoXCube)
    {
        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfoXCube);
    }

    __aicore__ inline void ComputeMm2Cube(RunInfoCube &runInfoXCube)
    {
        if constexpr (KernelBase::BMM2_TOUB) {
            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfoXCube);
        } else {
            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfoXCube);
        }
    }

    __aicore__ inline void CreateTaskCube(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                          RunInfoCube taskRunInfoXCube[KernelBase::PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoCube &runInfoXCube = taskRunInfoXCube[loop % KernelBase::PRELOAD_TASK_CACHE_SIZE];
        CalcParamsCube(loop, bN2Cur, gS1Cur, s2Cur, runInfoXCube);
        runInfoXCube.isValid = true;
    }

    __aicore__ inline void CalcParamsCube(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoCube &runInfoXCube)
    {
        runInfoXCube.bIdx = bN2Cur / this->constInfo.n2Size;
        runInfoXCube.n2Idx = bN2Cur % this->constInfo.n2Size;
        runInfoXCube.gS1Idx = gS1Cur * KernelBase::mBaseSize;
        runInfoXCube.s2Idx = s2Cur * KernelBase::s2BaseSize;

        uint64_t actS1Size = this->actSeqLensQ;
        uint64_t actS2Size = this->actSeqLensKv;
        runInfoXCube.actMSize = KernelBase::mBaseSize;
        uint64_t gS1Size = actS1Size * this->constInfo.gSize;
        if (((gS1Cur + 1) * KernelBase::mBaseSize) > gS1Size) {
            runInfoXCube.actMSize = gS1Size - gS1Cur * KernelBase::mBaseSize;
        }
        runInfoXCube.actSingleLoopS2Size = KernelBase::s2BaseSize;
        if (((s2Cur + 1) * KernelBase::s2BaseSize) > actS2Size) {
            runInfoXCube.actSingleLoopS2Size = actS2Size - s2Cur * KernelBase::s2BaseSize;
        }

        if (this->constInfo.isKvContinuous) {
            runInfoXCube.isChangeBatch = false;
        } else {
            if (loop == 0) {
                runInfoXCube.isChangeBatch = true;
            } else {
                runInfoXCube.isChangeBatch = (runInfoXCube.n2Idx == 0 && s2Cur == this->curS2Start);
            }
        }

        runInfoXCube.isFirstS2Loop = ((loop == 0) || (s2Cur == this->curS2Start));
        runInfoXCube.isLastS2Loop = (s2Cur + 1 == this->curS2End);
    }

    __aicore__ inline void UpdateAxisInfoCube(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
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
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_CUBE_H_