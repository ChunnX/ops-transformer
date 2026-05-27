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

template <typename CubeData>
class FlashAttentionCubeProcessor {
public:
    CubeData* cubeData;

    __aicore__ inline FlashAttentionCubeProcessor(CubeData* data) : cubeData(data){};

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            cubeData->ctx->bmm1Buffers.Get().WaitCrossCore();
            cubeData->ctx->bmm1Buffers.Get().WaitCrossCore();
            if constexpr (CubeData::BMM2_TOUB) {
                cubeData->ctx->bmm2Buffers.Get().WaitCrossCore();
                cubeData->ctx->bmm2Buffers.Get().WaitCrossCore();
            }
        }
    }

    __aicore__ inline void ProcessCube(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIC {
            cubeData->cubeBlock.InitBuffers();
            cubeData->cubeBlock.AllocEventID();
            if (cubeData->constInfo.aicIdx < cubeData->ctx->tilingData->flashAttnBaseParams.coreNum) {
                FlashAttentionCube(sectionIdx);
            }
            cubeData->cubeBlock.FreeEventID();
        }
    }

private:
    __aicore__ inline void FlashAttentionCube(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIC {
            if (cubeData->constInfo.aicIdx >= cubeData->ctx->tilingData->flashAttnBaseParams.coreNum) {
                return;
            }

            cubeData->GetFASectionInfo(sectionIdx);
            RunInfoXCube taskRunInfoCube[CubeData::PRELOAD_TASK_CACHE_SIZE] = {};

            cubeData->createdTaskCount = 0;
            cubeData->executedTaskCount = 0;
            cubeData->mloop = 0;
            cubeData->accGS1Loops = 0;
            cubeData->headS2Split = false;
            cubeData->tailS2Split = false;

            uint32_t bN2Cur = cubeData->bN2Start_;
            uint32_t gS1Cur = cubeData->gS1OStart_;
            uint32_t s2Cur = cubeData->s2OStart_;
            cubeData->prevBN2Idx = bN2Cur;
            cubeData->prevGS1Idx = gS1Cur;

            bool shouldDispatchTask = true;
            bool shouldExecuteTask = false;
            while (shouldDispatchTask || shouldExecuteTask) {
                shouldDispatchTask = ShouldDispatchTaskCube(bN2Cur, gS1Cur, s2Cur);
                if (shouldDispatchTask) {
                    TASK_DEAL_MODE taskDealMode = GetTaskDealModeCube(bN2Cur, gS1Cur, s2Cur);
                    if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                        CreateTaskCube(cubeData->createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfoCube);
                        cubeData->createdTaskCount++;
                        UpdateAxisInfoCube(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                        UpdateAxisInfoCube(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                        continue;
                    } else {
                        UpdateAxisInfoCube(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                        continue;
                    }
                }
                shouldExecuteTask = ShouldExecuteTaskCube(taskRunInfoCube);
                if (shouldExecuteTask) {
                    ExecuteTaskCube(cubeData->executedTaskCount, taskRunInfoCube);
                    cubeData->executedTaskCount++;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTaskCube(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if ASCEND_IS_AIC {
            return ((bN2Cur != cubeData->ctx->bN2End_) || (gS1Cur != cubeData->ctx->gS1OEnd_) || (s2Cur != cubeData->ctx->s2OEnd_));
        }
        return false;
    }

    __aicore__ inline bool ShouldExecuteTaskCube(RunInfoXCube taskRunInfoCube[CubeData::PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < CubeData::PRELOAD_TASK_CACHE_SIZE; i++) {
                if (taskRunInfoCube[i].isValid) {
                    return true;
                }
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealModeCube(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if ASCEND_IS_AIC {
            bool isFirstTask = (bN2Cur == cubeData->bN2Start_) && (gS1Cur == cubeData->gS1OStart_) && (s2Cur == cubeData->s2OStart_);
            uint32_t bIdx = bN2Cur / cubeData->ctx->tilingData->flashAttnBaseParams.n2Size;
            if (isFirstTask || cubeData->ctx->prevBIdx != bIdx) {
                cubeData->ctx->prevBIdx = bIdx;
                if (cubeData->ctx->tilingData->flashAttnBaseParams.cuSeqLensKVSize == 0 && cubeData->ctx->tilingData->flashAttnBaseParams.seqUsedKvSize == 0 &&
                    !cubeData->ctx->tilingData->flashAttnBaseParams.isKvContinuous) {
                    cubeData->ctx->actSeqLensKv = SeqLenFromTensorList<CubeData::LAYOUT_KV>(cubeData->ctx->keyPtr, bIdx);
                } else {
                    if constexpr (CubeData::LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                        cubeData->ctx->actSeqLensKv = cubeData->kvCuSeqLensParser.GetActualSeqLength(bIdx);
                    } else {
                        cubeData->ctx->actSeqLensKv = cubeData->kvSeqUsedParser.GetActualSeqLength(bIdx);
                    }
                }
                if constexpr (CubeData::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    cubeData->ctx->actSeqLensQ = cubeData->qCuSeqLensParser.GetActualSeqLength(bIdx);
                } else {
                    cubeData->ctx->actSeqLensQ = cubeData->qSeqUsedParser.GetActualSeqLength(bIdx);
                }
            }
            uint64_t s2LoopTimes = (cubeData->ctx->actSeqLensKv + CubeData::s2BaseSize - 1) / CubeData::s2BaseSize;
            uint64_t gS1Size = cubeData->ctx->actSeqLensQ * cubeData->ctx->tilingData->flashAttnBaseParams.gSize;
            uint64_t gS1LoopTimes = (gS1Size + CubeData::mBaseSize - 1) / CubeData::mBaseSize;
            if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
                if (gS1Cur == 0 && s2Cur == 0) {
                    return TASK_DEAL_MODE::DEAL_ZERO;
                }
                return TASK_DEAL_MODE::SKIP_ZERO;
            }

            if (isFirstTask || bN2Cur != cubeData->prevBN2Idx || gS1Cur != cubeData->prevGS1Idx) {
                if (cubeData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == 0) {
                    CalcCurS2StartEndNoSparseCube(bN2Cur, gS1Cur);
                } else {
                    CalcCurS2StartEndWithSparseCube(bN2Cur, gS1Cur);
                }
                cubeData->prevBN2Idx = bN2Cur;
                cubeData->prevGS1Idx = gS1Cur;
            }

            if (cubeData->ctx->curS2Start < cubeData->ctx->curS2End && s2Cur >= cubeData->ctx->curS2End) {
                return TASK_DEAL_MODE::S2_END;
            }

            if (s2Cur < cubeData->ctx->curS2Start || s2Cur >= cubeData->ctx->curS2End) {
                return TASK_DEAL_MODE::SKIP;
            }

            if (s2Cur == cubeData->ctx->curS2Start) {
                cubeData->mloop++;
            }

            return TASK_DEAL_MODE::CREATE_TASK;
        }
        return TASK_DEAL_MODE::SKIP;
    }

    __aicore__ inline void GetPreNextTokenLeftUpCube(int64_t actSeqLensQ, int64_t actSeqLensKv,
                                                      int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp)
    {
        if ASCEND_IS_AIC {
            preTokenLeftUp = cubeData->ctx->tilingData->flashAttnAttenMaskParams.winLefts;
            nextTokenLeftUp = cubeData->ctx->tilingData->flashAttnAttenMaskParams.winRights;
            fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                             cubeData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode);

            if (cubeData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::BAND) {
                preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
            }

            if (cubeData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
                nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
            } else if (cubeData->ctx->tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::BAND) {
                nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
            }
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparseCube(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIC {
            cubeData->ctx->curS2Start = 0U;
            cubeData->ctx->curS2End = (static_cast<uint32_t>(cubeData->ctx->actSeqLensKv) + CubeData::s2BaseSize - 1) / CubeData::s2BaseSize;

            if ((bN2Cur == cubeData->bN2Start_) && (gS1Cur == cubeData->gS1OStart_)) {
                cubeData->headS2Split = cubeData->s2OStart_ != 0U;
                cubeData->ctx->curS2Start = cubeData->s2OStart_;
            }

            if ((bN2Cur == cubeData->ctx->bN2End_) && (gS1Cur == cubeData->ctx->gS1OEnd_)) {
                cubeData->tailS2Split = cubeData->ctx->s2OEnd_ != 0U;
                cubeData->ctx->curS2End = cubeData->ctx->s2OEnd_;
            }
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparseCube(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIC {
            int64_t preTokenLeftUp = 0;
            int64_t nextTokenLeftUp = 0;
            GetPreNextTokenLeftUpCube(cubeData->ctx->actSeqLensQ, cubeData->ctx->actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

            int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(CubeData::mBaseSize);
            int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(CubeData::mBaseSize),
                                        static_cast<int64_t>(cubeData->ctx->actSeqLensQ) * static_cast<int64_t>(cubeData->ctx->tilingData->flashAttnBaseParams.gSize)) -
                                    1;
            int64_t s1FirstToken = 0;
            int64_t s1LastToken = 0;
            if constexpr (GetOutUbFormat<CubeData::LAYOUT_Q>() == UbFormat::S1G) {
                s1FirstToken = static_cast<int64_t>(s1GFirstToken / cubeData->ctx->tilingData->flashAttnBaseParams.gSize);
                s1LastToken = static_cast<int64_t>(s1GLastToken / cubeData->ctx->tilingData->flashAttnBaseParams.gSize);
            } else {
                if (s1GFirstToken / static_cast<int64_t>(cubeData->ctx->actSeqLensQ) == s1GLastToken / static_cast<int64_t>(cubeData->ctx->actSeqLensQ)) {
                    s1FirstToken = s1GFirstToken % static_cast<int64_t>(cubeData->ctx->actSeqLensQ);
                    s1LastToken = s1GLastToken % static_cast<int64_t>(cubeData->ctx->actSeqLensQ);
                } else {
                    s1FirstToken = 0;
                    s1LastToken = static_cast<int64_t>(cubeData->ctx->actSeqLensQ);
                }
            }

            uint32_t s2StartWithSparse = 0U;
            uint32_t s2EndWithSparse = 0U;
            int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
            int64_t s2LastToken = s1LastToken + nextTokenLeftUp;

            if (s2FirstToken >= static_cast<int64_t>(cubeData->ctx->actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
                cubeData->ctx->curS2Start = 0U;
                cubeData->ctx->curS2End = 0U;
                return;
            }

            s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(cubeData->ctx->actSeqLensKv - 1));
            s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(cubeData->ctx->actSeqLensKv - 1));

            s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / CubeData::s2BaseSize;
            s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / CubeData::s2BaseSize + 1U;

            cubeData->ctx->curS2Start = s2StartWithSparse;
            cubeData->ctx->curS2End = s2EndWithSparse;

            if (bN2Cur == cubeData->bN2Start_ && gS1Cur == cubeData->gS1OStart_) {
                cubeData->headS2Split = cubeData->s2OStart_ > s2StartWithSparse ? true : false;
                cubeData->ctx->curS2Start = Max(s2StartWithSparse, cubeData->s2OStart_);
            }
            if (bN2Cur == cubeData->ctx->bN2End_ && gS1Cur == cubeData->ctx->gS1OEnd_) {
                cubeData->tailS2Split = cubeData->ctx->s2OEnd_ > 0U ? true : false;
                cubeData->ctx->curS2End = cubeData->ctx->s2OEnd_ > 0U ? Min(s2EndWithSparse, cubeData->ctx->s2OEnd_) : s2EndWithSparse;
            }
        }
    }

    __aicore__ inline void ExecuteTaskCube(uint64_t loop, RunInfoXCube taskRunInfoCube[CubeData::PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIC {
            RunInfoXCube &runInfo0 = taskRunInfoCube[loop % CubeData::PRELOAD_TASK_CACHE_SIZE];
            RunInfoXCube &runInfoNegN = taskRunInfoCube[(loop - CubeData::PRELOAD_N) % CubeData::PRELOAD_TASK_CACHE_SIZE];
            if (runInfo0.isValid) {
                ComputeMm1Cube(runInfo0);
            }
            if (loop >= CubeData::PRELOAD_N) {
                if (runInfoNegN.isValid) {
                    ComputeMm2Cube(runInfoNegN);
                    runInfoNegN.isValid = false;
                }
            }
        }
    }

    __aicore__ inline void ComputeMm1Cube(RunInfoXCube &runInfoCube)
    {
        if ASCEND_IS_AIC {
            cubeData->cubeBlock.IterateBmm1(cubeData->ctx->bmm1Buffers.Get(), runInfoCube);
        }
    }

    __aicore__ inline void ComputeMm2Cube(RunInfoXCube &runInfoCube)
    {
        if ASCEND_IS_AIC {
            if constexpr (CubeData::BMM2_TOUB) {
                cubeData->cubeBlock.IterateBmm2(cubeData->ctx->bmm2Buffers.Get(), cubeData->ctx->l1PBuffers, runInfoCube);
            } else {
                cubeData->cubeBlock.IterateBmm2(cubeData->ctx->bmm2ResGmBuffers.Get(), cubeData->ctx->l1PBuffers, runInfoCube);
            }
        }
    }

    __aicore__ inline void CreateTaskCube(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                          RunInfoXCube taskRunInfoCube[CubeData::PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIC {
            RunInfoXCube &runInfo = taskRunInfoCube[loop % CubeData::PRELOAD_TASK_CACHE_SIZE];
            CalcParamsCube(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
            runInfo.isValid = true;
        }
    }

    __aicore__ inline void CalcParamsCube(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoXCube &info)
    {
        if ASCEND_IS_AIC {
            info.bIdx = bN2Cur / cubeData->ctx->tilingData->flashAttnBaseParams.n2Size;
            info.n2Idx = bN2Cur % cubeData->ctx->tilingData->flashAttnBaseParams.n2Size;
            info.gS1Idx = gS1Cur * CubeData::mBaseSize;
            info.s2Idx = s2Cur * CubeData::s2BaseSize;

            uint64_t actS1Size = cubeData->ctx->actSeqLensQ;
            uint64_t actS2Size = cubeData->ctx->actSeqLensKv;

            info.actMSize = CubeData::mBaseSize;
            uint64_t gS1Size = actS1Size * cubeData->ctx->tilingData->flashAttnBaseParams.gSize;
            if (((gS1Cur + 1) * CubeData::mBaseSize) > gS1Size) {
                info.actMSize = gS1Size - gS1Cur * CubeData::mBaseSize;
            }
            info.actSingleLoopS2Size = CubeData::s2BaseSize;
            if (((s2Cur + 1) * CubeData::s2BaseSize) > actS2Size) {
                info.actSingleLoopS2Size = actS2Size - s2Cur * CubeData::s2BaseSize;
            }

            if (cubeData->ctx->tilingData->flashAttnBaseParams.isKvContinuous) {
                info.isChangeBatch = false;
            } else {
                if (loop == 0) {
                    info.isChangeBatch = true;
                } else {
                    info.isChangeBatch = (info.n2Idx == 0 && s2Cur == cubeData->ctx->curS2Start);
                }
            }

            info.isFirstS2Loop = ((loop == 0) || (s2Cur == cubeData->ctx->curS2Start));
            info.isLastS2Loop = (s2Cur + 1 == cubeData->ctx->curS2End);
        }
    }

    __aicore__ inline void UpdateAxisInfoCube(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                              uint32_t &s2Cur)
    {
        if ASCEND_IS_AIC {
            uint64_t s2LoopTimes = (cubeData->ctx->actSeqLensKv + CubeData::s2BaseSize - 1) / CubeData::s2BaseSize;
            uint64_t gS1Size = cubeData->ctx->actSeqLensQ * cubeData->ctx->tilingData->flashAttnBaseParams.gSize;
            uint64_t gS1LoopTimes = (gS1Size + CubeData::mBaseSize - 1) / CubeData::mBaseSize;

            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }

            if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK || taskDealMode == TASK_DEAL_MODE::S2_END ||
                taskDealMode == TASK_DEAL_MODE::SKIP_S1OUT) {
                cubeData->accGS1Loops++;
            }

            s2Cur = 0;
            if (gS1Cur + 1 < gS1LoopTimes) {
                gS1Cur++;
                return;
            }

            if ((taskDealMode == TASK_DEAL_MODE::DEAL_ZERO || taskDealMode == TASK_DEAL_MODE::SKIP_ZERO)) {
                cubeData->accGS1Loops++;
            }

            gS1Cur = 0;
            bN2Cur++;
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_CUBE_H_