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
 * \brief Flash Attention GQA cube processor (fully independent)
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_CUBE_H_
#define FLASH_ATTENTION_NOQUANT_GQA_CUBE_H_

#include "flash_attn_kernel_base_gqa_common.h"

namespace BaseApi {

template <typename CubeBlockType>
class FlashAttentionCubeProcessor {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;
    static constexpr bool HAS_PREFIX = CubeBlockType::HAS_PREFIX;

    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;

    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    static constexpr bool IS_TND_LAYOUT = (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND || LAYOUT_Q == LayOutTypeEnum::LAYOUT_NTD);

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    using QCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                           ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>,
                                                           ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t>>::type;
    using KvCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                            ActualSeqLensParser<KV_MODE, int32_t, true>,
                                                            ActualSeqLensParser<KV_MODE, int32_t>>::type;
    using QSeqUsedParserType = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;
    using KvSeqUsedParserType = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;

    CubeBlockType cubeBlock;

    ConstInfo_t<FiaKernelType::NO_QUANT> constInfo;

    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    QCuSeqLensParserType qCuSeqLensParser;
    KvCuSeqLensParserType kvCuSeqLensParser;
    QSeqUsedParserType qSeqUsedParser;
    KvSeqUsedParserType kvSeqUsedParser;

    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData;
    __gm__ uint8_t *keyPtr;
    GlobalTensor<uint32_t> faMetaDataGm;
    TPipe *pipe;
    uint32_t sectionNum;

    uint32_t createdTaskCount = 0U;
    uint32_t executedTaskCount = 0U;
    uint32_t accGS1Loops = 0U;
    uint32_t mloop = 0;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start;
    uint32_t curS2End = 0;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    bool headS2Split = false;
    bool tailS2Split = false;

    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;

    __aicore__ inline FlashAttentionCubeProcessor()
        : cubeBlock(constInfo), tilingData(nullptr), keyPtr(nullptr), pipe(nullptr), sectionNum(0) {}

    __aicore__ inline void InitConstInfo()
    {
        constInfo.aicIdx = GetBlockIdx();
        constInfo.dBasicBlock = Align64Func((uint16_t)tilingData->flashAttnBaseParams.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);

        l1BufferManager.Init(pipe, 524286);
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2);
        }
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);

        if constexpr (!BMM2_TOUB) {
            int64_t mm2ResultSizeCalc = mBaseSize * tilingData->flashAttnBaseParams.dSizeV;
            int64_t prevCoretotalOffset = constInfo.aicIdx * 3 * mm2ResultSizeCalc;
            workspace = workspace + tilingData->flashAttnBaseParams.coreNum * 3 * mm2ResultSizeCalc * sizeof(T);
        }
    }

    __aicore__ inline void InitQCuSeqLensParser(GlobalTensor<int32_t> cuSeqLensGmQ, GlobalTensor<int32_t> seqUsedGmQ)
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            qCuSeqLensParser.Init(cuSeqLensGmQ, seqUsedGmQ,
                                  tilingData->flashAttnBaseParams.cuSeqLensQSize,
                                  tilingData->flashAttnBaseParams.seqUsedQSize);
        } else {
            qSeqUsedParser.Init(seqUsedGmQ,
                                tilingData->flashAttnBaseParams.seqUsedQSize,
                                tilingData->flashAttnBaseParams.s1Size);
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser(GlobalTensor<int32_t> cuSeqLensGmKv, GlobalTensor<int32_t> seqUsedGmKv)
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser.Init(cuSeqLensGmKv, seqUsedGmKv,
                                   tilingData->flashAttnBaseParams.cuSeqLensKVSize,
                                   tilingData->flashAttnBaseParams.seqUsedKvSize);
        } else {
            kvSeqUsedParser.Init(seqUsedGmKv,
                                 tilingData->flashAttnBaseParams.seqUsedKvSize,
                                 tilingData->flashAttnBaseParams.s2Size);
        }
    }

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query, key, value,
                                    blockTable, qCuSeqLensParser, kvCuSeqLensParser, tilingData);
        } else {
            cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query, key, value,
                                    blockTable, qSeqUsedParser, kvSeqUsedParser, tilingData);
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
            if constexpr (BMM2_TOUB) {
                bmm2Buffers.Get().WaitCrossCore();
                bmm2Buffers.Get().WaitCrossCore();
            }
        }
    }

    __aicore__ inline void ProcessCube(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIC {
            cubeBlock.InitBuffers();
            cubeBlock.AllocEventID();
            if (constInfo.aicIdx < tilingData->flashAttnBaseParams.coreNum) {
                FlashAttentionCube(sectionIdx);
            }
            cubeBlock.FreeEventID();
        }
    }

private:
    __aicore__ inline void FlashAttentionCube(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIC {
            if (constInfo.aicIdx >= tilingData->flashAttnBaseParams.coreNum) {
                return;
            }

            GetFASectionInfo(sectionIdx);
            RunInfoXCube taskRunInfoCube[PRELOAD_TASK_CACHE_SIZE] = {};

            createdTaskCount = 0;
            executedTaskCount = 0;
            mloop = 0;
            accGS1Loops = 0;
            headS2Split = false;
            tailS2Split = false;

            uint32_t bN2Cur = bN2Start_;
            uint32_t gS1Cur = gS1OStart_;
            uint32_t s2Cur = s2OStart_;
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;

            bool shouldDispatchTask = true;
            bool shouldExecuteTask = false;
            while (shouldDispatchTask || shouldExecuteTask) {
                shouldDispatchTask = ShouldDispatchTaskCube(bN2Cur, gS1Cur, s2Cur);
                if (shouldDispatchTask) {
                    TASK_DEAL_MODE taskDealMode = GetTaskDealModeCube(bN2Cur, gS1Cur, s2Cur);
                    if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                        CreateTaskCube(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfoCube);
                        createdTaskCount++;
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
                    ExecuteTaskCube(executedTaskCount, taskRunInfoCube);
                    executedTaskCount++;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTaskCube(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if ASCEND_IS_AIC {
            return ((bN2Cur != bN2End_) || (gS1Cur != gS1OEnd_) || (s2Cur != s2OEnd_));
        }
        return false;
    }

    __aicore__ inline bool ShouldExecuteTaskCube(RunInfoXCube taskRunInfoCube[PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < PRELOAD_TASK_CACHE_SIZE; i++) {
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
            bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
            uint32_t bIdx = bN2Cur / tilingData->flashAttnBaseParams.n2Size;
            if (isFirstTask || prevBIdx != bIdx) {
                prevBIdx = bIdx;
                if (tilingData->flashAttnBaseParams.cuSeqLensKVSize == 0 && tilingData->flashAttnBaseParams.seqUsedKvSize == 0 &&
                    !tilingData->flashAttnBaseParams.isKvContinuous) {
                    actSeqLensKv = SeqLenFromTensorList<LAYOUT_KV>(keyPtr, bIdx);
                } else {
                    if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
                        actSeqLensKv = kvCuSeqLensParser.GetActualSeqLength(bIdx);
                    } else {
                        actSeqLensKv = kvSeqUsedParser.GetActualSeqLength(bIdx);
                    }
                }
                if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    actSeqLensQ = qCuSeqLensParser.GetActualSeqLength(bIdx);
                } else {
                    actSeqLensQ = qSeqUsedParser.GetActualSeqLength(bIdx);
                }
            }
            uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
            uint64_t gS1Size = actSeqLensQ * tilingData->flashAttnBaseParams.gSize;
            uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

            if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
                if (gS1Cur == 0 && s2Cur == 0) {
                    return TASK_DEAL_MODE::DEAL_ZERO;
                }
                return TASK_DEAL_MODE::SKIP_ZERO;
            }

            if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
                if (tilingData->flashAttnAttenMaskParams.sparseMode == 0) {
                    CalcCurS2StartEndNoSparseCube(bN2Cur, gS1Cur);
                } else {
                    CalcCurS2StartEndWithSparseCube(bN2Cur, gS1Cur);
                }
                prevBN2Idx = bN2Cur;
                prevGS1Idx = gS1Cur;
            }

            if (curS2Start < curS2End && s2Cur >= curS2End) {
                return TASK_DEAL_MODE::S2_END;
            }

            if (s2Cur < curS2Start || s2Cur >= curS2End) {
                return TASK_DEAL_MODE::SKIP;
            }

            if (s2Cur == curS2Start) {
                mloop++;
            }

            return TASK_DEAL_MODE::CREATE_TASK;
        }
        return TASK_DEAL_MODE::SKIP;
    }

    __aicore__ inline void GetPreNextTokenLeftUpCube(int64_t actSeqLensQ, int64_t actSeqLensKv,
                                                      int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp)
    {
        if ASCEND_IS_AIC {
            preTokenLeftUp = tilingData->flashAttnAttenMaskParams.winLefts;
            nextTokenLeftUp = tilingData->flashAttnAttenMaskParams.winRights;
            fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                             tilingData->flashAttnAttenMaskParams.sparseMode);

            if (tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::BAND) {
                preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
            }

            if (tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
                nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
            } else if (tilingData->flashAttnAttenMaskParams.sparseMode == fa_base_vector::BAND) {
                nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
            }
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparseCube(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIC {
            curS2Start = 0U;
            curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;

            if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                headS2Split = s2OStart_ != 0U;
                curS2Start = s2OStart_;
            }

            if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                tailS2Split = s2OEnd_ != 0U;
                curS2End = s2OEnd_;
            }
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparseCube(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIC {
            int64_t preTokenLeftUp = 0;
            int64_t nextTokenLeftUp = 0;
            GetPreNextTokenLeftUpCube(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

            int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
            int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                        static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(tilingData->flashAttnBaseParams.gSize)) -
                                    1;
            int64_t s1FirstToken = 0;
            int64_t s1LastToken = 0;
            if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
                s1FirstToken = static_cast<int64_t>(s1GFirstToken / tilingData->flashAttnBaseParams.gSize);
                s1LastToken = static_cast<int64_t>(s1GLastToken / tilingData->flashAttnBaseParams.gSize);
            } else {
                if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                    s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                    s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
                } else {
                    s1FirstToken = 0;
                    s1LastToken = static_cast<int64_t>(actSeqLensQ);
                }
            }

            uint32_t s2StartWithSparse = 0U;
            uint32_t s2EndWithSparse = 0U;
            int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
            int64_t s2LastToken = s1LastToken + nextTokenLeftUp;

            if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
                curS2Start = 0U;
                curS2End = 0U;
                return;
            }

            s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
            s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

            s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
            s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

            curS2Start = s2StartWithSparse;
            curS2End = s2EndWithSparse;

            if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) {
                headS2Split = s2OStart_ > s2StartWithSparse ? true : false;
                curS2Start = Max(s2StartWithSparse, s2OStart_);
            }
            if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) {
                tailS2Split = s2OEnd_ > 0U ? true : false;
                curS2End = s2OEnd_ > 0U ? Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
            }
        }
    }

    __aicore__ inline void ExecuteTaskCube(uint64_t loop, RunInfoXCube taskRunInfoCube[PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIC {
            RunInfoXCube &runInfo0 = taskRunInfoCube[loop % PRELOAD_TASK_CACHE_SIZE];
            RunInfoXCube &runInfoNegN = taskRunInfoCube[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfo0.isValid) {
                ComputeMm1Cube(runInfo0);
            }
            if (loop >= PRELOAD_N) {
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
            cubeBlock.IterateBmm1(bmm1Buffers.Get(), runInfoCube);
        }
    }

    __aicore__ inline void ComputeMm2Cube(RunInfoXCube &runInfoCube)
    {
        if ASCEND_IS_AIC {
            if constexpr (BMM2_TOUB) {
                cubeBlock.IterateBmm2(bmm2Buffers.Get(), l1PBuffers, runInfoCube);
            } else {
                cubeBlock.IterateBmm2(bmm2ResGmBuffers.Get(), l1PBuffers, runInfoCube);
            }
        }
    }

    __aicore__ inline void CreateTaskCube(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                          RunInfoXCube taskRunInfoCube[PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIC {
            RunInfoXCube &runInfo = taskRunInfoCube[loop % PRELOAD_TASK_CACHE_SIZE];
            CalcParamsCube(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
            runInfo.isValid = true;
        }
    }

    __aicore__ inline void CalcParamsCube(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoXCube &info)
    {
        if ASCEND_IS_AIC {
            info.bIdx = bN2Cur / tilingData->flashAttnBaseParams.n2Size;
            info.n2Idx = bN2Cur % tilingData->flashAttnBaseParams.n2Size;
            info.gS1Idx = gS1Cur * mBaseSize;
            info.s2Idx = s2Cur * s2BaseSize;

            uint64_t actS1Size = actSeqLensQ;
            uint64_t actS2Size = actSeqLensKv;

            info.actMSize = mBaseSize;
            uint64_t gS1Size = actS1Size * tilingData->flashAttnBaseParams.gSize;
            if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
                info.actMSize = gS1Size - gS1Cur * mBaseSize;
            }
            info.actSingleLoopS2Size = s2BaseSize;
            if (((s2Cur + 1) * s2BaseSize) > actS2Size) {
                info.actSingleLoopS2Size = actS2Size - s2Cur * s2BaseSize;
            }

            if (tilingData->flashAttnBaseParams.isKvContinuous) {
                info.isChangeBatch = false;
            } else {
                if (loop == 0) {
                    info.isChangeBatch = true;
                } else {
                    info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
                }
            }

            info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
            info.isLastS2Loop = (s2Cur + 1 == curS2End);
        }
    }

    __aicore__ inline void UpdateAxisInfoCube(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                              uint32_t &s2Cur)
    {
        if ASCEND_IS_AIC {
            uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
            uint64_t gS1Size = actSeqLensQ * tilingData->flashAttnBaseParams.gSize;
            uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }

            if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK || taskDealMode == TASK_DEAL_MODE::S2_END ||
                taskDealMode == TASK_DEAL_MODE::SKIP_S1OUT) {
                accGS1Loops++;
            }

            s2Cur = 0;
            if (gS1Cur + 1 < gS1LoopTimes) {
                gS1Cur++;
                return;
            }

            if ((taskDealMode == TASK_DEAL_MODE::DEAL_ZERO || taskDealMode == TASK_DEAL_MODE::SKIP_ZERO)) {
                accGS1Loops++;
            }

            gS1Cur = 0;
            bN2Cur++;
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_CUBE_H_