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
 * \brief Flash Attention GQA vector processor (fully independent)
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_VECTOR_H_
#define FLASH_ATTENTION_NOQUANT_GQA_VECTOR_H_

#include "flash_attn_kernel_base_gqa_common.h"

namespace BaseApi {

template <typename VecFaBlockType, typename VecFdBlockType, typename InputT, LayOutTypeEnum LayoutQ = LayOutTypeEnum::None, bool PageAttention = false>
class FlashAttentionVectorProcessor {
public:
    using VecFaBlockTypeAlias = VecFaBlockType;
    static constexpr uint32_t mBaseSize = VecFaBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = VecFaBlockType::s2BaseSize;

    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr LayOutTypeEnum LAYOUT_Q = LayoutQ;
    static constexpr LayOutTypeEnum LAYOUT_KV = LayoutQ;
    static constexpr bool PAGE_ATTENTION = PageAttention;

    static constexpr bool IS_TND_LAYOUT = (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND || LAYOUT_Q == LayOutTypeEnum::LAYOUT_NTD);

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = InputT;
    using ConstInfoX = typename VecFaBlockType::ConstInfoX;

    using QCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                           ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                           ActualSeqLensParser<Q_MODE, int32_t>>::type;
    using KvCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                            ActualSeqLensParser<KV_MODE, int32_t, true>,
                                                            ActualSeqLensParser<KV_MODE, int32_t>>::type;
    using QSeqUsedParserType = QCuSeqLensParserType;
    using KvSeqUsedParserType = KvCuSeqLensParserType;

    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

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

    bool isFd = false;
    uint32_t coreFirstTmpOutWsPos_ = 0;
    TQue<QuePosition::VECOUT, 1> SharedBuffer1[2];
    TQue<QuePosition::VECIN, 1> SharedBuffer2[2];
    TBuf<> SharedBuffer3;

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

    __aicore__ inline FlashAttentionVectorProcessor()
        : vecFaBlock(constInfo), vecFdBlock(constInfo), tilingData(nullptr), keyPtr(nullptr), pipe(nullptr), sectionNum(0) {}

    __aicore__ inline void InitConstInfo()
    {
        constInfo.aivIdx = GetBlockIdx();
        constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
        constInfo.subBlockIdx = GetSubBlockIdx();
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
        using MM_T = float;
        using VecMm2ResPos = typename VecFaBlockTypeAlias::mm2ResPos;
        using UbBufferType = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
        constexpr bool BMM2_TOUB_ALIAS = std::is_same_v<VecMm2ResPos, UbBufferType>;

        uint32_t mm1OutDtype = sizeof(MM_T);
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        uint32_t dVBaseSizeVal = Align64Func((uint16_t)tilingData->flashAttnBaseParams.dSizeV);
        uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSizeVal * sizeof(MM_T);
        uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);

        l1BufferManager.Init(pipe, 524286);
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB_ALIAS) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2);
        }
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
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

    __aicore__ inline void InitVectorBlock(__gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                           __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                           __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                           __gm__ uint8_t *workspace, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv)
    {
        vecFaBlock.isFd = isFd;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            vecFaBlock.InitVecBlock(pipe, cuSeqLensQ, cuSeqLensKv, attenMask,
                                     learnableSink, softmaxLse, attentionOut, workspace, tilingData);
            vecFaBlock.SetCuSeqLensParser(qCuSeqLensParser);
        } else {
            vecFaBlock.InitVecBlock(pipe, seqUsedQ, seqUsedKv, attenMask, learnableSink,
                                     softmaxLse, attentionOut, workspace, tilingData);
            vecFaBlock.SetCuSeqLensParser(qSeqUsedParser);
        }
        vecFaBlock.ClearOutput();

        if (isFd) {
            vecFdBlock.SetTilingData(tilingData);
            vecFdBlock.InitParams();
            vecFdBlock.InitGlobalTensor(vecFaBlock.softmaxFDMaxGm, vecFaBlock.softmaxFDSumGm,
                                         vecFaBlock.accumOutGm, vecFaBlock.attentionOutGm, keyPtr);
            if (learnableSink != nullptr) {
                GlobalTensor<INPUT_T> sinkGm;
                sinkGm.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
                vecFaBlock.learnableSinkFlag = true;
                vecFdBlock.InitLearnableSinkGm(sinkGm);
            }
            if (tilingData->flashAttnBaseParams.isSoftMaxLseEnable) {
                GlobalTensor<float> softmaxLseGm;
                softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
            }
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                vecFdBlock.SetCuSeqLensParsers(qCuSeqLensParser, kvCuSeqLensParser);
            } else {
                vecFdBlock.SetCuSeqLensParsers(qSeqUsedParser, kvSeqUsedParser);
            }
        }
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            using VecMm2ResPos = typename VecFaBlockTypeAlias::mm2ResPos;
            using UbBufferType = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
            if constexpr (std::is_same_v<VecMm2ResPos, UbBufferType>) {
                bmm2Buffers.Get().SetCrossCore();
                bmm2Buffers.Get().SetCrossCore();
            }
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void ProcessVector(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIV {
            vecFaBlock.InitBuffers(SharedBuffer1, SharedBuffer2, SharedBuffer3);
            vecFaBlock.AllocEventID();
            if (constInfo.aicIdx < tilingData->flashAttnBaseParams.coreNum) {
                FlashAttentionVector(sectionIdx);
            }
            FlashDecodeVector(sectionIdx);
            vecFaBlock.FreeEventID();
        }
    }

private:
    __aicore__ inline void FlashAttentionVector(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIV {
            if (constInfo.aicIdx >= tilingData->flashAttnBaseParams.coreNum) {
                return;
            }

            GetFASectionInfo(sectionIdx);
            RunInfoXVec taskRunInfoVec[PRELOAD_TASK_CACHE_SIZE] = {};

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

            coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
                GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));

            bool shouldDispatchTask = true;
            bool shouldExecuteTask = false;
            while (shouldDispatchTask || shouldExecuteTask) {
                shouldDispatchTask = ShouldDispatchTaskVector(bN2Cur, gS1Cur, s2Cur);
                if (shouldDispatchTask) {
                    TASK_DEAL_MODE taskDealMode = GetTaskDealModeVector(bN2Cur, gS1Cur, s2Cur);
                    if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                        CreateTaskVector(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfoVec);
                        createdTaskCount++;
                        UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                        vecFaBlock.DealZeroActSeqLen(bN2Cur);
                        UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                        continue;
                    } else {
                        UpdateAxisInfoVector(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                        continue;
                    }
                }
                shouldExecuteTask = ShouldExecuteTaskVector(taskRunInfoVec);
                if (shouldExecuteTask) {
                    ExecuteTaskVector(executedTaskCount, taskRunInfoVec);
                    executedTaskCount++;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTaskVector(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if ASCEND_IS_AIV {
            return ((bN2Cur != bN2End_) || (gS1Cur != gS1OEnd_) || (s2Cur != s2OEnd_));
        }
        return false;
    }

    __aicore__ inline bool ShouldExecuteTaskVector(RunInfoXVec taskRunInfoVec[PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < PRELOAD_TASK_CACHE_SIZE; i++) {
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
                    CalcCurS2StartEndNoSparseVector(bN2Cur, gS1Cur);
                } else {
                    CalcCurS2StartEndWithSparseVector(bN2Cur, gS1Cur);
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

    __aicore__ inline void GetPreNextTokenLeftUpVector(int64_t actSeqLensQ, int64_t actSeqLensKv,
                                                        int64_t &preTokenLeftUp, int64_t &nextTokenLeftUp)
    {
        if ASCEND_IS_AIV {
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

    __aicore__ inline void CalcCurS2StartEndNoSparseVector(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIV {
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

    __aicore__ inline void CalcCurS2StartEndWithSparseVector(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        if ASCEND_IS_AIV {
            int64_t preTokenLeftUp = 0;
            int64_t nextTokenLeftUp = 0;
            GetPreNextTokenLeftUpVector(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

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

    __aicore__ inline void ExecuteTaskVector(uint64_t loop, RunInfoXVec taskRunInfoVec[PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIV {
            RunInfoXVec &runInfo0 = taskRunInfoVec[loop % PRELOAD_TASK_CACHE_SIZE];
            RunInfoXVec &runInfoNegN = taskRunInfoVec[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfo0.isValid) {
                ComputeVec1Vector(runInfo0);
            }
            if (loop >= PRELOAD_N) {
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
            vecFaBlock.ProcessVec1(l1PBuffers.Get(), bmm1Buffers.Get(), runInfoVec);
        }
    }

    __aicore__ inline void ComputeVec2Vector(RunInfoXVec &runInfoVec)
    {
        if ASCEND_IS_AIV {
            using VecMm2ResPos = typename VecFaBlockTypeAlias::mm2ResPos;
            using UbBufferType = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
            if constexpr (std::is_same_v<VecMm2ResPos, UbBufferType>) {
                vecFaBlock.ProcessVec2(bmm2Buffers.Get(), runInfoVec);
            } else {
                vecFaBlock.ProcessVec2(bmm2ResGmBuffers.Get(), runInfoVec);
            }
        }
    }

    __aicore__ inline void CreateTaskVector(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                            RunInfoXVec taskRunInfoVec[PRELOAD_TASK_CACHE_SIZE])
    {
        if ASCEND_IS_AIV {
            RunInfoXVec &runInfo = taskRunInfoVec[loop % PRELOAD_TASK_CACHE_SIZE];
            CalcParamsVector(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
            runInfo.isValid = true;
        }
    }

    __aicore__ inline void CalcParamsVector(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoXVec &info)
    {
        if ASCEND_IS_AIV {
            info.loop = loop;
            info.mloop = mloop;
            info.bIdx = bN2Cur / tilingData->flashAttnBaseParams.n2Size;
            info.n2Idx = bN2Cur % tilingData->flashAttnBaseParams.n2Size;
            info.gS1Idx = gS1Cur * mBaseSize;
            info.s2Idx = s2Cur * s2BaseSize;
            info.actS1Size = actSeqLensQ;
            info.actS2Size = actSeqLensKv;

            info.actMSize = mBaseSize;
            uint64_t gS1Size = info.actS1Size * tilingData->flashAttnBaseParams.gSize;
            if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
                info.actMSize = gS1Size - gS1Cur * mBaseSize;
            }
            info.actSingleLoopS2Size = s2BaseSize;
            if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
                info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
            }
            info.actSingleLoopS2SizeAlign =
                Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

            GetPreNextTokenLeftUpVector(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

            info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
            info.isS2SplitCore = false;
            info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
            info.isLastS2Loop = (s2Cur + 1 == curS2End);

            info.actVecMSize = (info.actMSize + 1) >> 1;
            info.vecMbaseIdx = 0;
            if (constInfo.subBlockIdx == 1) {
                info.vecMbaseIdx = info.actVecMSize;
                info.actVecMSize = info.actMSize - info.actVecMSize;
            }

            info.gIdx = 0;
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;

            if (bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_) {
                info.isS2SplitCore = true;
            } else {
                if (headS2Split && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                    info.isS2SplitCore = true;
                } else if (tailS2Split && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                    info.isS2SplitCore = true;
                    info.faTmpOutWsPos = headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
                }
            }
        }
    }

    __aicore__ inline void UpdateAxisInfoVector(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                                uint32_t &s2Cur)
    {
        if ASCEND_IS_AIV {
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

    __aicore__ inline void FlashDecodeVector(uint32_t sectionIdx)
    {
        if ASCEND_IS_AIV {
            if (!isFd) {
                return;
            }
            vecFdBlock.GetFDSectionInfo(sectionIdx);
            vecFdBlock.InitBuffers(pipe, SharedBuffer1, SharedBuffer2, SharedBuffer3);
            AscendC::ICachePreLoad(2);
            vecFdBlock.AllocEventID();
            vecFdBlock.InitDecodeParams();
            SyncAll();
            vecFdBlock.FlashDecode();
            SyncAll();
            vecFdBlock.FreeBuffers(SharedBuffer1, SharedBuffer2);
            vecFdBlock.FreeEventID();
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_VECTOR_H_