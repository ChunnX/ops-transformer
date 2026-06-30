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
 * \file mixed_quant_flash_attn_kernel_base_gqa_template.h
 * \brief MixedQuantFlashAttn GQA Kernel入口class
 */

#ifndef MIXED_QUANT_FLASH_ATTN_KERNEL_BASE_GQA_TEMPLATE_H_
#define MIXED_QUANT_FLASH_ATTN_KERNEL_BASE_GQA_TEMPLATE_H_

#include "flash_attn_block_cube_mixedquant_gqa.h"
#include "flash_attn_block_vec_mixedquant_gqa.h"
#include "flash_attention_mixedquant_block_vec_flashdecode.h"
#include "mixed_quant_flash_attn_tiling_data.h"
#include "../utils/mixed_quant_flash_attn_common_def.h"

#include "../../../common/op_kernel/const_def.h"
#include "../../../common/op_kernel/fia_public_define.h"
#include "../../../common/op_kernel/arch35/fia_public_define_arch35.h"
#include "../../../common/op_kernel/buffer_manager.h"
#include "../../../common/op_kernel/buffers_policy.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../../common/op_kernel/vector_common.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;
using namespace fa_base_matmul;

template <inferFaLayOutTypeEnum LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormatMQ()
{
    if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BSND || LAYOUT == inferFaLayOutTypeEnum::LAYOUT_TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BNSD) {
        return UbFormat::GS1;
    }
}

namespace BaseApi {

template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionMixedQuantGqaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;
    static constexpr bool HAS_PREFIX = CubeBlockType::HAS_PREFIX;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;

    static constexpr inferFaLayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT_Q_MQ;
    static constexpr inferFaLayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT_Q_MQ;

    static constexpr ActualSeqLensMode Q_MODE =
        (LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND) ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;
    static constexpr ActualSeqLensMode KV_MODE = ActualSeqLensMode::BY_BATCH;

    using QSeqParserType = typename std::conditional<(LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND),
                                                     ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;
    using KvSeqParserType = ActualSeqLensParser<KV_MODE, int32_t>;

    typename std::conditional<(LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<Q_MODE, int32_t, true>, ActualSeqLensParser<Q_MODE, int32_t>>::type
        qCuSeqLensParser;

    ActualSeqLensParser<KV_MODE, int32_t> kvCuSeqLensParser;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser;

    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 2,
                  "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 2 for 3-stage pipeline");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                  "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmmBuffers;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<INPUT_T> sinkGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoX constInfo;

    const __gm__ MixedQuantFlashAttnTilingArch35 *__restrict tilingData;
    TPipe *pipe;
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    uint32_t coreGS1Loops = 0U;
    uint32_t frontGS1Count = 0U;
    uint32_t invalidGS1Size = 0U;
    uint32_t accGS1Loops = 0U;
    uint32_t totalSize = 0U;
    uint32_t createdTaskCount = 0U;
    uint32_t varlenCalcTimes = 0U;
    bool enableS1OutSplit = false;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    int32_t s1SizeCurrBatch = 0;
    uint32_t curS2Start;
    uint32_t curS2End = 0;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;
    bool emptyBatch = false;
    uint32_t dealZerosMSize = 0;

    uint32_t sectionNum_;
    bool isFd;
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;
    FDparamsX fdParams_;

    TQue<QuePosition::VECOUT, 1> SharedBuffer1[2];
    TQue<QuePosition::VECIN, 1> SharedBuffer2[2];
    TBuf<> SharedBuffer3;

    __aicore__ inline FlashAttentionMixedQuantGqaKernel()
        : cubeBlock(constInfo), vecFaBlock(constInfo), vecFdBlock(constInfo){};

    __aicore__ inline void
    Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *kDescale,
         __gm__ uint8_t *vDescale, __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
         __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *blocktable, __gm__ uint8_t *learnableSink,
         __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *metaData,
         const __gm__ optiling::MixedQuantFlashAttnTilingArch35 *__restrict tilingData, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tilingData = tilingData;

        InitConstInfo();

        keyPtr = key;
        valuePtr = value;

        if constexpr (LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, constInfo.cuSeqLensQSize + 1);
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)sequsedQ, constInfo.seqUsedQSize);
        } else {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)sequsedQ, constInfo.seqUsedQSize);
        }
        if constexpr (LAYOUT_KV == inferFaLayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, constInfo.cuSeqLensKVSize + 1);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)sequsedKv, constInfo.seqUsedKvSize);
        } else {
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)sequsedKv, constInfo.seqUsedKvSize);
        }

        sectionNum_ = ((__gm__ uint32_t *)metaData)[0];
        isFd = static_cast<bool>(((__gm__ uint32_t *)metaData)[1]);

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(metaData + MQFA_METADATA_HEADER_OFFSET),
                                     MQFA_AIC_CORE_NUM * 16U * sectionNum_);

        InitQCuSeqLensParser();
        InitKvCuSeqLensParser();

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            fdMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(metaData + MQFA_METADATA_HEADER_OFFSET +
                                                             MIXED_QUANT_FLASH_ATTN_METADATA_SIZE * MQFA_AIC_CORE_NUM *
                                                                 sectionNum_ * sizeof(uint32_t)),
                                         MQFA_AIV_CORE_NUM * 16U * sectionNum_);
            vecFaBlock.isFd = isFd;
            if constexpr (LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND) {
                vecFaBlock.InitVecBlock(tPipe, cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse,
                                        attentionOut, workspace);
                vecFaBlock.SetCuSeqLensParser(qCuSeqLensParser);
            } else {
                vecFaBlock.InitVecBlock(tPipe, sequsedQ, sequsedKv, attenMask, learnableSink, softmaxLse, attentionOut,
                                        workspace);
                vecFaBlock.SetCuSeqLensParser(qSeqUsedParser);
            }
            vecFaBlock.ClearOutput();
        }

        if ASCEND_IS_AIC {
            if constexpr (LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND) {
                cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blocktable, qCuSeqLensParser,
                                        kvCuSeqLensParser);
            } else {
                cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blocktable, qSeqUsedParser,
                                        kvSeqUsedParser);
            }
        }

        if (isFd) {
            if ASCEND_IS_AIV {
                vecFdBlock.InitParams();
                vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                            this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm, keyPtr);
                if (learnableSink != nullptr) {
                    sinkGm.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
                    constInfo.learnableSinkFlag = true;
                    vecFdBlock.InitLearnableSinkGm(sinkGm);
                }
                if (constInfo.isSoftmaxLseEnable) {
                    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
                }
            }
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);

        uint32_t mm1ResultSize = mBaseSize / MQFA_CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager.Init(pipe, 524288);
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        ubBufferManager.Init(pipe, mm1ResultSize * 2);
        bmmBuffers.Init(ubBufferManager, mm1ResultSize);
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }
        constInfo.bSize = this->tilingData->flashAttnBaseParams.bSize;
        constInfo.t1Size = this->tilingData->flashAttnBaseParams.t1Size;
        constInfo.t2Size = this->tilingData->flashAttnBaseParams.t2Size;
        constInfo.n2Size = this->tilingData->flashAttnBaseParams.n2Size;
        constInfo.gSize = this->tilingData->flashAttnBaseParams.gSize;
        constInfo.s1Size = this->tilingData->flashAttnBaseParams.s1Size;
        constInfo.s2Size = this->tilingData->flashAttnBaseParams.s2Size;
        constInfo.dSize = this->tilingData->flashAttnBaseParams.dSize;
        constInfo.dSizeV = this->tilingData->flashAttnBaseParams.dSizeV;
        constInfo.cuSeqLensQSize = this->tilingData->flashAttnBaseParams.cuSeqLensQSize;
        constInfo.cuSeqLensKVSize = this->tilingData->flashAttnBaseParams.cuSeqLensKVSize;
        constInfo.seqUsedQSize = this->tilingData->flashAttnBaseParams.seqUsedQSize;
        constInfo.seqUsedKvSize = this->tilingData->flashAttnBaseParams.seqUsedKvSize;
        constInfo.scaleValue = this->tilingData->flashAttnBaseParams.scaleValue;
        constInfo.isKvContinuous = this->tilingData->flashAttnBaseParams.isKvContinuous != 0;
        constInfo.coreNum = this->tilingData->flashAttnBaseParams.coreNum;
        constInfo.outputLayout = this->tilingData->flashAttnBaseParams.outputLayout;
        constInfo.needInitOutput = this->tilingData->flashAttnBaseParams.needInitOutput;

        constInfo.sparseMode = this->tilingData->flashAttnAttenMaskParams.sparseMode;
        constInfo.preTokens = this->tilingData->flashAttnAttenMaskParams.winLefts;
        constInfo.nextTokens = this->tilingData->flashAttnAttenMaskParams.winRights;
        constInfo.attenMaskBatch = this->tilingData->flashAttnAttenMaskParams.attenMaskBatch;
        constInfo.attenMaskS1Size = this->tilingData->flashAttnAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = this->tilingData->flashAttnAttenMaskParams.attenMaskS2Size;
        constInfo.isExistRowInvalid = this->tilingData->flashAttnAttenMaskParams.isExistRowInvalid;

        constInfo.accumOutSize = this->tilingData->flashAttnWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = this->tilingData->flashAttnWorkspaceParams.logSumExpSize;

        constInfo.maxBlockNumPerBatch = this->tilingData->flashAttnPageAttentionParams.maxBlockNumPerBatch;
        constInfo.blockSize = this->tilingData->flashAttnPageAttentionParams.blockSize;
        constInfo.paLayoutType = this->tilingData->flashAttnPageAttentionParams.paLayoutType;

        constInfo.isSoftmaxLseEnable = this->tilingData->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline void InitQCuSeqLensParser()
    {
        if constexpr (LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND) {
            qCuSeqLensParser.Init(cuSeqLensGmQ, seqUsedGmQ, constInfo.cuSeqLensQSize, constInfo.seqUsedQSize);
        } else {
            qSeqUsedParser.Init(seqUsedGmQ, constInfo.seqUsedQSize, constInfo.s1Size);
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser()
    {
        if constexpr (LAYOUT_KV == inferFaLayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser.Init(cuSeqLensGmKv, seqUsedGmKv, constInfo.cuSeqLensKVSize, constInfo.seqUsedKvSize);
        } else {
            kvSeqUsedParser.Init(seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.s2Size);
        }
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return MIXED_QUANT_FLASH_ATTN_METADATA_SIZE * MQFA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }
    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return MQFA_FD_METADATA_SIZE * MQFA_AIV_CORE_NUM * sectionIdx + MQFA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmmBuffers.Get().SetCrossCore();
            bmmBuffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmmBuffers.Get().WaitCrossCore();
            bmmBuffers.Get().WaitCrossCore();
        }
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ =
            faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, MIXED_QUANT_FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, MQFA_FD_M_START_INDEX, sectionIdx));
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
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
        uint32_t validTaskCount = 0;
        while (shouldDispatchTask || validTaskCount) {
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        vecFaBlock.DealZeroActSeqLen(bN2Cur);
                    }
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            if (validTaskCount) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
                if (executedTaskCount > PRELOAD_N + 1) {
                    validTaskCount--;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != bN2End_) || (gS1Cur != gS1OEnd_) || (s2Cur != s2OEnd_));
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if (constInfo.cuSeqLensKVSize == 0 && constInfo.seqUsedKvSize == 0 && !constInfo.isKvContinuous) {
                actSeqLensKv = SeqLenFromTensorListMQ(keyPtr, bIdx);
            } else {
                if constexpr (LAYOUT_KV == inferFaLayOutTypeEnum::LAYOUT_TND) {
                    actSeqLensKv = kvCuSeqLensParser.GetActualSeqLength(bIdx);
                } else {
                    actSeqLensKv = kvSeqUsedParser.GetActualSeqLength(bIdx);
                }
            }
            if constexpr (LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND) {
                actSeqLensQ = qCuSeqLensParser.GetActualSeqLength(bIdx);
            } else {
                actSeqLensQ = qSeqUsedParser.GetActualSeqLength(bIdx);
            }
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
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

    __aicore__ inline uint64_t SeqLenFromTensorListMQ(__gm__ uint8_t *keyPtr, uint32_t bIndex)
    {
        uint64_t dimInfo[4];
        AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
        keyTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
        if constexpr (LAYOUT_KV == inferFaLayOutTypeEnum::LAYOUT_BSND) {
            return keyTensorDesc.GetShape(1);
        } else {
            return keyTensorDesc.GetShape(2);
        }
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo.preTokens;
        nextTokenLeftUp = constInfo.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo.sparseMode);

        if (constInfo.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
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

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.gSize)) -
            1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormatMQ<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.gSize);
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
            curS2Start = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) {
            tailS2Split = s2OEnd_ > 0U ? true : false;
            curS2End = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK];
        RunInfoX &runInfoNegNMinus1 = taskRunInfo[(loop - PRELOAD_N - 1) & PRELOAD_TASK_CACHE_MASK];

        if (runInfo0.isValid) {
            if ASCEND_IS_AIV {
                AntiquantKey(runInfo0);
            } else {
                ComputeMm1(runInfo0);
            }
        }

        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIV {
                    ComputeVec1(runInfoNegN);
                    AntiquantValue(runInfoNegN);
                } else {
                    ComputeMm2(runInfoNegN);
                }
            }
        }

        if (loop >= PRELOAD_N + 1) {
            if (runInfoNegNMinus1.isValid) {
                if ASCEND_IS_AIV {
                    ComputeVec2(runInfoNegNMinus1);
                }
                runInfoNegNMinus1.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void AntiquantKey(RunInfoX &runInfo)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void AntiquantValue(RunInfoX &runInfo)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop;
        info.bIdx = bN2Cur / constInfo.n2Size;
        info.n2Idx = bN2Cur % constInfo.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_BSND || LAYOUT_Q == inferFaLayOutTypeEnum::LAYOUT_TND) {
            info.s1Idx = info.gS1Idx / constInfo.gSize;
        } else {
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(MQFA_BYTE_BLOCK / sizeof(INPUT_T)));
        if (constInfo.isKvContinuous) {
            info.isChangeBatch = false;
        } else {
            if (loop == 0) {
                info.isChangeBatch = true;
            } else {
                info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
            }
        }

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

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

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
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

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx < constInfo.coreNum) {
            CrossCoreBufferInit();
            if ASCEND_IS_AIV {
                vecFaBlock.InitBuffers(SharedBuffer1, SharedBuffer2, SharedBuffer3);
                vecFaBlock.AllocEventID();
            } else {
                cubeBlock.InitBuffers();
                cubeBlock.AllocEventID();
            }
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo.aicIdx < constInfo.coreNum) {
                FlashAttention(sectionIdx);
            }
            if ASCEND_IS_AIV {
                FlashDecode(sectionIdx);
            }
        }

        if (constInfo.aicIdx < constInfo.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock.FreeEventID();
            } else {
                cubeBlock.FreeEventID();
            }
            CrossCoreBufferUnInit();
        }
    }
}; // FlashAttentionMixedQuantGqaKernel

} // namespace BaseApi

#endif // MIXED_QUANT_FLASH_ATTN_KERNEL_BASE_GQA_TEMPLATE_H_
