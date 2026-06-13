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
 * \file flash_attn_gqa_cube.h
 * \brief FlashAttn arch35 AIC-side kernel (non-quant GQA)
 */

#ifndef FLASH_ATTN_GQA_CUBE_H_
#define FLASH_ATTN_GQA_CUBE_H_

#include "../../../common/op_kernel/fia_public_define.h"
#include "kernel_operator_list_tensor_intf.h" // TensorDesc
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_utils.h"

#include "flash_attn_block_cube_noquant_gqa.h"
#include "flash_attn_block_vec_noquant_gqa.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "flash_attention_noquant_block_vec_flashdecode.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_attn_tiling_data.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;
    static constexpr bool HAS_PREFIX = CubeBlockType::HAS_PREFIX;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 1,
                  "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 1");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                  "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");


    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;

    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // ===== 核间共享 buffer slots =====
    LocalTensor<uint8_t> bmm1UbSlot[2];
    LocalTensor<uint8_t> bmm2UbSlot[2];
    LocalTensor<uint8_t> l1PSlot[3];
    GlobalTensor<T>      bmm2GmSlot[3];

    // ===== 静态 cross-core 事件 ID =====
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16U;
    static constexpr uint32_t CC_BMM1_0 = 0U;
    static constexpr uint32_t CC_BMM1_1 = 1U;
    static constexpr uint32_t CC_BMM2_0 = 2U;
    static constexpr uint32_t CC_BMM2_1 = 3U;
    static constexpr uint32_t CC_BMM2_2 = 4U;
    static constexpr uint32_t CC_L1P_0  = 5U;
    static constexpr uint32_t CC_L1P_1  = 8U;
    static constexpr uint32_t CC_L1P_2  = 9U;

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<uint32_t> faMetaDataGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoX constInfo;

    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData;
    CubeBlockType cubeBlock;

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
    // fa metadata
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;

    typename std::conditional<(LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<Q_MODE, int32_t, true>, ActualSeqLensParser<Q_MODE>>::type
        qCuSeqLensParser;

    typename std::conditional<(LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE>>::type
        kvCuSeqLensParser;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser;

    // ==============================Functions============================================
    __aicore__ inline FlashAttentionNoQuantGqaKernel()
        : cubeBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling)
    {
        this->tilingData = tiling;

        InitConstInfo();

        keyPtr = key;
        valuePtr = value;

        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, constInfo.cuSeqLensQSize + 1);
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfo.seqUsedQSize);
        } else {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfo.seqUsedQSize);
        }
        if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, constInfo.cuSeqLensKVSize + 1);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfo.seqUsedKvSize);
        } else {
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfo.seqUsedKvSize);
        }
        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);

        InitQCuSeqLensParser();
        InitKvCuSeqLensParser();

        InitMMResBuf(workspace);

        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cubeBlock.InitCubeBlock(query, key, value, blockTable, qCuSeqLensParser,
                                    kvCuSeqLensParser);
        } else {
            cubeBlock.InitCubeBlock(query, key, value, blockTable, qSeqUsedParser,
                                    kvSeqUsedParser);
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        constexpr uint32_t mm1SlotSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2SlotSize = BMM2_TOUB ?
            (mBaseSize / CV_RATIO * dVBaseSize * sizeof(T)) : 0U;
        constexpr uint32_t l1PSlotSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        constexpr uint32_t bmm1UbBase  = 2U * mm2SlotSize;

        if constexpr (BMM2_TOUB) {
            bmm2UbSlot[0] = LocalTensor<uint8_t>(TPosition::VECIN, 0U,          mm2SlotSize);
            bmm2UbSlot[1] = LocalTensor<uint8_t>(TPosition::VECIN, mm2SlotSize, mm2SlotSize);
        }
        bmm1UbSlot[0] = LocalTensor<uint8_t>(TPosition::VECIN, bmm1UbBase,               mm1SlotSize);
        bmm1UbSlot[1] = LocalTensor<uint8_t>(TPosition::VECIN, bmm1UbBase + mm1SlotSize, mm1SlotSize);

        l1PSlot[0] = LocalTensor<uint8_t>(TPosition::A1, 0U,               l1PSlotSize);
        l1PSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1PSlotSize,      l1PSlotSize);
        l1PSlot[2] = LocalTensor<uint8_t>(TPosition::A1, 2U * l1PSlotSize, l1PSlotSize);

        if constexpr (!BMM2_TOUB) {
            int64_t mm2GmSlotElems = mBaseSize * constInfo.dBasicBlock;
            int64_t prevCoreOffset = constInfo.aicIdx * 3 * mm2GmSlotElems;
            bmm2GmSlot[0].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset);
            bmm2GmSlot[1].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset + mm2GmSlotElems);
            bmm2GmSlot[2].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset + 2 * mm2GmSlotElems);
            workspace = workspace + constInfo.coreNum * 3 * mm2GmSlotElems * sizeof(T);
        }
    }

    __aicore__ inline void InitQCuSeqLensParser()
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            qCuSeqLensParser.Init(cuSeqLensGmQ, seqUsedGmQ, constInfo.cuSeqLensQSize, constInfo.seqUsedQSize);
        } else {
            qSeqUsedParser.Init(seqUsedGmQ, constInfo.seqUsedQSize, constInfo.s1Size);
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser()
    {
        if constexpr (LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            kvCuSeqLensParser.Init(cuSeqLensGmKv, seqUsedGmKv, constInfo.cuSeqLensKVSize, constInfo.seqUsedKvSize);
        } else {
            kvSeqUsedParser.Init(seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.s2Size);
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        constInfo.aicIdx = GetBlockIdx();
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
        constInfo.scaleValue = static_cast<float>(this->tilingData->flashAttnBaseParams.scaleValue);
        constInfo.isKvContinuous = this->tilingData->flashAttnBaseParams.isKvContinuous != 0;
        constInfo.coreNum = this->tilingData->flashAttnBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FA_LAYOUT>(this->tilingData->flashAttnBaseParams.outputLayout);
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

        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = this->tilingData->flashAttnPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = this->tilingData->flashAttnPageAttentionParams.blockSize;
            constInfo.paLayoutType = this->tilingData->flashAttnPageAttentionParams.paLayoutType;
        }
        constInfo.isSoftmaxLseEnable = this->tilingData->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_1 + AIV0_AIV1_OFFSET);
        if constexpr (BMM2_TOUB) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0 + AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_1 + AIV0_AIV1_OFFSET);
        }
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
                if (executedTaskCount > PRELOAD_N) {
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
        int64_t s1GLastToken = AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.gSize)) -
                               1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
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
        if (runInfo0.isValid) {
            ComputeMm1(runInfo0);
        }
        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                ComputeMm2(runInfoNegN);
                runInfoNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        uint32_t slotIdx = runInfo.loop % 2U;
        auto bmm1Tensor = bmm1UbSlot[slotIdx].template ReinterpretCast<T>();
        cubeBlock.IterateBmm1(bmm1Tensor, CC_BMM1_0 + slotIdx, CC_BMM1_0 + slotIdx, runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        uint32_t outSlot = runInfo.loop % 2U;
        uint32_t l1PIdx  = runInfo.loop % 3U;
        uint32_t l1PCcId = (l1PIdx == 0U) ? CC_L1P_0 : (l1PIdx == 1U) ? CC_L1P_1 : CC_L1P_2;
        auto l1PTensor   = l1PSlot[l1PIdx].template ReinterpretCast<INPUT_T>();
        if constexpr (BMM2_TOUB) {
            auto bmm2Tensor = bmm2UbSlot[outSlot].template ReinterpretCast<T>();
            cubeBlock.IterateBmm2(bmm2Tensor, CC_BMM2_0 + outSlot, CC_BMM2_0 + outSlot,
                                  l1PTensor, l1PCcId, runInfo);
        } else {
            cubeBlock.IterateBmm2(bmm2GmSlot[runInfo.loop % 3U],
                                  CC_BMM2_0 + (runInfo.loop % 3U), 0U,
                                  l1PTensor, l1PCcId, runInfo);
        }
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
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                      LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
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
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));
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

        if constexpr (USE_DN) {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
        } else {
            info.actVecMSize = (info.actMSize + 1) >> 1;
        }
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

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx < constInfo.coreNum) {
            cubeBlock.InitBuffers();
            cubeBlock.AllocEventID();
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo.aicIdx < constInfo.coreNum) {
                FlashAttention(sectionIdx);
            }
        }

        if (constInfo.aicIdx < constInfo.coreNum) {
            cubeBlock.FreeEventID();
            CrossCoreBufferUnInit();
        }
    }
}; // FlashAttentionNoQuantGqaKernel

} // namespace BaseApi

#endif // FLASH_ATTN_GQA_CUBE_H_
