/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_kernel_dn.h
 * \brief
 */

#ifndef QFA_KERNEL_DN_H
#define QFA_KERNEL_DN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_flash_attn_template_tiling_key.h"
#include "quant_flash_attn_common_def.h"
#include "../../common/op_kernel/const_def.h"

using namespace optiling;
using namespace AscendC;
using namespace AttentionCommon;

namespace QFA_KERNEL {

template <typename QFAT, typename CubeBlock, typename VectorBlock>
class QuantFlashAttnKernelDn {
public:
    using QUANT_T = typename QFAT::quantType;
    using SCALE_T = typename QFAT::scaleType;
    using OUT_T = typename QFAT::outputType;
    using SEQLEN_T = uint32_t;
    static constexpr bool PAGE_ATTENTION = QFAT::pageAttention;
    static constexpr bool HAS_MASK = QFAT::hasMask;
    static constexpr QFA_LAYOUT LAYOUT_Q = QFAT::qLayout;
    static constexpr QFA_LAYOUT LAYOUT_KV = QFAT::kvLayout;

    static constexpr uint32_t mBaseSize = 128;
    static constexpr uint32_t s2BaseSize = 256;
    static constexpr uint32_t dBaseSize = 128;
    static constexpr uint32_t dVBaseSize = 128;

    static constexpr uint8_t SYNC_MODE_2 = 2;
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {9, 10};
    static constexpr uint16_t CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_BUF1_GMAX_UB_TO_L1 = 6;
    static constexpr uint16_t CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB = 7;
    static constexpr uint16_t CROSS_CORE_SYNC_BUF1_GMAX_L1_TO_UB = 8;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_0 = 3;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_1 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 2;
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 1;
    static constexpr uint16_t CROSS_CORE_SYNC_UB_L1 = 0;

    static constexpr uint32_t PRELOAD_N = 20;
    static constexpr uint32_t DELAY_P_SCALE_N = 3;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;
    static constexpr uint32_t TILE_N = 16;
    static constexpr uint32_t SUB_S2_BASE_SIZE = s2BaseSize / 2;

    ConstInfo constInfo;
    const FlashAttnTilingData *__restrict tilingData;

    // metadata
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    uint32_t sectionNum_ = 0;
    // fa metadata
    uint32_t bN2Start_ = 0;
    uint32_t bN2End_ = 0;
    uint32_t gS1OStart_ = 0;
    uint32_t gS1OEnd_ = 0;
    uint32_t s2OStart_ = 0;
    uint32_t s2OEnd_ = 0;
    uint32_t s2FirstStartVecCore = 0;
    uint32_t tileMaxIdx = 3;
    uint32_t updateScaleNum = 3;
    // fd metadata
    FDparams fdParams_;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint64_t gS1Size_ = 0;      // actSeqLensQ * gSize, 换batch才变
    uint64_t s2LoopTimes_ = 0;  // ceil(actSeqLensKv / s2BaseSize), 换batch才变
    uint64_t gS1LoopTimes_ = 0; // ceil(gS1Size / mBaseSize), 换batch才变
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t prevBIdx = 0;
    uint32_t prevBN2Idx = 0;
    uint32_t prevGS1Idx = 0;
    uint32_t pscaleNum = 0;

    SeqLensTool<LAYOUT_Q, SEQLEN_T> qSeqLensTool;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> kvSeqLensTool;

    CubeBlock cubeBlock;
    VectorBlock vectorBlock;

    __aicore__ inline QuantFlashAttnKernelDn()
        : cubeBlock(constInfo, qSeqLensTool, kvSeqLensTool),
          vectorBlock(constInfo, qSeqLensTool, kvSeqLensTool){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensKv, __gm__ uint8_t *seqUsedQ,
                                __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                __gm__ uint8_t *fiaMetaData, const FlashAttnTilingData *__restrict tiling)
    {
        this->tilingData = tiling;

        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);
        fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * sectionNum_);

        InitConstInfo();

        qSeqLensTool.Init(cuSeqlensQ, constInfo.qCuSeqLensSize, seqUsedQ, constInfo.qSeqUsedSize, constInfo.maxSeqlenQ);
        kvSeqLensTool.Init(cuSeqlensKv, constInfo.kvCuSeqLensSize, seqUsedKv, constInfo.kvSeqUsedSize,
                           constInfo.maxSeqlenKv);

        if ASCEND_IS_AIC {
            cubeBlock.InitInput(query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable);
        } else {
            vectorBlock.InitInput(attentionOut);
            vectorBlock.ClearOutput();
        }
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

        auto fiaBaseParams = this->tilingData->flashAttnBaseParams;
        auto fiaAttenMaskParams = this->tilingData->flashAttnAttenMaskParams;
        auto fiaPageAttentionParams = this->tilingData->flashAttnPageAttentionParams;
        auto fiaWorkspaceParams = this->tilingData->flashAttnWorkspaceParams;
        auto fiaEmptyTensorParams = this->tilingData->flashAttnEmptyTensorParams;

        constInfo.bSize = fiaBaseParams.bSize;
        constInfo.t1Size = fiaBaseParams.t1Size;
        constInfo.t2Size = fiaBaseParams.t2Size;
        constInfo.n2Size = fiaBaseParams.n2Size;
        constInfo.gSize = fiaBaseParams.gSize;
        constInfo.s1Size = fiaBaseParams.s1Size;
        constInfo.s2Size = fiaBaseParams.s2Size;
        constInfo.dSize = fiaBaseParams.dSize;
        constInfo.dSizeV = fiaBaseParams.dSizeV;
        constInfo.qCuSeqLensSize = fiaBaseParams.qCuSeqLensSize;
        constInfo.kvCuSeqLensSize = fiaBaseParams.kvCuSeqLensSize;
        constInfo.qSeqUsedSize = fiaBaseParams.qSeqUsedSize;
        constInfo.kvSeqUsedSize = fiaBaseParams.kvSeqUsedSize;

        constInfo.maxSeqlenQ =
            (fiaBaseParams.maxSeqlenQ >= 0) ? static_cast<uint64_t>(fiaBaseParams.maxSeqlenQ) : constInfo.s1Size;
        constInfo.maxSeqlenKv =
            (fiaBaseParams.maxSeqlenKv >= 0) ? static_cast<uint64_t>(fiaBaseParams.maxSeqlenKv) : constInfo.s2Size;
        constInfo.scaleValue = static_cast<float>(fiaBaseParams.scaleValue);
        constInfo.coreNum = fiaBaseParams.coreNum;

        constInfo.sparseMode = fiaAttenMaskParams.sparseMode;
        constInfo.preTokens = fiaAttenMaskParams.winLefts;
        constInfo.nextTokens = fiaAttenMaskParams.winRights;
        constInfo.attenMaskS1Size = fiaAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = fiaAttenMaskParams.attenMaskS2Size;

        constInfo.accumOutSize = fiaWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = fiaWorkspaceParams.logSumExpSize;
        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = fiaPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = fiaPageAttentionParams.blockSize;
            constInfo.paLayoutType = fiaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = fiaBaseParams.isSoftMaxLseEnable;
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        // AICPU metadata format: 16 fields per AIC core, 0-indexed (no leading CORE_ENABLE).
        // Kernel field constants ( FLASH_ATTN_BN2_START_INDEX=1, etc.) are 1-based, so subtract 1.
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FA_FD_METADATA_SIZE * FA_AIV_CORE_NUM * sectionIdx + FA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // Reset pipeline state for each section to avoid cross-section deadlock
        uint32_t createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
        uint32_t validTaskCount = 0;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        while (shouldDispatchTask || validTaskCount) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        // vectorBlock.DealZeroActSeqLen(bN2Cur);
                    }
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
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
        // 三层区间判定, 逐级回退: bN2 未到尾 -> 判 gS1; gS1 未到尾 -> 判 s2
        return (bN2Cur < bN2End_) ||
               (bN2Cur == bN2End_ && ((gS1Cur < gS1OEnd_) || (gS1Cur == gS1OEnd_ && s2Cur < s2OEnd_)));
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = static_cast<uint32_t>(s2LoopTimes_); // 与 GetTaskDealMode 中缓存一致

        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            curS2Start = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            curS2End = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur) {}

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            actSeqLensQ = qSeqLensTool.GetActualSeqLength(bIdx);
            actSeqLensKv = kvSeqLensTool.GetActualSeqLength(bIdx);
            // 循环次数只依赖 actSeqLens, 换batch才变, 缓存避免每个任务重复除法
            gS1Size_ = actSeqLensQ * constInfo.gSize;
            s2LoopTimes_ = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
            gS1LoopTimes_ = (gS1Size_ + mBaseSize - 1) / mBaseSize;
        }
        if (s2LoopTimes_ == 0 || gS1LoopTimes_ == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }

        if (s2Cur < curS2Start && curS2Start < curS2End) {
            return TASK_DEAL_MODE::NOT_START;
        }

        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfo &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];

        if (loop >= PRELOAD_N) {
            RunInfo &runInfo20 = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfo20.isValid) {
                if ASCEND_IS_AIC {
                    if (unlikely(runInfo20.isC2Sync)) { // 1/16结构频率
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo20.pscaleNum);
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_0 +
                                                                  runInfo20.pscaleNum + 16);
                    }
                    if (unlikely(runInfo20.isUpdatePScale)) { // ~1/16结构频率
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    }
                    ComputeMm2(runInfo20);
                    if (unlikely(runInfo20.isUpdatePScale)) { // ~1/16结构频率
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
                    }
                } else {
                    if (unlikely(runInfo20.isUpdatePScale)) { // ~1/16结构频率
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
                        ComputeVec2(runInfo20);
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    }
                }
                runInfo20.isValid = false;
            }
        }

        if (loop >= DELAY_P_SCALE_N) {
            RunInfo &runInfo3 = taskRunInfo[(loop - DELAY_P_SCALE_N) % PRELOAD_TASK_CACHE_SIZE];
            if (runInfo3.isValid && unlikely(runInfo3.isUpdatePScale)) { // ~1/16结构频率
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                              runInfo3.tileMaxIdx / 2);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                              runInfo3.tileMaxIdx / 2 + 16);
                    CopyGMaxL1ToUb(runInfo3);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB +
                                                             runInfo3.tileMaxIdx / 2);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB +
                                                             runInfo3.tileMaxIdx / 2 + 16);
                    if (runInfo3.tileMaxIdx == 3) {
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1);
                        CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1 + 16);
                    }
                } else {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_BUF0_GMAX_L1_TO_UB +
                                                           runInfo3.tileMaxIdx / 2);
                    UpdatePScale(runInfo3);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2_0 + runInfo3.pscaleNum);
                }
            }
        }

        if (runInfo0.isValid) {
            uint32_t mm1ResBufId = (runInfo0.loop / 2) % 2;
            uint32_t subBlockIdx = runInfo0.loop % 2;
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
                ComputeMm1(runInfo0);
                CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
            } else {
                if (subBlockIdx == constInfo.subBlockIdx) {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                    ComputeVec1(runInfo0);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                }
                if (runInfo0.isUpdatePScale) {
                    if (runInfo0.tileMaxIdx == 0) {
                        CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_UB_L1);
                    }
                    CopyGMaxUbToL1(runInfo0);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_BUF0_GMAX_UB_TO_L1 +
                                                             runInfo0.tileMaxIdx / 2);
                }
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfo &runInfo) { cubeBlock.ComputeMm1(runInfo); }

    __aicore__ inline void ComputeMm2(RunInfo &runInfo) { cubeBlock.ComputeMm2(runInfo); }

    __aicore__ inline void ComputeVec1(RunInfo &runInfo) { vectorBlock.ComputeVec1(runInfo); }

    __aicore__ inline void CopyGMaxUbToL1(RunInfo &runInfo) { vectorBlock.CopyGMaxUbToL1(runInfo); }

    __aicore__ inline void CopyGMaxL1ToUb(RunInfo &runInfo) { cubeBlock.CopyGMaxL1ToUb(runInfo); }

    __aicore__ inline void UpdatePScale(RunInfo &runInfo) { vectorBlock.UpdatePScale(runInfo); }

    __aicore__ inline void ComputeVec2(RunInfo &runInfo) { vectorBlock.ComputeVec2(runInfo); }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfo &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfo &info)
    {
        info.loop = loop;
        info.bIdx = bN2Cur / constInfo.n2Size;
        info.n2Idx = bN2Cur % constInfo.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.s2Idx = s2Cur * s2BaseSize;
        info.curS2LoopIdx = s2Cur - curS2Start;

        info.actMSize = mBaseSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size_) {
            uint64_t tailM = (gS1Size_ > gS1Cur * mBaseSize) ? (gS1Size_ - gS1Cur * mBaseSize) : mBaseSize;
            info.actMSize = (tailM < mBaseSize) ? static_cast<uint32_t>(tailM) : mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
            info.actSingleLoopS2Size =
                (actSeqLensKv > (uint64_t)s2Cur * s2BaseSize) ? (actSeqLensKv - (uint64_t)s2Cur * s2BaseSize) : 0;
        }
        info.actSingleLoopS2SizeAlign =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)AttentionCommon::BYTE_BLOCK);      // 统一对齐到32
        info.actSingleLoopS2SizeAlign16 = Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)16); // 统一对齐到16
        info.actSingleLoopS2SizeAlign64 =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)BUFFER_SIZE_BYTE_64B); // 统一对齐到64

        info.prefetched = (((s2Cur - curS2Start) & 1) == 1);
        info.pairKVCopyS2Size = info.actSingleLoopS2Size;
        if (!info.prefetched && (s2Cur + 1) < curS2End) {
            uint64_t nextSize = actSeqLensKv - (uint64_t)(s2Cur + 1) * s2BaseSize;
            info.pairKVCopyS2Size = s2BaseSize + static_cast<uint32_t>(nextSize > s2BaseSize ? s2BaseSize : nextSize);
        }

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
        info.isUpdatePScale = (info.isLastS2Loop || ((info.curS2LoopIdx + 1) % TILE_N == 0));
        info.isC2Sync = (info.curS2LoopIdx % TILE_N == 0);
        if (info.isFirstS2Loop) {
            s2FirstStartVecCore = loop % 2;
        }
        info.s2FirstStartVecCore = s2FirstStartVecCore;
        if (info.isFirstS2Loop || info.isC2Sync) {
            tileMaxIdx = (tileMaxIdx + 1) % 4;
            pscaleNum = (pscaleNum + 1) % 20;
        }
        if (info.isUpdatePScale && info.curS2LoopIdx / 16 != 0) {
            updateScaleNum = (updateScaleNum + 1) % 4;
        }
        info.updateScaleNum = updateScaleNum;
        info.tileMaxIdx = tileMaxIdx;
        info.isS2FirstTilePerCore = (info.curS2LoopIdx % TILE_N / 2 == 0);
        info.pscaleNum = pscaleNum / 10;

        info.actMSizeAlign128 = (info.actMSize + 127) >> 7 << 7;
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        // s2LoopTimes_/gS1LoopTimes_ 已在 GetTaskDealMode 中随 batch 缓存
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            if (s2Cur + 1 < s2LoopTimes_) {
                s2Cur++;
                return;
            }
        }

        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes_) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完
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
    }

    __aicore__ inline void Process()
    {
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo.aicIdx < constInfo.coreNum) {
                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[0]);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);

                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    vectorBlock.InitTensors();
                } else {
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1);
                    CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_UB_L1 + 16);
                    cubeBlock.InitTensors();
                }
                FlashAttention(sectionIdx);
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0]);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0] + 16);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1]);
                    CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1] + 16);
                }
            }
        }
    }
};

} // namespace QFA_KERNEL

#endif
