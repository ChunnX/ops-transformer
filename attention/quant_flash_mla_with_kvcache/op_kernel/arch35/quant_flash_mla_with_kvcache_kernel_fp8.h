/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_with_kvcache_kernel_fp8.h
 * \brief QuantFlashMlaWithKvcache FP8 kernel（自QFA kernel_fp8与FIA MLA kernel裁剪适配，
 *        多section metadata调度 + int32序列解析（cu_seq/seq_used/cache_seqlens） + MLA不合轴）
 */

#ifndef QUANT_FLASH_MLA_WITH_KVCACHE_KERNEL_FP8_H_
#define QUANT_FLASH_MLA_WITH_KVCACHE_KERNEL_FP8_H_

#include "quant_flash_mla_with_kvcache_common_def.h"
#include "quant_flash_mla_with_kvcache_tiling_data.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "quant_flash_mla_block_cube_fp8.h"
#include "quant_flash_mla_block_vec_fp8.h"
#include "quant_flash_mla_block_vec_flashdecode_fp8.h"
#include "memory_copy_arch35_quant_flash_mla_with_kvcache.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class QuantFlashMlaKernelFp8 {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_Q, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    BufferManager<BufferType::UB> ubBufferManager_;
    BufferManager<BufferType::L1> l1BufferManager_;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers_;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers_;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers_;

    GlobalTensor<int32_t> cacheSeqLensGm_;
    GlobalTensor<uint32_t> faMetaDataGm_;
    GlobalTensor<uint32_t> fdMetaDataGm_;
    GlobalTensor<float> softmaxLseGm_;

    ConstInfoX constInfo_;

    const __gm__ QuantFlashMlaWithKvcacheTilingData *__restrict tilingData_ = nullptr;
    TPipe *pipe_ = nullptr;
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

    uint32_t createdTaskCount_ = 0U;

    // scheduler params
    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    uint32_t curS2Start_ = 0;
    uint32_t curS2End_ = 0;
    uint32_t prevBIdx_ = 0;
    uint32_t prevBN2Idx_ = 0;
    uint32_t prevGS1Idx_ = 0;
    uint32_t mloop_ = 0;
    bool headS2Split_ = false;
    bool tailS2Split_ = false;

    // metadata
    uint32_t sectionNum_ = 0;
    // fa metadata
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;
    // fd metadata
    FDparamsX fdParams_;

    // Q: TND, cu_seq(含前导0)+seq_used, int32（ACCUM模式）
    using QSeqParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>;
    QSeqParserType qCuSeqLensParser_;
    // KV: PA, cache_seqlens按batch, int32（BY_BATCH模式）
    using KvSeqParserType = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;
    KvSeqParserType kvSeqUsedParser_;

    // ==============================function========================================
    __aicore__ inline QuantFlashMlaKernelFp8()
        : cubeBlock_(constInfo_),
          vecFaBlock_(constInfo_),
          vecFdBlock_(constInfo_){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *qDescale,
                                __gm__ uint8_t *kDescale, __gm__ uint8_t *blockTable, __gm__ uint8_t *cacheSeqLens,
                                __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *attnMask,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *workspace,
                                const __gm__ QuantFlashMlaWithKvcacheTilingData *__restrict tiling, TPipe *tPipe)
    {
        this->pipe_ = tPipe;
        this->tilingData_ = tiling;

        InitConstInfo();

        // metadata header: [0]=sectionNum
        sectionNum_ = ((__gm__ uint32_t *)metadata)[QMLA_HEAD_SECTION_NUM_INDEX];
        // FA区: header之后, 布局[section][aicIdx][16]
        faMetaDataGm_.SetGlobalBuffer(
            (__gm__ uint32_t *)(metadata + QMLA_METADATA_HEADER_SIZE * sizeof(uint32_t)),
            QMLA_AIC_CORE_NUM * QMLA_FA_METADATA_SIZE * sectionNum_);
        // FD区: FA区之后, 布局[section][aivIdx][16]
        fdMetaDataGm_.SetGlobalBuffer(
            (__gm__ uint32_t *)(metadata +
                                (QMLA_METADATA_HEADER_SIZE + sectionNum_ * QMLA_AIC_CORE_NUM * QMLA_FA_METADATA_SIZE) *
                                    sizeof(uint32_t)),
            QMLA_AIV_CORE_NUM * QMLA_FD_METADATA_SIZE * sectionNum_);

        // Q序列解析: cu_seqlens_q(B+1) + seqused_q(B)
        qCuSeqLensParser_.Init(cuSeqLensQ, constInfo_.cuSeqLensQSize + 1, sequsedQ, constInfo_.seqUsedQSize);
        // KV序列解析: cache_seqlens(B)
        cacheSeqLensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSeqLens, constInfo_.kvSeqUsedSize);
        kvSeqUsedParser_.Init(cacheSeqLensGm_, constInfo_.kvSeqUsedSize, constInfo_.s2Size);
        // FD统一开关: metadata header[1], 由metadata算子按是否存在FD任务写入, 所有核一致
        constInfo_.enableFlashDecode =
            static_cast<bool>(((__gm__ uint32_t *)metadata)[QMLA_HEAD_IS_FD_INDEX]);
        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            // MLA: V=K_nope, descale共用k_descale
            vecFaBlock_.InitVecBlock(tPipe, qDescale, kDescale, kDescale, softmaxLse, attnOut, workspace,
                                     qCuSeqLensParser_, attnMask);
            vecFaBlock_.ClearOutput();
        }

        if ASCEND_IS_AIC {
            // MLA: 无独立v_cache, value复用key指针
            cubeBlock_.InitCubeBlock(tPipe, &l1BufferManager_, query, key, key, blockTable, qDescale, kDescale, kDescale,
                                      qCuSeqLensParser_, kvSeqUsedParser_);
        }
        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                vecFdBlock_.InitParams();
                vecFdBlock_.InitGlobalTensor(this->vecFaBlock_.softmaxFDMaxGm_, this->vecFaBlock_.softmaxFDSumGm_,
                                             this->vecFaBlock_.accumOutGm_, this->vecFaBlock_.attentionOutGm_);
                vecFdBlock_.SetCuSeqLensParsers(qCuSeqLensParser_);
                if (constInfo_.isSoftmaxLseEnable) {
                    softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    vecFdBlock_.InitSoftmaxLseGm(softmaxLseGm_);
                }
            }
        }
        (void)attnMask;
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);

        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager_.Init(pipe_, 524288); // 512 * 1024
        l1PBuffers_.Init(l1BufferManager_, mm2LeftSize);
        ubBufferManager_.Init(pipe_, mm1ResultSize * 2 + mm2ResultSize);
        bmm2Buffers_.Init(ubBufferManager_, mm2ResultSize);
        bmm1Buffers_.Init(ubBufferManager_, mm1ResultSize);
        workspace = GetUserWorkspace(workspace);
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = GetBlockIdx();
        } else {
            constInfo_.aivIdx = GetBlockIdx();
            constInfo_.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo_.subBlockIdx = GetSubBlockIdx();
        }

        const auto &baseParams = this->tilingData_->baseParams;
        const auto &attenMaskParams = this->tilingData_->attenMaskParams;
        const auto &pageAttentionParams = this->tilingData_->pageAttentionParams;
        const auto &workspaceParams = this->tilingData_->workspaceParams;
        const auto &emptyTensorParams = this->tilingData_->emptyTensorParams;

        constInfo_.bSize = baseParams.bSize;
        constInfo_.t1Size = baseParams.t1Size;
        constInfo_.t2Size = baseParams.t2Size;
        constInfo_.n2Size = baseParams.n2Size;
        constInfo_.gSize = baseParams.gSize;
        constInfo_.s1Size = baseParams.s1Size;
        constInfo_.s2Size = baseParams.s2Size;
        constInfo_.dSize = baseParams.dSize;
        constInfo_.dSizeV = baseParams.dSizeV;
        constInfo_.dSizeRope = constInfo_.dSize - constInfo_.dSizeV;
        // MLA不合轴: N2=1(KV头), G轴承载Q头数(n1Size/n2Size)
        constInfo_.realN2Size = constInfo_.n2Size;
        constInfo_.realGSize = (constInfo_.n2Size > 0) ? (baseParams.n1Size / constInfo_.n2Size) : 1U;
        constInfo_.cuSeqLensQSize = baseParams.cuSeqLensQSize;
        constInfo_.seqUsedQSize = baseParams.seqUsedQSize;
        // cache_seqlens元素个数=B（MLA场景必传）
        constInfo_.kvSeqUsedSize = baseParams.bSize;
        constInfo_.scaleValue = static_cast<float>(baseParams.scaleValue);
        constInfo_.coreNum = baseParams.coreNum;
        constInfo_.needInitOutput = emptyTensorParams.needInit;
        constInfo_.outputLayout = ConvertToQmlaKernelLayout(baseParams.outputLayout);
        // strides
        constInfo_.keyStrides.bnStride = 0;
        constInfo_.keyStrides.n2Stride = 0;
        constInfo_.valueStrides.bnStride = 0;
        constInfo_.valueStrides.n2Stride = 0;

        constInfo_.sparseMode = attenMaskParams.maskMode;
        constInfo_.preTokens = 0;
        constInfo_.nextTokens = 0;
        constInfo_.attenMaskS1Size = attenMaskParams.attenMaskS1Size;
        constInfo_.attenMaskS2Size = attenMaskParams.attenMaskS2Size;

        constInfo_.accumOutSize = workspaceParams.accumOutSize;
        constInfo_.logSumExpSize = workspaceParams.logSumExpSize;

        // pageAttention（MLA固定PA场景）
        constInfo_.maxBlockNumPerBatch = pageAttentionParams.maxBlockNumPerBatch;
        constInfo_.blockSize = pageAttentionParams.blockSize;
        constInfo_.paLayoutType = pageAttentionParams.paLayoutType;
        // LSE
        constInfo_.isSoftmaxLseEnable = baseParams.isSoftMaxLseEnable;

        constInfo_.dBasicBlock = Align64Func((uint16_t)constInfo_.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return QMLA_FA_METADATA_SIZE * QMLA_AIC_CORE_NUM * sectionIdx + QMLA_FA_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return QMLA_FD_METADATA_SIZE * QMLA_AIV_CORE_NUM * sectionIdx + QMLA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm2Buffers_.Get().SetCrossCore();
            bmm1Buffers_.Get().SetCrossCore();
            bmm1Buffers_.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers_.Get().WaitCrossCore();
            bmm1Buffers_.Get().WaitCrossCore();
            bmm2Buffers_.Get().WaitCrossCore();
        }
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo_.aicIdx >= constInfo_.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        createdTaskCount_ = 0;
        uint32_t executedTaskCount = 0;
        mloop_ = 0;
        headS2Split_ = false;
        tailS2Split_ = false;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx_ = bN2Cur;
        prevGS1Idx_ = gS1Cur;

        bool shouldDispatchTask = true;
        uint32_t validTaskCount = 0;
        while (shouldDispatchTask || validTaskCount) {
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTask(createdTaskCount_, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount_++;
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
        if (bN2Cur != bN2End_) {
            return bN2Cur < bN2End_;
        }
        if (gS1Cur != gS1OEnd_) {
            return gS1Cur < gS1OEnd_;
        }
        return s2Cur < s2OEnd_;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfo_.realN2Size;
        if (isFirstTask || prevBIdx_ != bIdx) {
            prevBIdx_ = bIdx;
            // MLA PA场景: KV实际长度来自cache_seqlens
            actSeqLensKv_ = kvSeqUsedParser_.GetActualSeqLength(bIdx);
            // Q实际长度: cu_seq+seq_used解析
            actSeqLensQ_ = qCuSeqLensParser_.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }

        if (isFirstTask || bN2Cur != prevBN2Idx_ || gS1Cur != prevGS1Idx_) {
            CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }

        if (s2Cur < curS2Start_ && curS2Start_ < curS2End_) {
            return TASK_DEAL_MODE::NOT_START;
        }
        if (s2Cur < curS2Start_ || s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start_) {
            mloop_++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ_, int64_t actSeqLensKv_, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = 0;
        nextTokenLeftUp = 0;
        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            headS2Split_ = s2OStart_ != 0U;
            curS2Start_ = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            tailS2Split_ = s2OEnd_ != 0U;
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1(runInfo0);
            } else {
                ComputeVec1(runInfo0);
            }
        }

        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIC {
                    ComputeMm2(runInfoNegN);
                } else {
                    ComputeVec2(runInfoNegN);
                }
                runInfoNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm1(this->bmm1Buffers_.Get(), runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm2(this->bmm2Buffers_.Get(), this->l1PBuffers_, runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        vecFaBlock_.ProcessVec1(this->l1PBuffers_.Get(), this->bmm1Buffers_.Get(), runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        this->vecFaBlock_.ProcessVec2(this->bmm2Buffers_.Get(), runInfo);
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / (constInfo_.realN2Size);
        info.n2Idx = (bN2Cur / (constInfo_.realN2Size / constInfo_.n2Size)) % constInfo_.n2Size;
        info.realN2Idx = bN2Cur % constInfo_.realN2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        // MLA TND: S1G layout, s1Idx = gS1Idx / realGSize
        info.s1Idx = info.gS1Idx / constInfo_.realGSize;
        info.s2Idx = s2Cur * s2BaseSize;
        info.s2LocalIdx = (s2Cur - curS2Start_) * s2BaseSize;
        info.actS1Size = actSeqLensQ_;
        info.actS2Size = actSeqLensKv_;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo_.realGSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

        info.isChangeBatch = false;

        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, info.preTokensLeftUp, info.nextTokensLeftUp);

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);

        // MLA不合轴: vec视角M减半（V0/V1两核分担）
        info.actVecMSize = (info.actMSize + 1) >> 1;
        info.vecMbaseIdx = 0;
        if (constInfo_.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_)) {
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }

        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!constInfo_.enableFlashDecode) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        // FD buffers为绝对偏移LocalTensor纯标量构造（不触碰TPipe），SyncAll前调用无副作用，
        // 与flash_attn FD保持一致；FD内部用Mutex轮转双buffer，不占用EventID池
        vecFdBlock_.InitBuffers();
        AscendC::ICachePreLoad(2);
        // FA→FD切换前排空本核全部在途流水线：FA最后tile的workspace GM写（MTE3）与
        // brdcst源buffer读（UB区域与FD工作区重叠）必须全部落地，FD的MTE2/V才能安全覆写；
        // PipeBarrier<PIPE_ALL>等待所有流水线（MTE1/2/3/SCALAR/VECTOR/FIX）在途操作完成，
        // 不依赖EventID分配，杜绝事件ID错位导致的排空失效；所有AIV均在各自SyncAll前排空
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();
        vecFdBlock_.FlashDecode(fdParams_);
        // FD→FA(i+1)切换前排空本核全部在途流水线：FD的attenOut/lse GM写与fdOutputBuf_源读
        // 必须全部完成，FA(i+1)的V/MTE2才能安全覆写FD曾用UB区域
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QMLA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QMLA_FD_BN_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QMLA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QMLA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QMLA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, QMLA_FD_M_START_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QMLA_FA_BN_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QMLA_FA_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QMLA_FA_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QMLA_FA_BN_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QMLA_FA_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, QMLA_FA_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, QMLA_FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        if (constInfo_.aicIdx < constInfo_.coreNum) {
            CrossCoreBufferInit();
            if ASCEND_IS_AIV {
                vecFaBlock_.InitBuffers();
                vecFaBlock_.AllocEventID();
            } else {
                cubeBlock_.InitBuffers();
                cubeBlock_.AllocEventID();
            }
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                FlashAttention(sectionIdx);
            }
            if constexpr (FLASH_DECODE) {
                if ASCEND_IS_AIV {
                    FlashDecode(sectionIdx);
                }
            }
        }

        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.FreeEventID();
            } else {
                cubeBlock_.FreeEventID();
            }
            CrossCoreBufferUnInit();
        }
    }
}; // QuantFlashMlaKernelFp8

} // namespace BaseApi
#endif // QUANT_FLASH_MLA_WITH_KVCACHE_KERNEL_FP8_H_
