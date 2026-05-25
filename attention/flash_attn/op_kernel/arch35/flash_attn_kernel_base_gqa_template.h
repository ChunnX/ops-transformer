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
 * \file flash_attention_noquant_kernel_base.h
 * \brief
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_
#define FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_

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
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 2; // 4 (power-of-2) for fast & mask
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;


    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;

    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    BufferManager<BufferType::GM> gmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

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

    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData;
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
    uint32_t executedTaskCount = 0U;
    uint32_t varlenCalcTimes = 0U;
    bool enableS1OutSplit = false;

    // ── 标量指令优化：缓存逐 batch 不变的除法结果 ────────────────────────────
    uint32_t cachedS2LoopTimes   = 0;    // (actSeqLensKv + s2BaseSize-1) / s2BaseSize
    uint32_t cachedGS1LoopTimes  = 0;    // (actSeqLensQ*gSize + mBaseSize-1) / mBaseSize
    uint32_t cachedBIdxForCalc   = 0;    // bN2Cur / n2Size，换 bN2 才更新
    uint32_t cachedN2IdxForCalc  = 0;    // bN2Cur % n2Size
    uint32_t cachedBN2ForCalc    = ~0U;  // 上一次 CalcParams 的 bN2Cur
    uint32_t nextBatchBN2Start_  = 0;    // 下一个 batch 的起始 bN2（避免每次 GetTaskDealMode 做除法）
    int64_t  cachedPreTokensLeftUp_  = 0; // GetPreNextTokenLeftUp 结果缓存（每 batch 算一次）
    int64_t  cachedNextTokensLeftUp_ = 0;
    uint64_t cachedGS1Size_      = 0;    // actSeqLensQ * gSize，供 CalcParams / CalcCurS2StartEnd 复用

    // ── per-gS1-row 不变的 CalcParams 缓存（内层 S2 循环每 tile 约省 ~10 次标量操作）────
    uint32_t cachedRowBN2_         = ~0U; // 上次 CalcParams 的 bN2Cur
    uint32_t cachedRowGS1_         = ~0U; // 上次 CalcParams 的 gS1Cur
    uint64_t cachedRowGS1Idx_      = 0;
    uint64_t cachedRowS1Idx_       = 0;
    uint32_t cachedRowActMSize_    = 0;
    uint32_t cachedRowActMSizeA32_ = 0;
    uint32_t cachedRowActVecMSize_ = 0;
    uint32_t cachedRowVecMbaseIdx_ = 0;
    bool     cachedRowIsS2Split_   = false;
    uint32_t cachedRowTmpOutWsPos_ = 0;
    uint64_t cachedRowMloop_       = 0;

    // schduler params
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

    // metadata
    uint32_t sectionNum_;
    bool isFd;
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

    //  Shared buffer between fa and fd
    TQue<QuePosition::VECOUT, 1> SharedBuffer1[2]; // 共用
    TQue<QuePosition::VECIN, 1> SharedBuffer2[2];  // 共用
    TBuf<> SharedBuffer3;                          // 共用

    // 非tnd时不使用cuSeqLens，这里保证定义
    typename std::conditional<(LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<Q_MODE, int32_t, true>, ActualSeqLensParser<Q_MODE>>::type
        qCuSeqLensParser;

    typename std::conditional<(LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND),
                              ActualSeqLensParser<KV_MODE, int32_t, true>, ActualSeqLensParser<KV_MODE>>::type
        kvCuSeqLensParser;

    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> qSeqUsedParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t> kvSeqUsedParser;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantGqaKernel()
        : cubeBlock(constInfo), vecFaBlock(constInfo), vecFdBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;
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
        isFd = static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[1]);

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);
        

        InitQCuSeqLensParser();
        InitKvCuSeqLensParser();

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * sectionNum_);
            vecFaBlock.isFd = isFd;
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                vecFaBlock.InitVecBlock(tPipe, cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse,
                                        attentionOut, workspace);
                vecFaBlock.SetCuSeqLensParser(qCuSeqLensParser);
            } else {
                vecFaBlock.InitVecBlock(tPipe, seqUsedQ, seqUsedKv, attenMask, learnableSink, softmaxLse, attentionOut,
                                        workspace);
                vecFaBlock.SetCuSeqLensParser(qSeqUsedParser);
            }
            vecFaBlock.ClearOutput();
        }

        if ASCEND_IS_AIC {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blockTable, qCuSeqLensParser,
                                        kvCuSeqLensParser);
            } else {
                cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blockTable, qSeqUsedParser,
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
                if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    vecFdBlock.SetCuSeqLensParsers(qCuSeqLensParser, kvCuSeqLensParser);
                } else {
                    vecFdBlock.SetCuSeqLensParsers(qSeqUsedParser, kvSeqUsedParser);
                }
            }
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);

        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager.Init(pipe, 524288); // 512 * 1024
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2);
        }
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);

        // GM Buffer
        if constexpr (!BMM2_TOUB) {
            int64_t mm2ResultSize =
                mBaseSize * constInfo.dBasicBlock; // 使用Cube计算的总大小，Gm上的数据按照实际的dSize存储
            int64_t prevCoretotalOffset = constInfo.aicIdx * 3 * mm2ResultSize; // 3为preload次数
            // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
            gmBufferManager.Init(workspace + prevCoretotalOffset * sizeof(T));
            bmm2ResGmBuffers.Init(gmBufferManager, mm2ResultSize * sizeof(T));
            workspace = workspace + constInfo.coreNum * 3 * mm2ResultSize * sizeof(T);
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

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = this->tilingData->flashAttnPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = this->tilingData->flashAttnPageAttentionParams.blockSize;
            constInfo.paLayoutType = this->tilingData->flashAttnPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = this->tilingData->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
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

    __aicore__ inline void CrossCoreBufferInit()
    {
        if constexpr (BMM2_TOUB) {
            if ASCEND_IS_AIV {
                bmm2Buffers.Get().SetCrossCore();
                bmm2Buffers.Get().SetCrossCore();
            }
        }
        if ASCEND_IS_AIV {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
        }
        if constexpr (BMM2_TOUB) {
            if ASCEND_IS_AIC {
                bmm2Buffers.Get().WaitCrossCore();
                bmm2Buffers.Get().WaitCrossCore();
            }
        }
    }
    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // Reset pipeline state for each section to avoid cross-section deadlock
        createdTaskCount = 0;
        executedTaskCount = 0;
        mloop = 0;
        accGS1Loops = 0;
        headS2Split = false;
        tailS2Split = false;
        nextBatchBN2Start_ = 0;    // 重置，保证新 section 第一次调用触发 batch 更新
        cachedBN2ForCalc   = ~0U;  // 重置 CalcParams batch 缓存
        cachedRowBN2_      = ~0U;  // 重置 CalcParams row 缓存

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        while (shouldDispatchTask || shouldExecuteTask) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
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
            // 执行任务
            shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
            if (shouldExecuteTask) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != bN2End_) || (gS1Cur != gS1OEnd_) || (s2Cur != s2OEnd_));
    }

    __aicore__ inline bool ShouldExecuteTask(RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        (void)taskRunInfo;
        // 需要 createdTaskCount + PRELOAD_N 次调用才能将最后 PRELOAD_N 个任务的 Mm2/Vec2 排空
        // （isValid 仅在 Mm2/Vec2 完成后清零，drain 阶段前 PRELOAD_N 次只执行 Mm2/Vec2 跳过 Mm1/Vec1）
        return executedTaskCount < createdTaskCount + PRELOAD_N;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        // 用 nextBatchBN2Start_ 代替每次 bN2Cur/n2Size 除法：仅比较一次整数
        if (isFirstTask || bN2Cur >= nextBatchBN2Start_) {
            uint32_t bIdx = bN2Cur / constInfo.n2Size;   // 仅换 batch 时执行一次除法
            prevBIdx = bIdx;
            nextBatchBN2Start_ = (bIdx + 1) * constInfo.n2Size;
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
            // 换 batch 时才重算（正确位置）
            cachedS2LoopTimes  = static_cast<uint32_t>((actSeqLensKv + s2BaseSize - 1) / s2BaseSize);
            uint64_t gS1Size_  = actSeqLensQ * constInfo.gSize;
            cachedGS1Size_     = gS1Size_;   // 供 CalcParams / CalcCurS2StartEnd 免重复乘法
            cachedGS1LoopTimes = static_cast<uint32_t>((gS1Size_ + mBaseSize - 1) / mBaseSize);
            GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv,
                                  cachedPreTokensLeftUp_, cachedNextTokensLeftUp_);
        }
        // 仅在换 batch 时重新计算（已移入 if 块，消除热路径每次调用的除法）
        uint32_t s2LoopTimes  = cachedS2LoopTimes;
        uint32_t gS1LoopTimes = cachedGS1LoopTimes;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
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
        curS2End = cachedS2LoopTimes;  // 复用缓存，避免重复除法

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
        // 1. 直接使用 GetTaskDealMode 换 batch 时已缓存的结果，免重复调用和条件判断
        int64_t preTokenLeftUp  = cachedPreTokensLeftUp_;
        int64_t nextTokenLeftUp = cachedNextTokensLeftUp_;

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        // cachedGS1Size_ = actSeqLensQ * gSize，换 batch 时更新，免重复乘法
        int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(cachedGS1Size_)) - 1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            if (constInfo.gSize == 1) {
                s1FirstToken = s1GFirstToken;
                s1LastToken  = s1GLastToken;
            } else {
                s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
                s1LastToken  = static_cast<int64_t>(s1GLastToken  / constInfo.gSize);
            }
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
            }
        }

        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start, curS2End
        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            headS2Split = s2OStart_ > s2StartWithSparse ? true : false;
            curS2Start = Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            tailS2Split = s2OEnd_ > 0U ? true : false;
            curS2End = s2OEnd_ > 0U ? Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];                  // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK]; // 上PRELOAD_N轮任务
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
        cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        if constexpr (BMM2_TOUB) {
            cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo);
        } else {
            cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo);
        }
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        vecFaBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        if constexpr (BMM2_TOUB) {
            this->vecFaBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo);
        } else {
            this->vecFaBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo);
        }
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        // ── batch 级缓存（bN2 变化才更新）──
        if (bN2Cur != cachedBN2ForCalc) {
            cachedBN2ForCalc   = bN2Cur;
            cachedBIdxForCalc  = bN2Cur / constInfo.n2Size;
            cachedN2IdxForCalc = bN2Cur % constInfo.n2Size;
        }
        info.loop  = loop;
        info.bIdx  = cachedBIdxForCalc;
        info.n2Idx = cachedN2IdxForCalc;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;
        info.preTokensLeftUp  = cachedPreTokensLeftUp_;
        info.nextTokensLeftUp = cachedNextTokensLeftUp_;

        // ── row 级缓存（bN2Cur 或 gS1Cur 变化才更新，inner-S2 loop 直接读缓存）──
        if (bN2Cur != cachedRowBN2_ || gS1Cur != cachedRowGS1_) {
            cachedRowBN2_ = bN2Cur;
            cachedRowGS1_ = gS1Cur;
            cachedRowMloop_  = mloop;
            cachedRowGS1Idx_ = static_cast<uint64_t>(gS1Cur) * mBaseSize;
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                          LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                cachedRowS1Idx_ = cachedRowGS1Idx_ / constInfo.gSize;
            } else {
                cachedRowS1Idx_ = cachedRowGS1Idx_ % actSeqLensQ;
            }
            cachedRowActMSize_ = mBaseSize;
            if ((cachedRowGS1Idx_ + mBaseSize) > cachedGS1Size_) {
                cachedRowActMSize_ = static_cast<uint32_t>(cachedGS1Size_ - cachedRowGS1Idx_);
            }
            if constexpr (USE_DN) {
                cachedRowActMSizeA32_ = (cachedRowActMSize_ + 31) >> 5 << 5;
                uint32_t vBase = cachedRowActMSize_ <= 16 ? cachedRowActMSize_ : (cachedRowActMSizeA32_ >> 1);
                if (constInfo.subBlockIdx == 1) {
                    cachedRowVecMbaseIdx_ = vBase;
                    cachedRowActVecMSize_ = cachedRowActMSize_ - vBase;
                } else {
                    cachedRowVecMbaseIdx_ = 0;
                    cachedRowActVecMSize_ = vBase;
                }
            } else {
                uint32_t vBase = (cachedRowActMSize_ + 1) >> 1;
                if (constInfo.subBlockIdx == 1) {
                    cachedRowVecMbaseIdx_ = vBase;
                    cachedRowActVecMSize_ = cachedRowActMSize_ - vBase;
                } else {
                    cachedRowVecMbaseIdx_ = 0;
                    cachedRowActVecMSize_ = vBase;
                }
            }
            cachedRowIsS2Split_   = false;
            cachedRowTmpOutWsPos_ = coreFirstTmpOutWsPos_;
            if (bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_) {
                cachedRowIsS2Split_ = true;
            } else if (headS2Split && bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) {
                cachedRowIsS2Split_ = true;
            } else if (tailS2Split && bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) {
                cachedRowIsS2Split_   = true;
                cachedRowTmpOutWsPos_ = headS2Split ? (coreFirstTmpOutWsPos_ + 1) : coreFirstTmpOutWsPos_;
            }
        }
        // ── 填入 row 级缓存字段（O(1) 读取，无计算）──
        info.mloop           = cachedRowMloop_;
        info.gS1Idx          = cachedRowGS1Idx_;
        info.s1Idx           = cachedRowS1Idx_;
        info.actMSize        = cachedRowActMSize_;
        info.actMSizeAlign32 = cachedRowActMSizeA32_;
        info.actVecMSize     = cachedRowActVecMSize_;
        info.vecMbaseIdx     = cachedRowVecMbaseIdx_;
        info.isS2SplitCore   = cachedRowIsS2Split_;
        info.faTmpOutWsPos   = cachedRowTmpOutWsPos_;

        // ── tile 级字段（每个 S2 tile 都不同）──
        info.s2Idx = s2Cur * s2BaseSize;
        info.actSingleLoopS2Size = s2BaseSize;
        if ((info.s2Idx + s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = static_cast<uint32_t>(info.actS2Size) - info.s2Idx;
        }
        info.actSingleLoopS2SizeAlign =
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));
        if (constInfo.isKvContinuous) {
            info.isChangeBatch = false;
        } else {
            // for tensor-list
            if (loop == 0) { // 第一个有效任务才需要重置KV的tensor
                info.isChangeBatch = true;
            } else {
                info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
            }
        }
        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop  = (s2Cur + 1 == curS2End);
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint32_t s2LoopTimes  = cachedS2LoopTimes;   // 已在 GetTaskDealMode 换 batch 时更新
        uint32_t gS1LoopTimes = cachedGS1LoopTimes;

        // 当前S2未处理完
        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }

        if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK || taskDealMode == TASK_DEAL_MODE::S2_END ||
            taskDealMode == TASK_DEAL_MODE::SKIP_S1OUT) {
            accGS1Loops++;
        }

        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        if ((taskDealMode == TASK_DEAL_MODE::DEAL_ZERO || taskDealMode == TASK_DEAL_MODE::SKIP_ZERO)) {
            accGS1Loops++;
        }

        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!isFd) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock.InitBuffers(this->pipe, SharedBuffer1, SharedBuffer2, SharedBuffer3);
        AscendC::ICachePreLoad(2);
        vecFdBlock.AllocEventID();
        vecFdBlock.InitDecodeParams();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams_);
        SyncAll();
        vecFdBlock.FreeBuffers(SharedBuffer1, SharedBuffer2);
        vecFdBlock.FreeEventID();
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

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_START_INDEX, sectionIdx));
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
}; // FlashAttentionNoQuantGqaKernel

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_