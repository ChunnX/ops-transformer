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
#include "kernel_operator_list_tensor_intf.h"    //  TensorDesc
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_utils.h"

#include "fa_block_cube_noquant_gqa.h"
#include "fa_block_vec_noquant_gqa.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "flash_attention_noquant_block_vec_flashdecode.h"    // FiaBlockVecFlashDecode


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
    static constexpr bool SPLITD = CubeBlockType::SPLITD;
    static constexpr bool HAS_PREFIX = CubeBlockType::HAS_PREFIX;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;


    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr bool HAS_ROPE = CubeBlockType::HAS_ROPE;
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;
 
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT; // V100 只支持一种??
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    BufferManager<BufferType::GM> gmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<INPUT_T> sinkGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoX constInfo;

    const FlashAttnNoQuantTilingArch35 *__restrict tilingData;
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
    uint32_t reserved_;
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
    TQue<QuePosition::VECOUT, 1> SharedBuffer1[2];  // 共用
    TQue<QuePosition::VECIN, 1> SharedBuffer2[2];  // 共用
    TBuf<> SharedBuffer3; // 共用

    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantGqaKernel() : cubeBlock(constInfo), vecFaBlock(constInfo), vecFdBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                                __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
                                __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                const FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tilingData = tiling;

        reserved_ = ((__gm__ uint32_t *)fiaMetaData)[0];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
            FA_AIC_CORE_NUM * 16U * reserved_);
        fdMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
            FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * reserved_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * reserved_);

        InitConstInfo(queryPaddingSize, kvPaddingSize, actualSharedPrefixLen);
        

        keyPtr = key;
        valuePtr = value;

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualSeqLenSize);
        qActSeqLensParser.Init(actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);

        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsKv, constInfo.actualSeqLenKVSize);
        kvActSeqLensParser.Init(actualSeqLengthsGm, constInfo.actualSeqLenKVSize, constInfo.s2Size);

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            vecFaBlock.InitVecBlock(tPipe, pse, actualSeqLengths, actualSeqLengthsKv, postQuantScale, postQuantOffset,
                attenMask, learnableSink, softmaxLse, attentionOut, workspace);
            vecFaBlock.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blockTable, queryRope, keyRope,
                actualSeqLengths, actualSeqLengthsKv, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen);
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                vecFdBlock.InitParams();
                vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                            this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm,
                                            this->actualSeqLengthsGmQ, this->actualSeqLengthsGm, keyPtr, postQuantScale,
                                            postQuantOffset);
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

    __aicore__ inline void InitConstInfo(__gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                                         __gm__ uint8_t *actualSharedPrefixLen)
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
        auto fiaPseParams = this->tilingData->flashAttnPseParams;
        auto fiaSystemPrefixParams = this->tilingData->flashAttnSystemPrefixParams;
        auto fiaPageAttentionParams = this->tilingData->flashAttnPageAttentionParams;
        auto fiaLeftPaddingParams = this->tilingData->flashAttnLeftPaddingParams;
        auto fiaPostQuantParams = this->tilingData->flashAttnPostQuantParams;
        auto fiaWorkspaceParams = this->tilingData->flashAttnWorkspaceParams;
        auto fiaS1OuterSplitCoreParams = this->tilingData->flashAttnS1OuterSplitCoreParams;
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
        if constexpr (HAS_ROPE) {
            constInfo.dSizeRope = fiaBaseParams.dSizeRope;
        } else {
            constInfo.dSizeRope = 0;
        }
        constInfo.actualSeqLenSize = fiaBaseParams.cuSeqLengthsQSize;
        constInfo.actualSeqLenKVSize = fiaBaseParams.cuSeqLengthsKVSize;
        constInfo.scaleValue = static_cast<float>(fiaBaseParams.scaleValue);
        constInfo.isKvContinuous = fiaBaseParams.isKvContinuous != 0;
        constInfo.coreNum = fiaBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FA_LAYOUT>(fiaBaseParams.outputLayout);

        // if constexpr (HAS_MASK) {
        constInfo.sparseMode = fiaAttenMaskParams.sparseMode;      
        constInfo.preTokens = fiaAttenMaskParams.winLefts;
        constInfo.nextTokens = fiaAttenMaskParams.winRights;
        constInfo.attenMaskBatch = fiaAttenMaskParams.attenMaskBatch;
        constInfo.attenMaskS1Size = fiaAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = fiaAttenMaskParams.attenMaskS2Size;
        constInfo.isRowInvalidOpen = fiaAttenMaskParams.isRowInvalidOpen;
        constInfo.isExistRowInvalid = fiaAttenMaskParams.isExistRowInvalid;
        // }

        if ASCEND_IS_AIV {
            if constexpr (VecFaBlockType::hasPse) {
                constInfo.pseShiftByBatch = fiaPseParams.pseShiftByBatch;
                constInfo.pseS1Size = fiaPseParams.pseS1Size;
                constInfo.pseS2Size = fiaPseParams.pseS2Size;
                constInfo.pseStride = s2BaseSize;
            }
        }

        constInfo.isPostQuantPerChnl = fiaPostQuantParams.isPostQuantPerChnl;
        constInfo.isPostQuantBF16 = fiaPostQuantParams.isPostQuantBF16;

        constInfo.accumOutSize = fiaWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = fiaWorkspaceParams.logSumExpSize;

        constInfo.isQHasLeftPadding = fiaLeftPaddingParams.isQHasLeftPadding;
        constInfo.isKVHasLeftPadding = fiaLeftPaddingParams.isKVHasLeftPadding;
        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = fiaPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = fiaPageAttentionParams.blockSize;
            constInfo.paLayoutType = fiaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = fiaBaseParams.isSoftMaxLseEnable;
        // prefix
        if constexpr (HAS_PREFIX) {
            constInfo.isActualSharedPrefixLenNull = fiaSystemPrefixParams.isActualSharedPrefixLenNull;
            constInfo.actualKVPrefixSize = constInfo.kvPrefixSize = fiaSystemPrefixParams.prefixSeqInnerSize;
            if (!constInfo.isActualSharedPrefixLenNull) {
                constInfo.actualKVPrefixSize = ((__gm__ int64_t *)actualSharedPrefixLen)[0];
            }
        }

        if (constInfo.isQHasLeftPadding) {
            constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
            if (constInfo.queryRightPaddingSize < 0) {
                constInfo.queryRightPaddingSize = 0;
            }
        }

        if (constInfo.isKVHasLeftPadding) {
            constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
            if (constInfo.kvRightPaddingSize < 0) {
                constInfo.kvRightPaddingSize = 0;
            }
        }

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t reservedIdx)
    {
        // AICPU metadata format: 16 fields per AIC core, 0-indexed (no leading CORE_ENABLE).
        // Kernel field constants ( FLASH_ATTN_BN2_START_INDEX=1, etc.) are 1-based, so subtract 1.
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * reservedIdx + 16U * coreIdx + metaIdx;
    }
    
    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t reservedIdx)
    {
        return FA_FD_METADATA_SIZE * FA_AIV_CORE_NUM * reservedIdx + FA_FD_METADATA_SIZE * coreIdx + metaIdx ;
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
    __aicore__ inline void FlashAttention(uint32_t reservedIdx)
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        GetFASectionInfo(reservedIdx);
        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // Reset pipeline state for each section to avoid cross-section deadlock
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
        for (uint32_t i = 0; i < PRELOAD_TASK_CACHE_SIZE; i++) {
            if (taskRunInfo[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if (constInfo.actualSeqLenKVSize == 0 && !constInfo.isKvContinuous) {
                actSeqLensKv = SeqLenFromTensorList<LAYOUT_KV>(keyPtr, bIdx);
            } else {
                actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx);
            }
            actSeqLensKv += constInfo.actualKVPrefixSize;
            actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
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
        // 对paddingSize设置不合理的任务结果置0
        if ((constInfo.isQHasLeftPadding && (actSeqLensQ + constInfo.queryRightPaddingSize > constInfo.s1Size)) ||
            (constInfo.isKVHasLeftPadding && (actSeqLensKv + constInfo.kvRightPaddingSize > constInfo.s2Size))) {
            return TASK_DEAL_MODE::DEAL_ZERO;
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

        if (constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL || constInfo.sparseMode == fa_base_vector::TREE) {
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
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken = Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.gSize)) - 1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.gSize);
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
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
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
        RunInfoX &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
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
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.gSize;
        } else {
            // GS1 layout
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
            Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

        if (constInfo.isQHasLeftPadding) {
            info.qPaddingBeginOffset = constInfo.s1Size - actSeqLensQ - constInfo.queryRightPaddingSize;
        }
        if (constInfo.isKVHasLeftPadding) {
            info.kvPaddingBeginOffset = constInfo.s2Size - actSeqLensKv - constInfo.kvRightPaddingSize;
        }

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

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
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
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
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

        // 当前S2未处理完
        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }

        if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK ||
            taskDealMode == TASK_DEAL_MODE::S2_END ||
            taskDealMode == TASK_DEAL_MODE::SKIP_S1OUT) {
            accGS1Loops++;
        }

        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        if ((taskDealMode == TASK_DEAL_MODE::DEAL_ZERO ||
             taskDealMode == TASK_DEAL_MODE::SKIP_ZERO)) {
            accGS1Loops++;
        }

        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t reservedIdx)
    {
        vecFdBlock.InitBuffers(this->pipe, SharedBuffer1, SharedBuffer2, SharedBuffer3);
        AscendC::ICachePreLoad(2);
        GetFDSectionInfo(reservedIdx);
        vecFdBlock.AllocEventID();
        vecFdBlock.InitDecodeParams();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams_);
        SyncAll();
        vecFdBlock.FreeBuffers(SharedBuffer1, SharedBuffer2);
        vecFdBlock.FreeEventID();
    }

    __aicore__ inline void GetFASectionInfo(uint32_t reservedIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX, reservedIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX, reservedIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX, reservedIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX, reservedIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX, reservedIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX, reservedIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, reservedIdx)
        );
        // PRINTF("ori bN2Start_ is %d, gS1OStart_ is %d, s2OStart_ is %d\n", bN2Start_, gS1OStart_, s2OStart_);
        // PRINTF("ori bN2End_ is %d, gS1OEnd_ is %d, s2OEnd_ is %d\n", bN2End_, gS1OEnd_, s2OEnd_);

    }

    __aicore__ inline void GetFDSectionInfo(uint32_t reservedIdx)
    {
        fdParams_.fdBN2Idx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_BN2_IDX_INDEX, reservedIdx));
        fdParams_.fdMIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_IDX_INDEX, reservedIdx));
        fdParams_.fdWorkspaceIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_IDX_INDEX, reservedIdx));
        fdParams_.fdS2SplitNum =fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_NUM_INDEX, reservedIdx));
        fdParams_.mStart =fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_START_INDEX, reservedIdx));
        fdParams_.mLen = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_NUM_INDEX, reservedIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        // FDparamsX fdParams = {fdCoreEnable, fdBN2Idx, fdMIdx, fdS2SplitNum, mStart, mLen, fdWorkspaceIdx};
        // PRINTF("ori fdParams_.fdBN2Idx is %d, fdParams_.fdMIdx is %d, fdParams_.fdWorkspaceIdx is %d, fdParams_.fdS2SplitNum is %d\n", fdParams_.fdBN2Idx, fdParams_.fdMIdx, fdParams_.fdWorkspaceIdx, fdParams_.fdS2SplitNum);
        // PRINTF("ori fdParams_.mStart is %d, fdParams_.mLen is %d\n", fdParams_.mStart, fdParams_.mLen);
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
        for (uint32_t reservedIdx = 0; reservedIdx < reserved_; reservedIdx++) {
            if (constInfo.aicIdx < constInfo.coreNum) {
                FlashAttention(reservedIdx);
            }
            if constexpr (FLASH_DECODE) {
                if ASCEND_IS_AIV {
                    FlashDecode(reservedIdx);
                }
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