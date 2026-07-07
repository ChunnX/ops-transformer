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
 * \file quant_flash_attn_kernel_fullquant_mx.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_FULLQUANT_MX_H_
#define QUANT_FLASH_ATTN_KERNEL_FULLQUANT_MX_H_

#include "quant_flash_attn_public_define_arch35.h"
#include "quant_flash_attn_block_cube_fullquant_mx.h"
#include "quant_flash_attn_block_vec_fullquant_mx.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "quant_flash_attn_block_vec_flashdecode_fullquant.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "quant_flash_attn_tiling_data.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionFullQuantMxKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
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
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<uint64_t> cuSeqLensGmQ;
    GlobalTensor<uint64_t> cuSeqLensGmKv;
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<float> softmaxLseGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoX constInfo;

    const QuantFlashAttnTilingData *__restrict tilingData;
    TPipe *pipe = nullptr;
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    uint32_t createdTaskCount = 0U;
    uint32_t executedTaskCount = 0U;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t prevBIdx = 0;
    uint32_t prevBN2Idx = 0;
    uint32_t prevGS1Idx = 0;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;

    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionFullQuantMxKernel()
        : cubeBlock(constInfo), vecFaBlock(constInfo), vecFdBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *attnMask, __gm__ uint8_t *cuSeqLensQ,
                                __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attnOut,
                                __gm__ uint8_t *workspace, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
                                const QuantFlashAttnTilingData *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tilingData = tiling;

        metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata,
                                      QFA_AIC_CORE_NUM * QFA_METADATA_SIZE + QFA_AIV_CORE_NUM * QFA_FD_METADATA_SIZE);

        InitConstInfo();

        keyPtr = key;
        valuePtr = value;

        cuSeqLensGmQ.SetGlobalBuffer((__gm__ uint64_t *)cuSeqLensQ, constInfo.actualSeqLenSize);
        qActSeqLensParser.Init(cuSeqLensGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);

        cuSeqLensGmKv.SetGlobalBuffer((__gm__ uint64_t *)cuSeqLensKv, constInfo.actualSeqLenKVSize);
        kvActSeqLensParser.Init(cuSeqLensGmKv, constInfo.actualSeqLenKVSize, constInfo.s2Size);

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            vecFaBlock.InitVecBlock(tPipe, cuSeqLensQ, cuSeqLensKv, pScale, attnMask, softmaxLse,
                                    attnOut, workspace);
            vecFaBlock.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blockTable, nullptr, nullptr,
                                    cuSeqLensQ, cuSeqLensKv, dequantScaleQuery, dequantScaleKey,
                                    dequantScaleValue);
        }
        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                vecFdBlock.InitParams();
                vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                            this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm,
                                            this->cuSeqLensGmQ, this->cuSeqLensGmKv, keyPtr, nullptr,
                                            nullptr);
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

        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype / 2;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T) + mBaseSize * s2BaseSize / 32;
        l1BufferManager.Init(pipe, 524288); // 512 * 1024
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        // TODO，共享buffer初始化放到block层
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
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

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        auto qfaBaseParams = this->tilingData->baseTiling.quantFlashAttnBaseParams;
        auto qfaAttenMaskParams = this->tilingData->baseTiling.quantFlashAttnAttenMaskParams;
        auto qfaPageAttentionParams = this->tilingData->baseTiling.quantFlashAttnPageAttentionParams;
        auto qfaWorkspaceParams = this->tilingData->baseTiling.quantFlashAttnWorkspaceParams;

        constInfo.bSize = qfaBaseParams.bSize;
        constInfo.t1Size = qfaBaseParams.t1Size;
        constInfo.t2Size = qfaBaseParams.t2Size;
        constInfo.n2Size = qfaBaseParams.n2Size;
        constInfo.gSize = qfaBaseParams.gSize;
        constInfo.s1Size = qfaBaseParams.s1Size;
        constInfo.s2Size = qfaBaseParams.s2Size;
        constInfo.dSize = qfaBaseParams.dSize;
        constInfo.dSizeV = qfaBaseParams.dSizeV;
        if constexpr (USE_DN) { // prefill不合轴
            constInfo.realN2Size = constInfo.n2Size * constInfo.gSize;
            constInfo.realGSize = 1;
        } else { // decode合轴
            constInfo.realN2Size = constInfo.n2Size;
            constInfo.realGSize = constInfo.gSize;
        }
        constInfo.actualSeqLenSize = qfaBaseParams.cuSeqLensQSize;
        constInfo.actualSeqLenKVSize = qfaBaseParams.cuSeqLensKVSize;
        constInfo.scaleValue = static_cast<float>(qfaBaseParams.scaleValue);
        constInfo.isKvContinuous = qfaBaseParams.isKvContinuous != 0;
        constInfo.coreNum = qfaBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FIA_LAYOUT>(qfaBaseParams.outputLayout);
        // if constexpr (HAS_MASK) {
        constInfo.sparseMode =
            qfaAttenMaskParams.sparseMode; // TODO，后续sparseType、attenMaskCompressMode引用全部改成sparseMode
        constInfo.preTokens = qfaAttenMaskParams.winLefts;
        constInfo.nextTokens = qfaAttenMaskParams.winRights;
        constInfo.attenMaskBatch = qfaAttenMaskParams.attenMaskBatch;
        constInfo.attenMaskS1Size = qfaAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = qfaAttenMaskParams.attenMaskS2Size;
        constInfo.isExistRowInvalid = qfaAttenMaskParams.isExistRowInvalid;
        // }

        constInfo.accumOutSize = qfaWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = qfaWorkspaceParams.logSumExpSize;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = qfaPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = qfaPageAttentionParams.blockSize;
            constInfo.paLayoutType = qfaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = qfaBaseParams.isSoftMaxLseEnable;

        // 任务起始位置
        constInfo.bN2Start = metadataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_BN2_START_INDEX));
        constInfo.gS1OStart = metadataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_M_START_INDEX));
        constInfo.s2OStart = metadataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_S2_START_INDEX));
        constInfo.bN2End = metadataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_BN2_END_INDEX));
        constInfo.gS1OEnd = metadataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_M_END_INDEX));
        constInfo.s2OEnd = metadataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_S2_END_INDEX));
        constInfo.coreFirstTmpOutWsPos =
            metadataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, QFA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));
        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return QFA_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return QFA_METADATA_SIZE * QFA_AIC_CORE_NUM + QFA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if constexpr (BMM2_TOUB) {
            if ASCEND_IS_AIV {
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
            }
        }
    }

    __aicore__ inline void FlashAttention()
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE];
        uint32_t bN2Cur = constInfo.bN2Start;
        uint32_t gS1Cur = constInfo.gS1OStart;
        uint32_t s2Cur = constInfo.s2OStart;
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
        if (bN2Cur != constInfo.bN2End) {
            return bN2Cur < constInfo.bN2End;
        }
        if (gS1Cur != constInfo.gS1OEnd) {
            return gS1Cur < constInfo.gS1OEnd;
        }
        return s2Cur < constInfo.s2OEnd;
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
            (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart) && (s2Cur == constInfo.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo.realN2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if (constInfo.actualSeqLenKVSize == 0 && !constInfo.isKvContinuous) {
                actSeqLensKv = SeqLenFromTensorList<LAYOUT_KV>(keyPtr, bIdx);
            } else {
                actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx);
            }
            actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
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

        // S2有效块区间为[curS2Start, curS2End), s2Cur尚未到达有效区间且该行有有效块,
        // 需快进到curS2Start继续计算, 不能跳行 (BAND等sparse模式curS2Start常>0)
        if (s2Cur < curS2Start && curS2Start < curS2End) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块(curS2Start>=curS2End)或s2Cur已越过有效区间, 跳过当前行
        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
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
        if ((bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
            headS2Split = constInfo.s2OStart != 0U;
            curS2Start = constInfo.s2OStart;
        }

        if ((bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
            tailS2Split = constInfo.s2OEnd != 0U;
            curS2End = constInfo.s2OEnd;
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
        int64_t s1GLastToken = AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.realGSize)) -
                               1;

        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.realGSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.realGSize);
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

        if (bN2Cur == constInfo.bN2Start && gS1Cur == constInfo.gS1OStart) { // first line
            headS2Split = constInfo.s2OStart > s2StartWithSparse ? true : false;
            curS2Start = AttentionCommon::Max(s2StartWithSparse, constInfo.s2OStart);
        }
        if (bN2Cur == constInfo.bN2End && gS1Cur == constInfo.gS1OEnd) { // last line
            tailS2Split = constInfo.s2OEnd > 0U ? true : false;
            curS2End = constInfo.s2OEnd > 0U ? AttentionCommon::Min(s2EndWithSparse, constInfo.s2OEnd) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            uint32_t c1v1Loop = CeilDiv(runInfo0.actSingleLoopS2Size, 256);
            for (uint32_t subLoop = 0; subLoop < c1v1Loop; ++subLoop) {
                if ASCEND_IS_AIC {
                    ComputeMm1(runInfo0, subLoop);
                } else {
                    ComputeVec1(runInfo0, subLoop);
                }
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

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo, uint32_t subLoop)
    {
        cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo, subLoop);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        if constexpr (BMM2_TOUB) {
            cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo);
        } else {
            cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo);
        }
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo, uint32_t subLoop)
    {
        if (subLoop % 2 == 0) {
            vecFaBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo, subLoop);
        } else {
            vecFaBlock.ProcessVec1(this->l1PBuffers.GetPre(), this->bmm1Buffers.Get(), runInfo, subLoop);
        }
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
        info.bIdx = bN2Cur / (constInfo.realN2Size);
        info.n2Idx = (bN2Cur / (constInfo.realN2Size / constInfo.n2Size)) % constInfo.n2Size;
        info.realN2Idx = bN2Cur % constInfo.realN2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_SBH ||
                      LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.realGSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.realGSize;
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
        info.faTmpOutWsPos = constInfo.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);

        if constexpr (USE_DN) {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
        } else {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = (info.actMSize + 1) >> 1;
        }
        info.vecMbaseIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            info.vecMbaseIdx = USE_DN ? info.actVecMSize : (info.actMSizeAlign32 >> 1);
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((constInfo.bN2Start == constInfo.bN2End && constInfo.gS1OStart == constInfo.gS1OEnd)) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
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
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        // 尚未到达有效区间, 快进s2Cur到curS2Start, 不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            // 当前S2未处理完
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }

        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode()
    {
        vecFdBlock.InitBuffers(this->pipe);
        AscendC::ICachePreLoad(2);
        uint32_t fdBN2Idx = metadataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_BN2_IDX_INDEX));
        uint32_t fdMIdx = metadataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_M_IDX_INDEX));
        uint32_t fdS2SplitNum = metadataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_WORKSPACE_NUM_INDEX));
        uint32_t mStart = metadataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_M_START_INDEX));
        uint32_t mLen = metadataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_M_NUM_INDEX));
        uint32_t fdWorkspaceIdx = metadataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, QFA_FD_WORKSPACE_IDX_INDEX));

        FDparamsX fdParams = {fdBN2Idx, fdMIdx, fdS2SplitNum, mStart, mLen, fdWorkspaceIdx};
        vecFdBlock.AllocEventID();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams);
        vecFdBlock.FreeEventID();
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx < constInfo.coreNum) {
            CrossCoreBufferInit();
            if ASCEND_IS_AIV {
                vecFaBlock.InitBuffers();
                vecFaBlock.AllocEventID();
            } else {
                cubeBlock.InitBuffers();
                cubeBlock.AllocEventID();
            }
            FlashAttention();

            if ASCEND_IS_AIV {
                vecFaBlock.FreeEventID();
            } else {
                cubeBlock.FreeEventID();
            }
            CrossCoreBufferUnInit();
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                FlashDecode();
            }
        }
    }
}; // FlashAttentionFullQuantMxKernel

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_KERNEL_FULLQUANT_MX_H_