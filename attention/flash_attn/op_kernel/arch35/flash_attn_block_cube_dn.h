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
 * \file flash_attn_block_cube_dn.h
 * \brief FANoQuantGqaBlockCubeDn
 */
#ifndef FLASH_ATTN_BLOCK_CUBE_DN_H_
#define FLASH_ATTN_BLOCK_CUBE_DN_H_

#include "../utils/flash_attn_type.h"
#include "../utils/flash_attn_common_def.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/arch_info.h"

using namespace fa_base_matmul;

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;

template <typename FA_T>
class FANoQuantGqaBlockCubeDn {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;

    static constexpr FixpipeConfig FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = float;

    const ConstInfo_t &constInfo_;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool_;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<LAYOUT_KV, PAGE_ATTENTION>();
    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, SEQLEN_T, IS_TND<LAYOUT_T>()>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, SEQLEN_T, IS_TND<LAYOUT_KV>()>;
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    GlobalTensor<int32_t> blockTableGm_;

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_MM_0 = 0U;
    static constexpr uint32_t CROSSCORE_MM_1 = 1U;
    static constexpr uint32_t CROSSCORE_MM_2 = 2U;
    static constexpr uint32_t CROSSCORE_MM_3 = 3U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;
    // C2决策完成ack(每tile一个): V2读取GM标志前必须等待, 该ack在两个AIV的L1P等待之后置位,
    // 保证两个半块的v1标志GM写入对V2可见(含skip情形无fixpipe, 无CC_MM可接力)
    // 3槽轮转: V1(loop+3)被AIV程序序约束在V2(loop)之后, C2(loop+3)晚于V2(loop)消费同槽
    static constexpr uint32_t CROSSCORE_C2DONE_0 = 8U;
    static constexpr uint32_t C2DONE_RING = 3U;

    // 核内同步ID
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 0U;
    static constexpr uint32_t Q_L1_BUFFER_ID1 = 1U;
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 2U;
    static constexpr uint32_t KV_L1_BUFFER_ID1 = 3U;
    static constexpr uint32_t KV_L1_BUFFER_ID2 = 4U;
    static constexpr uint32_t KV_L1_BUFFER_ID3 = 5U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 6U;
    static constexpr uint32_t L0A_BUFFER_ID1 = 7U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 8U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 9U;
    static constexpr uint32_t L0C_BUFFER_ID0 = 10U;
    static constexpr uint32_t L0C_BUFFER_ID1 = 11U;
    static constexpr uint32_t L0C_BUFFER_ID2 = 12U;
    static constexpr uint32_t L0C_BUFFER_ID3 = 13U;

    // UB
    static constexpr uint32_t UB_MM_RES_BUFCNT = (dBaseSize > 128) ? 2U : 4U;
    static constexpr uint32_t UB_MM_RES_BUF_BYTES =
        mBaseSize / CV_RATIO * (s2BaseSize > dVBaseSize ? s2BaseSize : dVBaseSize) * sizeof(MM_T);
    LocalTensor<uint8_t> ubMmResBuffers_;
    // L1
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_KV_BUFCNT = 4U;
    static constexpr uint32_t L1_KV_BUF_BYTES = 64U * 1024U;
    // buffer位置+用途+Buffers, 例如l1PBuffers; 使用时命名: 用途+buffer位置+Tensor, 例如pL1Tensor
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    // L0C
    // 瞬态L0C仅轮转id 0~2; id 3 固定用作 Oj 的跨tile累加器(v2-skip场景BMM2在L0C上原子累加),
    // 累加器被 PIPE_M mutex 长期持有, 瞬态轮转必须避开, 否则BMM1取同一id会死锁
    static constexpr uint32_t L0C_TRANS_BUFCNT = 3U;
    static constexpr uint32_t L0C_ACC_BUF_ID = 3U;
    static_assert(dVBaseSize <= 128, "L0C accum mode requires single N tile (dVBaseSize <= 128)");
    static_assert(s2BaseSize == 128, "L0C accum mode requires single K tile (s2BaseSize == 128, DN configs 0/2/6)");
    static constexpr uint32_t L0C_BUFCNT = 4U;
    static constexpr uint32_t L0C_BUF_BYTES = 64U * 1024U;
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;
    uint32_t mmResBufId_ = 0U;
    // L0C Oj累加器状态: true表示id L0C_ACC_BUF_ID持有当前行(gS1)的部分和O
    bool accValid_ = false;
    // v1写出的块级max变化标志(v2-skip), C2读取后决定BMM2累加/flush
    GlobalTensor<float> faSkipFlagGm_;
    // L0A/B
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0A> l0aBufferManager_;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0B> l0bBufferManager_;
    using L0APolicyType =
        BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    using L0BPolicyType =
        BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    L0APolicyType mmL0APolicy_;
    L0BPolicyType mmL0BPolicy_;

    __aicore__ inline FANoQuantGqaBlockCubeDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                              SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *blockTable, __gm__ uint8_t *workspace)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        faSkipFlagGm_.SetGlobalBuffer((__gm__ float *)(workspace + GetFaSkipFlagGmOffset(
                                                          PAGE_ATTENTION, constInfo_.coreNum,
                                                          constInfo_.accumOutSize, constInfo_.logSumExpSize)));

        InitQBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyBnStride, constInfo_.keyN2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueBnStride, constInfo_.valueN2Stride);
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMmResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES);

        /*--------------------------------------------L1--------------------------------------------*/
        uint32_t addrL1 = 0;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);
        addrL1 = L1_P_BUFCNT * L1_P_BUF_BYTES;
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_Q_BUFCNT * L1_Q_BUF_BYTES);
        addrL1 += L1_Q_BUFCNT * L1_Q_BUF_BYTES;
        l1KvBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * L1_KV_BUF_BYTES);

        // /*--------------------------------------------L0A--------------------------------------------*/
        l0aBufferManager_.Init(BUFFER_SIZE_BYTE_64K);

        // /*--------------------------------------------L0B--------------------------------------------*/
        l0bBufferManager_.Init(BUFFER_SIZE_BYTE_64K);

        /*--------------------------------------------L0C--------------------------------------------*/
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool_.seqUsedParser);
        } else {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm, uint64_t bnStride = 0, uint64_t n2Stride = 0)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, kvSeqLensTool_.seqUsedParser);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_1 + AIV0_AIV1_OFFSET);
        if constexpr (dBaseSize <= 128) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_2 + AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_3 + AIV0_AIV1_OFFSET);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mmL0APolicy_.Init(l0aBufferManager_, BUFFER_SIZE_BYTE_32K, L0A_BUFFER_ID0, L0A_BUFFER_ID1);
        mmL0BPolicy_.Init(l0bBufferManager_, BUFFER_SIZE_BYTE_32K, L0B_BUFFER_ID0, L0B_BUFFER_ID1);
    }

    __aicore__ inline void FreeEventID()
    {
        mmL0APolicy_.Uninit(l0aBufferManager_);
        mmL0BPolicy_.Uninit(l0bBufferManager_);
    }

    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfo &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.n2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = dOffset,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dRealSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfo &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfo &runInfo)
    {
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void ResetAccState()
    {
        accValid_ = false;
    }

    // mmRes UB buffer 与 CC_MM flag 一一对应, AIC/AIV 两侧按相同的消费序列推进计数器
    __aicore__ inline uint32_t AllocMmResBufId()
    {
        uint32_t bufId = mmResBufId_;
        mmResBufId_ = (mmResBufId_ + 1) % UB_MM_RES_BUFCNT;
        return bufId;
    }

    __aicore__ inline LocalTensor<MM_T> GetAccL0CTensor()
    {
        return l0CBuffers_[L0C_ACC_BUF_ID * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
    }

    // 读取v1写出的块级max变化标志(两个AIV半块任一非0即为true)
    __aicore__ inline bool ReadBlockFlushFlag(uint32_t loop)
    {
        uint32_t elemBase = constInfo_.aicIdx * FA_SKIP_FLAG_AIC_FLOATS +
                            (loop % FA_SKIP_FLAG_LOOPS) * (FA_SKIP_FLAG_SUBS * FA_SKIP_FLAG_SLOT_FLOATS);
        float flag0 = faSkipFlagGm_.GetValue(elemBase);
        float flag1 = faSkipFlagGm_.GetValue(elemBase + FA_SKIP_FLAG_SLOT_FLOATS);
        return (flag0 != 0.0f) || (flag1 != 0.0f);
    }

    __aicore__ inline void IterateBmm1(RunInfo &runInfo)
    {
        uint32_t mmResUbBufId = AllocMmResBufId();
        LocalTensor<MM_T> mm1ResUbTensor =
            ubMmResBuffers_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<MM_T>();
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm1Dn(mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                        RunInfo &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = runInfo.actSingleLoopS2Size;
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm1Dn(LocalTensor<MM_T> &mm1ResUbTensor, RunInfo &runInfo)
    {
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            CopyQuerySlice(qL1Tensor, 0, constInfo_.dSize, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyKeySlice(kL1Tensor, 0, constInfo_.dSize, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            // Mmad K=D 精确累加; LoadData 搬运宽度按分形粒度对齐(对齐D=原值, D=72→80与Nd2Nz写入组数一致)
            uint32_t ldK = ((constInfo_.dSize + 15) >> 4) << 4;
            MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                        constInfo_.dSize, false, true, true, true, 0, 0, ldK, 0);
            if constexpr (dBaseSize > 128) {
                MatmulK<KV_T, Q_T, MM_T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(kL1Tensor, qL1Tensor, mmL0APolicy_,
                                                                                    mmL0BPolicy_, l0CSubTensor, param);
            } else {
                MatmulBase<KV_T, Q_T, MM_T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
                    kL1Tensor, qL1Tensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor, param);
            }
            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

            FixpipeMm1Dn(mm1ResUbTensor, l0CSubTensor, runInfo);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_TRANS_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<MM_T> &l0C,
                                              uint32_t realN, RunInfo &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = (mBaseSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = (dVBaseSize + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    // BMM2 的 L0C 累加器写法: cmatrixInitVal=true 时重置C矩阵(行首tile/flush后首个tile),
    // false 时在已有部分和上原子累加(mj==mj-1 的 tile, Oj = Oj-1 + exp(Sj-mj)*Vj)
    __aicore__ inline void IterateBmm2Acc(const LocalTensor<Q_T> &pL1Tensor, const LocalTensor<KV_T> &vL1Tensor,
                                          RunInfo &runInfo, bool cmatrixInitVal)
    {
        if (cmatrixInitVal) {
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + L0C_ACC_BUF_ID);
        }
        LocalTensor<MM_T> accL0C = GetAccL0CTensor();
        MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo_.dSizeV,
                                    (uint32_t)runInfo.actSingleLoopS2Size, true, false, cmatrixInitVal, cmatrixInitVal,
                                    0, 0, 0, AttentionCommon::Align(constInfo_.dSizeV, 16U));
        MatmulFull<Q_T, KV_T, MM_T, 128, dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
            pL1Tensor, vL1Tensor, mmL0APolicy_, mmL0BPolicy_, accL0C, param);
        accValid_ = true;
    }

    // L0C 累加器 flush 到 UB(供 v2 合并/归一), 释放累加器占用
    __aicore__ inline void FlushAccToUb(LocalTensor<MM_T> &dstUbTensor, RunInfo &runInfo)
    {
        Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + L0C_ACC_BUF_ID);
        Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + L0C_ACC_BUF_ID);
        FixpipeMm2PartialN(dstUbTensor, GetAccL0CTensor(), constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + L0C_ACC_BUF_ID);
        accValid_ = false;
    }

    // 输出一个 BMM2/累加器 fixpipe 结果: 分配mmRes buffer, 等释放, fixpipe, 置就绪
    __aicore__ inline void FixpipeAccToMmRes(RunInfo &runInfo)
    {
        uint32_t mmResUbBufId = AllocMmResBufId();
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;
        LocalTensor<MM_T> mm2ResUbTensor =
            ubMmResBuffers_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<MM_T>();
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        FlushAccToUb(mm2ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void IterateBmm2(RunInfo &runInfo)
    {
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        LocalTensor<Q_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<Q_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);

        // v1写出的块级max变化标志: false表示所有行 mj == mj-1
        bool blockFlush = false;
        if (!runInfo.isFirstS2Loop) {
            blockFlush = ReadBlockFlushFlag(runInfo.loop);
        }
        // 行首tile必然建立累加器; mj==mj-1且非行尾: 累加进L0C, 跳过fixpipe与整个v2阶段
        bool isSkip = (!runInfo.isFirstS2Loop) && (!blockFlush) && (!runInfo.isLastS2Loop);

        LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyValueSlice(vL1Tensor, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);

        if (isSkip) {
            IterateBmm2Acc(pL1Tensor, vL1Tensor, runInfo, false);
        } else {
            if (!runInfo.isFirstS2Loop) {
                // flush: 老累加器(旧max尺度部分和)fixpipe到UB, v2阶段合并并重标定
                FixpipeAccToMmRes(runInfo);
            }
            // 新累加器: 本tile P*V 作为初值(重置C矩阵), 驻留L0C
            IterateBmm2Acc(pL1Tensor, vL1Tensor, runInfo, true);
            if (runInfo.isLastS2Loop) {
                // 行尾: 最终累加器fixpipe到UB, v2阶段完成 (pre+cur)/sum 与写出
                FixpipeAccToMmRes(runInfo);
            }
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        // C2决策完成ack: 通知V2可以读取GM标志(本tile的UB产出/跳过决策均已确定)
        uint32_t doneSyncIdx = CROSSCORE_C2DONE_0 + (runInfo.loop % C2DONE_RING);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(doneSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(doneSyncIdx + AIV0_AIV1_OFFSET);
    }
}; // FANoQuantGqaBlockCubeDn

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockCubeDummyDn {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockCubeDummyDn(ConstInfo_t &constInfo,
                                                   SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                   SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace FlashAttnKernel

#endif // FLASH_ATTN_BLOCK_CUBE_DN_H_
