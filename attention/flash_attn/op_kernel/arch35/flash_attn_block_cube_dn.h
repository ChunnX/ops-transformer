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
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/arch_info.h"
#include "flash_attn_dn_config.h"

using namespace fa_base_matmul;

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;

template <typename FA_T>
class FANoQuantGqaBlockCubeDn {
public:
    using Config = FaDnConfig<FA_T>;
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
    static constexpr uint32_t CROSSCORE_L1P_0 = Config::L1P_SYNC_BASE + 0U;

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

    bool pvDn_ = false;

    // UB
    static constexpr bool SPLIT_MM_QUEUES = Config::SPLIT_MM_QUEUES;
    static constexpr uint32_t UB_MM_RES_BUFCNT = Config::UB_MM_RES_BUFCNT;
    static constexpr uint32_t UB_MM_RES_BUF_BYTES = Config::UB_MM_RES_BUF_BYTES;
    static constexpr uint32_t UB_MM1_BUF_BYTES = Config::UB_MM1_BUF_BYTES;
    static constexpr uint32_t UB_MM2_BUF_BYTES = Config::UB_MM2_BUF_BYTES;
    static constexpr uint32_t UB_MM_RES_TOTAL_BYTES = Config::UB_MM_RES_TOTAL_BYTES;
    LocalTensor<uint8_t> ubMmResBuffers_;
    // L1
    static constexpr uint32_t L1_P_BUFCNT = Config::L1_P_BUFCNT;
    static constexpr uint32_t L1_P_BUF_BYTES = Config::L1_P_BUF_BYTES;
    static constexpr uint32_t L1_Q_BUFCNT = Config::L1_Q_BUFCNT;
    static constexpr uint32_t L1_Q_BUF_BYTES = Config::L1_Q_BUF_BYTES;
    static constexpr uint32_t L1_KV_BUFCNT = Config::L1_KV_BUFCNT;
    static constexpr uint32_t L1_KV_BUF_BYTES = Config::L1_KV_BUF_BYTES;
    static_assert(L1_P_BUFCNT * L1_P_BUF_BYTES + L1_Q_BUFCNT * L1_Q_BUF_BYTES + L1_KV_BUFCNT * L1_KV_BUF_BYTES <=
                      512U * 1024U,
                  "FA buffers exceed L1 capacity");
    // buffer位置+用途+Buffers, 例如l1PBuffers; 使用时命名: 用途+buffer位置+Tensor, 例如pL1Tensor
    __aicore__ inline uint32_t PSlotOffset(uint32_t id) const
    {
        return id * L1_P_BUF_BYTES;
    }
    __aicore__ inline uint32_t QSlotOffset(uint32_t id) const
    {
        return id * L1_Q_BUF_BYTES;
    }
    __aicore__ inline uint32_t KvSlotOffset(uint32_t id) const
    {
        return id * L1_KV_BUF_BYTES;
    }
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    // L0C
    static constexpr uint32_t L0C_BUFCNT = Config::L0C_BUFCNT;
    static constexpr uint32_t L0C_BUF_BYTES = Config::L0C_BUF_BYTES;
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;

    FaDnMmSlots<Config> mmSlots_;
    __aicore__ inline uint32_t AcquireMmSlot(bool second)
    {
        return mmSlots_.Acquire(second);
    }

    __aicore__ inline uint32_t MmSlotOffset(uint32_t id)
    {
        return Config::MmSlotOffset(id);
    }

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
                                     __gm__ uint8_t *blockTable)
    {
        if constexpr (dVBaseSize <= 128 && mBaseSize == 128 && IsSameType<INPUT_T, bfloat16_t>::value &&
                      IsSameType<OUTPUT_T, bfloat16_t>::value && !PAGE_ATTENTION && !FA_T::hasMask &&
                      LAYOUT_T == FA_LAYOUT::BNSD && LAYOUT_KV == FA_LAYOUT::BNSD &&
                      FA_T::attnOutLayout == FA_LAYOUT::BNSD) {
            pvDn_ = Config::UsePvDn(constInfo_, qSeqLensTool_, kvSeqLensTool_);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyBnStride, constInfo_.keyN2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueBnStride, constInfo_.valueN2Stride);
        // Whole-invocation guard: no mixing unit-managed and software-managed L0C reuse.

        // The full D72 path writes and drains the same padded L0C unit ranges.
    }

    __aicore__ inline void InitBuffers()
    {
        if constexpr (Config::DEEP_PRELOAD) {
            set_loop3_para(1);
            AscendC::PipeBarrier<PIPE_FIX>();
        }
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMmResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM_RES_TOTAL_BYTES);

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
        for (uint32_t slot = 0; slot < UB_MM_RES_BUFCNT; ++slot) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0 + slot);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM_0 + slot + AIV0_AIV1_OFFSET);
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

    __aicore__ inline void IterateBmm1(RunInfo &runInfo)
    {
        uint32_t mmResUbBufId = AcquireMmSlot(false);
        LocalTensor<MM_T> mm1ResUbTensor = ubMmResBuffers_[MmSlotOffset(mmResUbBufId)].template ReinterpretCast<MM_T>();
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm1Dn(mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void CopyMmResult(const DST_TENSOR_T &dstTensor, const LocalTensor<MM_T> &l0C,
                                        const FixpipeParamsC310<CO2Layout::ROW_MAJOR> &params)
    {
        if constexpr (Config::DEEP_PRELOAD) {
            // Match AscendC FixpipeL0cToUB, with fixed LOOP3_PARA initialized once.
            // L0C readiness and UB ownership remain protected by the existing flags.
            copy_matrix_cc_to_ub((__ubuf__ MM_T *)dstTensor.GetPhyAddr(), (__cc__ MM_T *)l0C.GetPhyAddr(), 0,
                                 params.nSize, params.mSize, params.dstStride, params.srcStride, params.dualDstCtl,
                                 params.subBlockId, 0, params.unitFlag, static_cast<uint64_t>(params.quantPre),
                                 static_cast<uint8_t>(params.reluEn), false, true,
                                 static_cast<uint64_t>(QuantMode_post::NoConv), 0, false, false, 0, false, false, false,
                                 false, false, false);
        } else {
            Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dstTensor, l0C, params);
        }
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
        fixpipeParams.unitFlag = 0U;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        CopyMmResult(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm1Dn(LocalTensor<MM_T> &mm1ResUbTensor, RunInfo &runInfo)
    {
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[QSlotOffset(qL1BufId_)].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            CopyQuerySlice(qL1Tensor, 0, constInfo_.dSize, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[KvSlotOffset(kvL1BufId_)].template ReinterpretCast<KV_T>();
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
                                        constInfo_.dSize, false, true, true, true, 0U, 0, ldK, 0);
            if constexpr (Config::SINGLE_PV_SLOT) {
                // K64 bounds a 256-row BF16 A tile to one 32KiB L0A slot.
                MatmulK<KV_T, Q_T, MM_T, 256, 128, 64, ABLayout::MK, ABLayout::KN>(kL1Tensor, qL1Tensor, mmL0APolicy_,
                                                                                   mmL0BPolicy_, l0CSubTensor, param);
            } else if constexpr (dBaseSize > 128) {
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

            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);

            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm2(RunInfo &runInfo)
    {
        uint32_t mmResUbBufId = AcquireMmSlot(true);
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        uint32_t mmSyncIdx = CROSSCORE_MM_0 + mmResUbBufId;
        LocalTensor<Q_T> pL1Tensor = l1PBuffers_[PSlotOffset(pL1BufId)].template ReinterpretCast<Q_T>();
        LocalTensor<MM_T> mm2ResUbTensor = ubMmResBuffers_[MmSlotOffset(mmResUbBufId)].template ReinterpretCast<MM_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm2l0Split(mm2ResUbTensor, pL1Tensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
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
        fixpipeParams.unitFlag = 0U;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        CopyMmResult(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2DnOutput(LocalTensor<MM_T> &dst, LocalTensor<Q_T> &p, RunInfo &runInfo)
    {
        auto v = l1KvBuffers_[KvSlotOffset(kvL1BufId_)].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyValueSlice(v, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);

        Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);

        auto accum = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
        MMParam param{};
        param.singleM = constInfo_.dSizeV;
        param.singleN = 128;
        param.singleK = s2BaseSize;
        param.isLeftTranspose = true;
        param.isRightTranspose = false;
        param.isOutKFisrt = true;
        param.unitFlag = 0U;
        MatmulK<KV_T, Q_T, MM_T, dVBaseSize, mBaseSize, 128, ABLayout::MK, ABLayout::KN>(v, p, mmL0APolicy_,
                                                                                         mmL0BPolicy_, accum, param);

        Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
        Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fix;
        fix.nSize = 128;
        fix.mSize = (constInfo_.dSizeV + 15U) / 16U * 16U;
        fix.srcStride = fix.mSize;
        fix.dstStride = 64;
        fix.dualDstCtl = 2;
        fix.unitFlag = 0U;
        fix.params.ndNum = 1;
        fix.params.srcNdStride = 0;
        fix.params.dstNdStride = 0;
        CopyMmResult(dst, accum, fix);

        Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

        l0cBufId_ = (l0cBufId_ + 1U) % L0C_BUFCNT;
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
    }

    __aicore__ inline void IterateBmm2l0Split(LocalTensor<MM_T> &mm2ResUbTensor, LocalTensor<Q_T> &pL1Tensor,
                                              RunInfo &runInfo)
    {
        if constexpr (dVBaseSize <= 128 && mBaseSize == 128 && IsSameType<INPUT_T, bfloat16_t>::value) {
            if (pvDn_ && !runInfo.isS2SplitCore) {
                IterateBmm2DnOutput(mm2ResUbTensor, pL1Tensor, runInfo);
                return;
            }
        }
        LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[KvSlotOffset(kvL1BufId_)].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyValueSlice(vL1Tensor, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            uint32_t nLoops = (constInfo_.dSizeV + 128 - 1) / 128;
            // Mmad N=真实尾块宽度精确计算; LoadData 搬运宽度按分形粒度对齐(D=72→80, 与Nd2Nz写入组数一致);
            // Fixpipe nSize=真实列 + dstStride=128(UB行距) → vec2 128宽读, 尾列脏但 D 轴逐元素隔离无害
            uint32_t lastTileN = constInfo_.dSizeV - (nLoops - 1) * 128;
            uint32_t ldN = ((lastTileN + 15) >> 4) << 4;
            for (uint32_t n = 0; n < nLoops; n++) {
                uint32_t tileN = (n == nLoops - 1) ? lastTileN : 128;
                uint32_t tileLdN = (n == nLoops - 1) ? ldN : 128;

                Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);

                LocalTensor<MM_T> l0CSubTensor =
                    l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
                MMParam param = {(uint32_t)mBaseSize,
                                 tileN,
                                 (uint32_t)runInfo.actSingleLoopS2Size,
                                 true,
                                 false,
                                 true,
                                 true,
                                 0,
                                 0,
                                 0,
                                 tileLdN};
                param.unitFlag = 0U;

                uint32_t s2Aligned = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U);
                uint64_t vL1Offset = n * 128U / 16U * s2Aligned * 16U;
                LocalTensor<KV_T> vL1TileTensor = vL1Tensor[vL1Offset];

                if constexpr (dVBaseSize > 128) {
                    MatmulFull<Q_T, KV_T, MM_T, 128, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                        pL1Tensor, vL1TileTensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor, param);
                } else {
                    if constexpr (s2BaseSize == 128) {
                        MatmulFull<Q_T, KV_T, MM_T, 128, dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
                            pL1Tensor, vL1TileTensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor, param);
                    } else {
                        // K128 keeps the full 128-row BF16 A tile within a 32KiB L0A slot.
                        MatmulK<Q_T, KV_T, MM_T, 128, dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
                            pL1Tensor, vL1TileTensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor, param);
                    }
                }

                Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
                Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

                uint32_t dstOffset = n * 128U;
                FixpipeMm2PartialN(mm2ResUbTensor[dstOffset], l0CSubTensor, tileN, runInfo);

                Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

                l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
            }
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
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
