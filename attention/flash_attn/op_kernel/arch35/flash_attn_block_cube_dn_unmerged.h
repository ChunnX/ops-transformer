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
 * \brief FANoQuantGqaBlockCubeDnUnmerged
 */
#ifndef FLASH_ATTN_BLOCK_CUBE_DN_UNMERGED_H_
#define FLASH_ATTN_BLOCK_CUBE_DN_UNMERGED_H_

#include "../utils/flash_attn_type.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/arch_info.h"

using namespace fa_base_matmul;

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;

template <typename FA_T>
class FANoQuantGqaBlockCubeDnUnmerged {
public:
    using ConstInfo = ConstInfo_t;
    static constexpr uint32_t PRELOAD_N =
        FA_T::causalOpt ? 3U : (static_cast<uint32_t>(FA_T::dBaseSize) == 64U ? 3U : 2U);
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

    const ConstInfo &constInfo_;

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
    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, InnerMLayout::S1_ONLY_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    GlobalTensor<int32_t> blockTableGm_;

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t CROSSCORE_MM1_0 = 0U;
    static constexpr uint32_t CROSSCORE_MM2_0 = 2U;
    static constexpr uint32_t CROSSCORE_L1P_0 = 5U;
    static constexpr uint32_t CROSSCORE_L1P_1 = 6U;
    static constexpr uint32_t CROSSCORE_L1P_2 = 7U;

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

    // UB
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(MM_T);
    static constexpr uint32_t UB_MM2_RES_BUFCNT =
        FA_T::causalOpt ? (dBaseSize == 80U ? 2U : 1U) : (dVBaseSize <= 64U ? 2U : 1U);
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(MM_T);
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    uint32_t mm1ResUbBufId_ = 0U;
    uint32_t mm2ResUbBufId_ = 0U;
    // L1
    static constexpr uint32_t L1_P_BUFCNT = PRELOAD_N + 1U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_KV_BUFCNT = FA_T::causalOpt ? 4U : (dBaseSize > 128U ? 2U : 4U);
    static constexpr uint32_t L1_KV_BUF_BYTES =
        s2BaseSize * (dBaseSize > dVBaseSize ? dBaseSize : dVBaseSize) * sizeof(KV_T);
    static_assert(L1_P_BUFCNT * L1_P_BUF_BYTES + L1_Q_BUFCNT * L1_Q_BUF_BYTES + L1_KV_BUFCNT * L1_KV_BUF_BYTES <=
                      512U * 1024U,
                  "DN L1 buffers exceed device capacity");
    // buffer位置+用途+Buffers, 例如l1PBuffers; 使用时命名: 用途+buffer位置+Tensor, 例如pL1Tensor
    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    // L0C
    static constexpr uint32_t L0C_BUFCNT =
        FA_T::causalOpt ? (dBaseSize == 80U ? 4U : 2U) : (dBaseSize == 64U ? 4U : 2U);
    static constexpr uint32_t MM1_S2_SPLIT_SIZE =
        FA_T::causalOpt ? (dBaseSize == 80U ? 112U : s2BaseSize) : (dBaseSize == 64U ? 128U : s2BaseSize);
    static constexpr uint32_t MM1_K_SPLIT_SIZE = FA_T::causalOpt || dBaseSize == 64U ? 128U : 64U;
    static constexpr uint32_t L0C_BUF_BYTES =
        mBaseSize * (MM1_S2_SPLIT_SIZE > dVBaseSize ? MM1_S2_SPLIT_SIZE : dVBaseSize) * sizeof(MM_T);
    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;
    // L0A/B
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0A> l0aBufferManager_;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0B> l0bBufferManager_;
    using L0APolicyType =
        BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    using L0BPolicyType =
        BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    L0APolicyType mmL0APolicy_;
    L0BPolicyType mmL0BPolicy_;

    __aicore__ inline FANoQuantGqaBlockCubeDnUnmerged(ConstInfo_t &constInfo,
                                                      SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                      SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyBnStride, constInfo_.keyN2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueBnStride, constInfo_.valueN2Stride);
        if constexpr (FA_T::causalOpt && dBaseSize == 80U) {
            queryGm_.gmTensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        static_assert(L1_P_BUFCNT * L1_P_BUF_BYTES + L1_Q_BUFCNT * L1_Q_BUF_BYTES + L1_KV_BUFCNT * L1_KV_BUF_BYTES <=
                          512U * 1024U,
                      "DN L1 capacity exceeded");
        static_assert(L0C_BUFCNT * L0C_BUF_BYTES <= 256U * 1024U, "DN L0C capacity exceeded");
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);
        addrUb += UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;

        /*--------------------------------------------L1--------------------------------------------*/
        uint32_t addrL1 = 0;
        if constexpr (FA_T::causalOpt && dBaseSize == 80U) {
            // Q is repeatedly loaded into L0B; keep both Q slots in the low L1 half.
            l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_Q_BUFCNT * L1_Q_BUF_BYTES);
            addrL1 = L1_Q_BUFCNT * L1_Q_BUF_BYTES;
            l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);
            addrL1 += L1_P_BUFCNT * L1_P_BUF_BYTES;
        } else {
            l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);
            addrL1 = L1_P_BUFCNT * L1_P_BUF_BYTES;
            l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_Q_BUFCNT * L1_Q_BUF_BYTES);
            addrL1 += L1_Q_BUFCNT * L1_Q_BUF_BYTES;
        }
        l1KvBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * L1_KV_BUF_BYTES);

        // /*--------------------------------------------L0A--------------------------------------------*/
        l0aBufferManager_.Init(BUFFER_SIZE_BYTE_64K);

        // /*--------------------------------------------L0B--------------------------------------------*/
        l0bBufferManager_.Init(BUFFER_SIZE_BYTE_64K);

        /*--------------------------------------------L0C--------------------------------------------*/
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
        SetFixPipeConfig<uint64_t>(l0CBuffers_.template ReinterpretCast<uint64_t>(), true);
        SetMMRowMajor();
        if ASCEND_IS_AIC {
            // Both DN matmuls own the same single-matrix LOOP3 configuration for this invocation.
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> loopParams;
            loopParams.params.ndNum = 1;
            loopParams.params.srcNdStride = 0;
            loopParams.params.dstNdStride = 0;
            AscendC::SetLoop3Para<FIXPIPE_ROW_MAJOR_UB>(loopParams);
            if constexpr (FA_T::causalOpt && dBaseSize == 80U) {
                PipeBarrier<PIPE_FIX>();
            }
        }
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
        for (uint32_t slot = 0; slot < UB_MM1_RES_BUFCNT; ++slot) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM1_0 + slot);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM1_0 + slot + AIV0_AIV1_OFFSET);
        }
        for (uint32_t slot = 0; slot < UB_MM2_RES_BUFCNT; ++slot) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM2_0 + slot);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_MM2_0 + slot + AIV0_AIV1_OFFSET);
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
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        // 不合轴: 任务为单 query 头(n1)的纯 s1 块, 按 (b, n1) 直接寻址(n1 = n2*g + g)
        GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.n2Idx,
                              .gIdx = runInfo.gIdx,
                              .s1Idx = runInfo.s1Idx,
                              .dIdx = dOffset,
                              .s1DealSize = runInfo.actMSize,
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

    // Reserve two distinct slots before the current QK/PV iteration. MTE2
    // may load V while MTE1/M execute QK; consumer events protect slot reuse.
    __aicore__ inline uint32_t PrefetchKey(RunInfo &runInfo)
    {
        const uint32_t slot = kvL1BufId_;
        kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
        auto k = l1KvBuffers_[slot * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + slot);
        CopyKeySlice(k, 0, constInfo_.dSize, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + slot);
        return slot;
    }

    __aicore__ inline uint32_t PrefetchValue(RunInfo &runInfo)
    {
        const uint32_t slot = kvL1BufId_;
        kvL1BufId_ = (kvL1BufId_ + 1U) % L1_KV_BUFCNT;
        auto v = l1KvBuffers_[slot * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + slot);
        CopyValueSlice(v, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + slot);
        return slot;
    }

    __aicore__ inline void IterateBmm1(RunInfo &runInfo, uint32_t prefetchedSlot = 0U)
    {
        uint32_t mm1ResUbBufId = mm1ResUbBufId_;
        mm1ResUbBufId_ = (mm1ResUbBufId_ + 1) % UB_MM1_RES_BUFCNT;
        LocalTensor<MM_T> mm1ResUbTensor =
            ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<MM_T>();
        uint32_t mmSyncIdx = CROSSCORE_MM1_0 + mm1ResUbBufId;

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm1Dn(mm1ResUbTensor, runInfo, prefetchedSlot);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    // CANN 9.1/3510 internal lowering, restricted to the two non-quantized DN calls below.
    // Retain the SDK FIX barrier and parameter validation; only LOOP3 writes are hoisted.
    __aicore__ inline void FixpipeDnFixedLoop(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &srcTensor,
                                              const FixpipeParamsC310<CO2Layout::ROW_MAJOR> &params)
    {
        if ASCEND_IS_AIC {
            __ubuf__ MM_T *dst = (__ubuf__ MM_T *)dstTensor.GetPhyAddr();
            __cc__ MM_T *src = (__cc__ MM_T *)srcTensor.GetPhyAddr();
            AscendC::CheckFixpipeL0C2UBParam<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dst, src, params);
            if constexpr (!(FA_T::causalOpt && dBaseSize == 80U)) {
                PipeBarrier<PIPE_FIX>();
            }
            AscendC::FixpipeTiling tiling;
            AscendC::FixpipeL0cToUB<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB>(dst, src, params, tiling, params.nSize);
        }
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                        RunInfo &runInfo, uint32_t s2Size)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = (s2Size + 15U) >> 4U << 4U;
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.unitFlag = UNITFLAG_EN_OUTER_LAST;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        FixpipeDnFixedLoop(dstTensor, l0C, fixpipeParams);
    }

    template <uint32_t BaseM, uint32_t BaseN, uint32_t BaseK, typename L0AType, typename L0BType>
    __aicore__ inline void MatmulDnPaired(const LocalTensor<KV_T> &aL1, const LocalTensor<Q_T> &bL1, L0AType &aBuffers,
                                          L0BType &bBuffers, const LocalTensor<MM_T> &cL0, const MMParam &param)
    {
        if constexpr (!FA_T::causalOpt) {
            MatmulK<KV_T, Q_T, MM_T, BaseM, BaseN, BaseK, ABLayout::MK, ABLayout::KN>(aL1, bL1, aBuffers, bBuffers, cL0,
                                                                                      param);
        } else {
            const uint32_t loops = (param.singleK + BaseK - 1U) / BaseK;
            const uint32_t tail = param.singleK % BaseK;
            const uint64_t aStep = param.isLeftTranspose ? BaseK << 4U : ((param.singleM + 15U) >> 4U << 4U) * BaseK;
            const uint64_t bStep = param.isRightTranspose ? ((param.singleN + 15U) >> 4U << 4U) * BaseK : BaseK << 4U;
            for (uint32_t k = 0; k < loops; ++k) {
                const uint32_t tileK = k + 1U == loops && tail != 0U ? tail : BaseK;
                const uint32_t loadK = k + 1U == loops && param.loadK != 0U ? ((tileK + 15U) >> 4U << 4U) : tileK;
                auto a = aBuffers.Get();
                a.template Wait<HardEvent::M_MTE1>();
                auto aTensor = a.template GetTensor<KV_T>();
                LoadDataToL0A(aTensor, aL1, param, k * aStep, loadK, param.singleM);
                // Mutex mode needs a complete producer/consumer pair for each physical slot.
                a.template Set<HardEvent::MTE1_M>();
                auto b = bBuffers.Get();
                b.template Wait<HardEvent::M_MTE1>();
                auto bTensor = b.template GetTensor<Q_T>();
                LoadDataToL0B(bTensor, bL1, param, k * bStep, loadK, GetLoadN(param),
                              param.isRightTranspose ? 1U : loops);
                b.template Set<HardEvent::MTE1_M>();
                a.template Wait<HardEvent::MTE1_M>();
                b.template Wait<HardEvent::MTE1_M>();
                MmadParams mm;
                mm.m = param.realM != 0U ? param.realM : param.singleM;
                if (mm.m == 1U) {
                    mm.m = 16U;
                }
                mm.n = param.singleN;
                mm.k = tileK;
                mm.cmatrixInitVal = param.isOutKFisrt && k == 0U;
                mm.cmatrixSource = false;
                if (param.unitFlag != 0U) {
                    mm.unitFlag = param.unitFlag == UNITFLAG_EN_OUTER_LAST && k + 1U == loops ? UNITFLAG_EN_OUTER_LAST :
                                                                                                UNITFLAG_ENABLE;
                }
                Mmad(cL0, aTensor, bTensor, mm);
                if (param.unitFlag != 0U && k + 1U < loops && ((mm.m + 15U) >> 4U) * ((mm.n + 15U) >> 4U) < 10U) {
                    PipeBarrier<PIPE_M>();
                }
                a.template Set<HardEvent::M_MTE1>();
                b.template Set<HardEvent::M_MTE1>();
            }
        }
    }

    __aicore__ inline void MatmulBmm1ReuseQ(const LocalTensor<KV_T> &kL1Tensor, const LocalTensor<MM_T> &l0CTensor,
                                            const MMParam &param)
    {
        static_assert(MM1_S2_SPLIT_SIZE < s2BaseSize && dBaseSize <= MM1_K_SPLIT_SIZE,
                      "Q reuse requires multiple S2 subtiles and a single K tile");
        auto l0aBuffer = mmL0APolicy_.Get();
        l0aBuffer.template Wait<HardEvent::M_MTE1>();
        LocalTensor<KV_T> l0ATensor = l0aBuffer.template GetTensor<KV_T>();
        uint32_t loadK = param.loadK != 0 ? ((param.singleK + 15U) >> 4U << 4U) : param.singleK;
        LoadDataToL0A(l0ATensor, kL1Tensor, param, 0, loadK, param.singleM);
        if constexpr (FA_T::causalOpt) {
            l0aBuffer.template Set<HardEvent::MTE1_M>();
        }

        // The preceding S2 subtile loaded the same Q; no other L0B producer intervenes.
        auto l0bBuffer = mmL0BPolicy_.GetPre();
        l0bBuffer.template Wait<HardEvent::M_MTE1>();
        LocalTensor<Q_T> l0BTensor = l0bBuffer.template GetTensor<Q_T>();
        // This ready event still protects the new K load into L0A.
        l0bBuffer.template Set<HardEvent::MTE1_M>();
        if constexpr (FA_T::causalOpt) {
            l0aBuffer.template Wait<HardEvent::MTE1_M>();
        }
        l0bBuffer.template Wait<HardEvent::MTE1_M>();

        MmadParams mmParams;
        mmParams.m = param.realM != 0 ? param.realM : param.singleM;
        if (mmParams.m == 1) {
            mmParams.m = 16;
        }
        mmParams.n = param.singleN;
        mmParams.k = param.singleK;
        mmParams.cmatrixInitVal = param.isOutKFisrt;
        mmParams.cmatrixSource = false;
        mmParams.unitFlag = param.unitFlag;
        Mmad(l0CTensor, l0ATensor, l0BTensor, mmParams);
        l0aBuffer.template Set<HardEvent::M_MTE1>();
        l0bBuffer.template Set<HardEvent::M_MTE1>();
    }

    __aicore__ inline void IterateBmm1Dn(LocalTensor<MM_T> &mm1ResUbTensor, RunInfo &runInfo, uint32_t prefetchedSlot)
    {
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            CopyQuerySlice(qL1Tensor, 0, constInfo_.dSize, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        const uint32_t kvSlot = FA_T::causalOpt ? prefetchedSlot : kvL1BufId_;
        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvSlot * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        if constexpr (!FA_T::causalOpt) {
            Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            CopyKeySlice(kL1Tensor, 0, constInfo_.dSize, runInfo);
            Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        }
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvSlot);
        for (uint32_t s2Offset = 0; s2Offset < runInfo.actSingleLoopS2Size; s2Offset += MM1_S2_SPLIT_SIZE) {
            uint32_t s2Size =
                AttentionCommon::Min(MM1_S2_SPLIT_SIZE, static_cast<uint32_t>(runInfo.actSingleLoopS2Size) - s2Offset);
            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            // Mmad K=D 精确累加; LoadData 搬运宽度按分形粒度对齐(对齐D=原值, D=72→80与Nd2Nz写入组数一致)
            uint32_t ldK = ((constInfo_.dSize + 15) >> 4) << 4;
            MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (runInfo.actMSize + 31U) >> 5U << 5U,
                                        constInfo_.dSize, false, true, true, true, UNITFLAG_EN_OUTER_LAST,
                                        (s2Size + 15U) >> 4U << 4U, ldK, 0);
            if constexpr (MM1_S2_SPLIT_SIZE < s2BaseSize && dBaseSize <= MM1_K_SPLIT_SIZE) {
                if (s2Offset != 0) {
                    MatmulBmm1ReuseQ(kL1Tensor[s2Offset * 16U], l0CSubTensor, param);
                } else {
                    MatmulDnPaired<MM1_S2_SPLIT_SIZE, mBaseSize, MM1_K_SPLIT_SIZE>(kL1Tensor, qL1Tensor, mmL0APolicy_,
                                                                                   mmL0BPolicy_, l0CSubTensor, param);
                }
            } else {
                MatmulDnPaired<MM1_S2_SPLIT_SIZE, mBaseSize, MM1_K_SPLIT_SIZE>(
                    kL1Tensor[s2Offset * 16U], qL1Tensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor, param);
            }

            // singleM keeps the full L1 row stride; realM limits each L0/MMAD tile.
            uint32_t ubRowStride = ((runInfo.actMSize + 31U) >> 5U << 5U) / CV_RATIO;
            FixpipeMm1Dn(mm1ResUbTensor[s2Offset * ubRowStride], l0CSubTensor, runInfo, s2Size);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvSlot);
        if constexpr (!FA_T::causalOpt) {
            kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm2(RunInfo &runInfo, uint32_t prefetchedSlot = 0U)
    {
        uint32_t mm2ResUbBufId = mm2ResUbBufId_;
        mm2ResUbBufId_ = (mm2ResUbBufId_ + 1) % UB_MM2_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CROSSCORE_L1P_0 + pL1BufId;
        uint32_t mmSyncIdx = CROSSCORE_MM2_0 + mm2ResUbBufId;
        LocalTensor<Q_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<Q_T>();
        LocalTensor<MM_T> mm2ResUbTensor =
            ubMm2ResBuffers_[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<MM_T>();

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm2DnOutput(mm2ResUbTensor, pL1Tensor, runInfo, prefetchedSlot);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void IterateBmm2DnOutput(LocalTensor<MM_T> &mm2ResUbTensor, LocalTensor<Q_T> &pL1Tensor,
                                               RunInfo &runInfo, uint32_t prefetchedSlot)
    {
        static_assert(mBaseSize == 128 && s2BaseSize % 16 == 0 && dVBaseSize <= 128,
                      "PV DN requires M128, KV aligned to 16 and DV storage <=128");
        const uint32_t kvSlot = FA_T::causalOpt ? prefetchedSlot : kvL1BufId_;
        LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvSlot * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        if constexpr (!FA_T::causalOpt) {
            Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            CopyValueSlice(vL1Tensor, 0, constInfo_.dSizeV, runInfo);
            Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        }
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvSlot);
        {
            LocalTensor<MM_T> l0CSubTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<MM_T>();
            uint32_t ldK = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U);
            MMParam param = {constInfo_.dSizeV,
                             128,
                             (uint32_t)runInfo.actSingleLoopS2Size,
                             true,
                             false,
                             true,
                             true,
                             UNITFLAG_EN_OUTER_LAST,
                             0,
                             ldK,
                             0};
            MatmulDnPaired<dVBaseSize, mBaseSize, 128>(vL1Tensor, pL1Tensor, mmL0APolicy_, mmL0BPolicy_, l0CSubTensor,
                                                       param);

            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = 128;
            fixpipeParams.mSize = (constInfo_.dSizeV + 15U) / 16U * 16U;
            fixpipeParams.srcStride = fixpipeParams.mSize;
            fixpipeParams.dstStride = 64;
            fixpipeParams.dualDstCtl = 2;
            fixpipeParams.unitFlag = UNITFLAG_EN_OUTER_LAST;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
            FixpipeDnFixedLoop(mm2ResUbTensor, l0CSubTensor, fixpipeParams);

            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvSlot);
        if constexpr (!FA_T::causalOpt) {
            kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
        }
    }
}; // FANoQuantGqaBlockCubeDnUnmerged

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockCubeDummyDnUnmerged {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockCubeDummyDnUnmerged(ConstInfo_t &constInfo,
                                                           SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                           SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace FlashAttnKernel
#endif
