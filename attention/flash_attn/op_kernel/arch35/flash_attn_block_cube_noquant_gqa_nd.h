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
 * \file flash_attn_block_cube_noquant_gqa_nd.h
 * \brief FANoQuantGqaBlockCubeNd
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_ND_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_ND_H_

#include "../utils/flash_attn_block_cube_noquant_gqa_comm.h"

namespace BaseApi {

template <typename FA_T>
class FANoQuantGqaBlockCubeNd {
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
    static constexpr bool BMM2_TOUB = FA_T::bmm2Write2Ub;
    static constexpr bool SPLITD = FA_T::splitD;

    static constexpr uint32_t l1BaseD = 128;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, BMM2_TOUB};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<LAYOUT_KV, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = float;
    using SEQLEN_T = uint32_t;

    using L0AType = typename L0ABuffSel<INPUT_T>::Type;
    using L0BType = typename L0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    static constexpr bool L1Q_IS_SINGLE = (dBaseSize > 256);
    static constexpr bool L1KV_IS_SINGLE = (s2BaseSize == 256 && dBaseSize > 128);
    static constexpr bool L0C_IS_4BUFF = (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                          mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4);
    static constexpr uint32_t L0C_SLOT_COUNT = L0C_IS_4BUFF ? 4 : 2;

    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;

    static constexpr TEventID l1QBufFreeEventId0 = static_cast<TEventID>(0);
    static constexpr TEventID l1QBufFreeEventId1 = static_cast<TEventID>(1);
    static constexpr TEventID l1KBufFreeEventId0 = static_cast<TEventID>(2);
    static constexpr TEventID l1KBufFreeEventId1 = static_cast<TEventID>(3);
    static constexpr TEventID l1VBufFreeEventId0 = static_cast<TEventID>(4);
    static constexpr TEventID l1VBufFreeEventId1 = static_cast<TEventID>(5);
    static constexpr TEventID l0aBufFreeEventId0 = static_cast<TEventID>(0);
    static constexpr TEventID l0aBufFreeEventId1 = static_cast<TEventID>(1);
    static constexpr TEventID l0bBufFreeEventId0 = static_cast<TEventID>(2);
    static constexpr TEventID l0bBufFreeEventId1 = static_cast<TEventID>(3);
    static constexpr TEventID l0cBufFreeEventId0 = static_cast<TEventID>(0);
    static constexpr TEventID l0cBufFreeEventId1 = static_cast<TEventID>(1);
    static constexpr TEventID l0cBufFreeEventId2 = static_cast<TEventID>(2);
    static constexpr TEventID l0cBufFreeEventId3 = static_cast<TEventID>(3);

    static constexpr uint32_t FA_L1_CUBE_BASE = 3U * mBaseSize * s2BaseSize * sizeof(Q_T);

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, SEQLEN_T, IS_TND<LAYOUT_T>()>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, SEQLEN_T, IS_TND<LAYOUT_KV>()>;

    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    GlobalTensor<int32_t> blockTableGm;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    LocalTensor<uint8_t> l1QTensor;
    LocalTensor<uint8_t> l1KTensor;
    LocalTensor<uint8_t> l1VTensor;
    LocalTensor<uint8_t> l0CTensor;

    L0AType mmL0ABuffers;
    L0BType mmL0BBuffers;

    static constexpr uint32_t L1_Q_BUFCNT  = L1Q_IS_SINGLE ? 1U : 2U;
    static constexpr uint32_t L1_KV_BUFCNT = L1KV_IS_SINGLE ? 1U : 2U;
    static constexpr uint32_t L0C_BUFCNT   = L0C_SLOT_COUNT;

    static constexpr uint32_t L1_Q_SLOT_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_K_SLOT_BYTES =
        (dBaseSize > 256) ? (s2BaseSize * l1BaseD * sizeof(KV_T)) : (dBaseSize * s2BaseSize * sizeof(KV_T));
    static constexpr uint32_t L1_V_SLOT_BYTES =
        (dBaseSize > 256) ? (s2BaseSize * l1BaseD * sizeof(KV_T)) : (dVBaseSize * s2BaseSize * sizeof(KV_T));
    static constexpr uint32_t L0C_SLOT_BYTES =
        L0C_IS_4BUFF ? ((uint32_t)(L0C_SIZE / NUM_4) * (uint32_t)KB_TO_BYTES)
                     : ((uint32_t)(L0C_SIZE / NUM_2) * (uint32_t)KB_TO_BYTES);

    uint32_t qBufId = 0;
    uint32_t kBufId = 0;
    uint32_t vBufId = 0;
    uint32_t cBufId = 0;

    const ConstInfoX &constInfo;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool;

    __aicore__ inline FANoQuantGqaBlockCubeNd(ConstInfoX &constInfo,
        SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool, SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool) :
        constInfo(constInfo), qSeqLensTool(qSeqLensTool), kvSeqLensTool(kvSeqLensTool) {};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        InitCubeInput(query, key, value, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t addrL1 = FA_L1_CUBE_BASE;
        l1QTensor = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_Q_BUFCNT * L1_Q_SLOT_BYTES);

        addrL1 += L1_Q_BUFCNT * L1_Q_SLOT_BYTES;
        l1KTensor = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * L1_K_SLOT_BYTES);

        addrL1 += L1_KV_BUFCNT * L1_K_SLOT_BYTES;
        l1VTensor = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_KV_BUFCNT * L1_V_SLOT_BYTES);

        constexpr uint32_t FA_L0_BUF_SIZE = 32U * 1024U;
        mmL0ABuffers.InitStaticDirect(0U, FA_L0_BUF_SIZE, FA_L0_BUF_SIZE,
                                      l0aBufFreeEventId0, l0aBufFreeEventId0,
                                      l0aBufFreeEventId1, l0aBufFreeEventId1);

        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            mmL0BBuffers.InitStaticDirect(0U, FA_L0_BUF_SIZE, l0bBufFreeEventId0, l0bBufFreeEventId0);
        } else {
            mmL0BBuffers.InitStaticDirect(0U, FA_L0_BUF_SIZE, FA_L0_BUF_SIZE,
                                          l0bBufFreeEventId0, l0bBufFreeEventId0,
                                          l0bBufFreeEventId1, l0bBufFreeEventId1);
        }

        l0CTensor = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_SLOT_BYTES);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize, queryGm,
                    query);

        if (constInfo.isKvContinuous) {
            InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, constInfo.dSize,
                         keyGm, key);
            InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV,
                         valueGm, value);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool.seqUsedParser);
        } else {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(kvSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensTool.cuSeqLensParser);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0);
        if constexpr (!L1Q_IS_SINGLE) {
            SetFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId1);
        }
        SetFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0);
        if constexpr (!L1KV_IS_SINGLE) {
            SetFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId1);
        }
        SetFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0);
        if constexpr (!L1KV_IS_SINGLE) {
            SetFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId1);
        }
        SetFlag<HardEvent::M_MTE1>(l0aBufFreeEventId0);
        SetFlag<HardEvent::M_MTE1>(l0aBufFreeEventId1);
        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            SetFlag<HardEvent::M_MTE1>(l0bBufFreeEventId0);
        } else {
            SetFlag<HardEvent::M_MTE1>(l0bBufFreeEventId0);
            SetFlag<HardEvent::M_MTE1>(l0bBufFreeEventId1);
        }
        if constexpr (L0C_IS_4BUFF) {
            SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0);
            SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId1);
            SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId2);
            SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId3);
        } else {
            SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0);
            SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId1);
        }
    }

    __aicore__ inline void FreeEventID()
    {
        if constexpr (L1Q_IS_SINGLE) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0);
            WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId1);
        }
        if constexpr (L1KV_IS_SINGLE) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0);
            WaitFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0);
            WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId1);
            WaitFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0);
            WaitFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId1);
        }
        mmL0ABuffers.UninitStaticDirect();
        mmL0BBuffers.UninitStaticDirect();
        if constexpr (L0C_IS_4BUFF) {
            WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0);
            WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId1);
            WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId2);
            WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId3);
        } else {
            WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0);
            WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId1);
        }
    }

    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        uint32_t nopeDealSize = dRealSize;

        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        if (nopeDealSize > 0) {
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = dOffset,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = nopeDealSize};
            copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;
        CopyQuerySlice(dstTensor, 0, dSize, runInfo);
    }

    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;

        uint32_t nopeDealSize = dRealSize;
        uint32_t ropeDealSize = 0;

        if (nopeDealSize > 0) {
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = dOffset,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = nopeDealSize};
            copyKvGmToL1(l1Tensor, keyGm, gmCoord);
        }
    }

    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;
        CopyKeySlice(dstTensor, 0, dSize, runInfo);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, 0, constInfo.dSizeV, runInfo);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);

        if constexpr (dBaseSize > 256) {
            IterateBmm1NdL1SplitK(outputTensor, runInfo);
            return;
        }
        IterateBmm1NdL0Split(outputTensor, runInfo);

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<MM_T, MM_T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 128, D > 128场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1NdL0Split(OutTensorT &outputTensor, RunInfoX &runInfo)
    {
        LocalTensor<Q_T> mm1ATensor = l1QTensor[qBufId * L1_Q_SLOT_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
            {
                CopyQueryTile(mm1ATensor, runInfo);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
        } else {
            SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
        }

        LocalTensor<KV_T> mm1BTensor = l1KTensor[kBufId * L1_K_SLOT_BYTES].template ReinterpretCast<KV_T>();
        WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
        {
            CopyKeyTile(mm1BTensor, runInfo);
        }
        SetFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);

        WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        {
            LocalTensor<MM_T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<MM_T>();
            MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                        (uint32_t)(constInfo.dSize), false, true);
            if constexpr (dBaseSize > 128) {
                if constexpr (s2BaseSize == 256) {
                    MatmulN<Q_T, KV_T, MM_T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
                        mm1ATensor, mm1BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
                } else {
                    MatmulK<Q_T, KV_T, MM_T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
                        mm1ATensor, mm1BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
                }
            } else {
                MatmulBase<Q_T, KV_T, MM_T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
                    mm1ATensor, mm1BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
            }

            SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
            WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

            FixpipeMm1(outputTensor, l0CSubTensor, runInfo);
        }
        SetFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
        kBufId = (kBufId + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
            qBufId = (qBufId + 1) % L1_Q_BUFCNT;
        }

        SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        cBufId = (cBufId + 1) % L0C_BUFCNT;
    }

    /* 针对S1Base=128, S2Base = 128, D > 256场景，L1层面切K，且左矩阵单Buffer+驻留，右矩阵每次重新搬运。*/
    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1NdL1SplitK(OutTensorT &outputTensor, RunInfoX &runInfo)
    {
        constexpr uint32_t baseK = l1BaseD;
        uint32_t kLoops = (constInfo.dSize + baseK - 1) / baseK;

        WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        LocalTensor<MM_T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<MM_T>();

        uint32_t dstNzC0Stride = ((runInfo.actMSize + 15) >> 4 << 4);
        uint64_t l1BaseKOffset = baseK * dstNzC0Stride;

        LocalTensor<Q_T> mm1AFullTensor = l1QTensor[qBufId * L1_Q_SLOT_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
        }

        uint32_t realK = baseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            if (k == kLoops - 1) {
                realK = constInfo.dSize - k * baseK;
            }
            if (unlikely(runInfo.isFirstS2Loop)) {
                CopyQuerySlice(mm1AFullTensor[k * l1BaseKOffset], k * baseK, realK, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
            } else {
                SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
            }
            LocalTensor<KV_T> mm1BTensor = l1KTensor[kBufId * L1_K_SLOT_BYTES].template ReinterpretCast<KV_T>();
            WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
            {
                CopyKeySlice(mm1BTensor, k * baseK, realK, runInfo);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
            WaitFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
            WaitFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);

            MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size, realK, false,
                                        true, k == 0, k == 0);

            MatmulFull<Q_T, KV_T, MM_T, 128, 128, baseK, ABLayout::MK, ABLayout::KN>(
                mm1AFullTensor[k * l1BaseKOffset], mm1BTensor, mmL0ABuffers, mmL0BBuffers,
                l0CSubTensor, param);

            SetFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
            kBufId = (kBufId + 1) % L1_KV_BUFCNT;
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
            qBufId = (qBufId + 1) % L1_Q_BUFCNT;
        }
        SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

        WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
        FixpipeMm1(outputTensor, l0CSubTensor, runInfo);

        SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        cBufId = (cBufId + 1) % L0C_BUFCNT;
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                        LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoX &runInfo)
    {
#ifdef SKIP_C2
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);
        if constexpr (BMM2_TOUB) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
        return;
#endif

        if constexpr (dVBaseSize > 256 || dBaseSize > 256) {
            IterateBmm2L1SplitN(outputTensor, outCcId0, outCcId1, l1PTensor, l1PCcId, runInfo);
        } else {
            IterateBmm2l0Split(outputTensor, outCcId0, outCcId1, l1PTensor, l1PCcId, runInfo);
        }
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2L1SplitN(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                                LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoX &runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);

        if constexpr (BMM2_TOUB) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
        }
        constexpr uint32_t baseN = l1BaseD;
        uint32_t nLoops = ((uint32_t)constInfo.dSizeV + baseN - 1) / baseN;
        uint32_t realN = baseN;
        for (uint32_t n = 0; n < nLoops; ++n) {
            if (n == nLoops - 1) {
                uint32_t tailSize = (uint32_t)constInfo.dSizeV % baseN;
                realN = tailSize ? tailSize : baseN;
            }
            LocalTensor<KV_T> mm2BTensor = l1VTensor[vBufId * L1_V_SLOT_BYTES].template ReinterpretCast<KV_T>();
            WaitFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0 + vBufId);
            {
                CopyValueSlice(mm2BTensor, n * baseN, realN, runInfo);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1VBufFreeEventId0 + vBufId);

            WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
            {
                LocalTensor<MM_T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<MM_T>();
                MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)realN,
                                            (uint32_t)(runInfo.actSingleLoopS2Size), false, false);
                param.realM = (uint32_t)runInfo.actMSize;
                WaitFlag<HardEvent::MTE2_MTE1>(l1VBufFreeEventId0 + vBufId);
                MatmulFull<Q_T, KV_T, MM_T, 128, baseN, 128, ABLayout::MK, ABLayout::KN>(
                    l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);

                SetFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0 + vBufId);
                vBufId = (vBufId + 1) % L1_KV_BUFCNT;
                SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
                WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

                FixpipeMm2PartialN(outputTensor[n * baseN], l0CSubTensor, realN, runInfo);
            }
            SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
            cBufId = (cBufId + 1) % L0C_BUFCNT;
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<MM_T> &l0C, uint32_t realN,
                                              RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        if constexpr (BMM2_TOUB) {
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = realN;
        }

        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;

        if constexpr (BMM2_TOUB || SPLITD) {
            fixpipeParams.dstStride = (dVBaseSize + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2l0Split(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                               LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoX &runInfo)
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);

        LocalTensor<KV_T> mm2BTensor = l1VTensor[vBufId * L1_V_SLOT_BYTES].template ReinterpretCast<KV_T>();
        WaitFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0 + vBufId);
        {
            CopyValueTile(mm2BTensor, runInfo);
        }
        SetFlag<HardEvent::MTE2_MTE1>(l1VBufFreeEventId0 + vBufId);

        WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1VBufFreeEventId0 + vBufId);
        {
            LocalTensor<MM_T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<MM_T>();
            MMParam param = {
                (uint32_t)mBaseSize,                   // singleM 128
                (uint32_t)constInfo.dSizeV,            // singleN 128
                (uint32_t)runInfo.actSingleLoopS2Size, // singleK
                false,                                 // isLeftTranspose
                false                                  // isRightTranspose
            };
            param.realM = (uint32_t)runInfo.actMSize;

            if constexpr (dVBaseSize > 128) {
                MatmulN<Q_T, KV_T, MM_T, mBaseSize, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                    l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
            } else {
                if constexpr (s2BaseSize == 128) {
                    MatmulFull<Q_T, KV_T, MM_T, 128, dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
                        l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
                } else {
                    MatmulBase<Q_T, KV_T, MM_T, 128, dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
                        l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
                }
            }

            SetFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0 + vBufId);
            vBufId = (vBufId + 1) % L1_KV_BUFCNT;
            SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
            WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

            if constexpr (BMM2_TOUB) {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
            }

            FixpipeMm2PartialN(outputTensor, l0CSubTensor, constInfo.dSizeV, runInfo);
        }
        SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        cBufId = (cBufId + 1) % L0C_BUFCNT;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
    }

}; // FANoQuantGqaBlockCubeNd

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockCubeDummyNd {
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
    static constexpr bool BMM2_TOUB = FA_T::bmm2Write2Ub;
    static constexpr bool SPLITD = FA_T::splitD;

    using Q_T = INPUT_T;
    using MM_T = float;
    using SEQLEN_T = uint32_t;
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline FANoQuantGqaBlockCubeDummyNd(ConstInfoX &constInfo,
        SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool, SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool) {};
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_ND_H_
