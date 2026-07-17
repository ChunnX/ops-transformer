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
 * \file flash_attn_block_cube_mixedquant_gqa.h
 * \brief MixedQuantFlashAttn Cube Block定义
 */

#ifndef FLASH_ATTN_BLOCK_CUBE_MIXEDQUANT_GQA_H_
#define FLASH_ATTN_BLOCK_CUBE_MIXEDQUANT_GQA_H_

#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/copy_gm_to_l1.h"
#include "../../../common/op_kernel/memcopy/fa_l1_tensor.h"
#include "../../../common/op_kernel/memcopy/gm_coord.h"
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/buffer_manager.h"
#include "../../../common/op_kernel/buffers_policy.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/mixed_quant_flash_attn_common_def.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;

namespace BaseApi {

template <inferFaLayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormatMQ()
{
    if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BSND) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <uint8_t KvLayoutType>
__aicore__ inline constexpr GmFormat GetKVGmFormatMQ()
{
    if constexpr (KvLayoutType == KvLayoutType_PA_BBH) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == KvLayoutType_PA_BNBD) {
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

template <inferFaLayOutTypeEnum LAYOUT>
__aicore__ inline uint64_t SeqLenFromTensorListMQ(__gm__ uint8_t *keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4];
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BSND) {
        return keyTensorDesc.GetShape(1);
    } else {
        return keyTensorDesc.GetShape(2);
    }
}

template <typename QINPUT_T, typename ACC_T, typename KV_TYPE, inferFaLayOutTypeEnum inputLayoutType,
           inferS1TemplateType s1TemplateType, inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType,
           uint8_t KvLayoutType>
class MixedQuantGqaCubeBlockBase {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using Q_T = QINPUT_T;
    using KV_T = KV_TYPE;
    using MM_T = ACC_T;

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;

    static constexpr bool PAGE_ATTENTION = true;
    static constexpr bool USE_DN = false;
    static constexpr bool BMM2_TOUB = true;
    static constexpr bool HAS_PREFIX = false;

    static constexpr inferFaLayOutTypeEnum LAYOUT_Q_MQ = inputLayoutType;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormatMQ<inputLayoutType>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormatMQ<KvLayoutType>();

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);
    static constexpr bool Q_NEEDS_WZH = IS_TND_LAYOUT;

    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;
    static constexpr ActualSeqLensMode KV_MODE = ActualSeqLensMode::BY_BATCH;

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;
    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t>;

    __aicore__ inline MixedQuantGqaCubeBlockBase(ConstInfoX &constInfo) : constInfo(constInfo){};

    const ConstInfoX &constInfo;
};

template <typename QINPUT_T, typename ACC_T, typename KV_TYPE, inferFaLayOutTypeEnum inputLayoutType,
           inferS1TemplateType s1TemplateType, inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType,
           uint8_t KvLayoutType>
class FAMixedQuantGqaBlockCube : public MixedQuantGqaCubeBlockBase<QINPUT_T, ACC_T, KV_TYPE, inputLayoutType,
                                                                    s1TemplateType, s2TemplateType, dTemplateType,
                                                                    KvLayoutType> {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using Q_T = QINPUT_T;
    using KV_T = KV_TYPE;
    using MM_T = ACC_T;

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;
    static constexpr bool PAGE_ATTENTION = true;
    static constexpr bool BMM2_TOUB = true;
    static constexpr bool SPLITD = false;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};

    static constexpr inferFaLayOutTypeEnum LAYOUT_Q_MQ = inputLayoutType;
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormatMQ<inputLayoutType>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormatMQ<KvLayoutType>();

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);
    static constexpr bool Q_NEEDS_WZH = IS_TND_LAYOUT;

    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;
    static constexpr ActualSeqLensMode KV_MODE = ActualSeqLensMode::BY_BATCH;

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;
    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t>;

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>;
    using KV_ABUF_T = Buffer<BufferType::L1, SyncType::NO_SYNC>;

    using L1QType = BuffersPolicyDB<BufferType::L1>;
    using L1KvType = BuffersPolicy4buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>;
    using L0AType = BuffersPolicyDB<BufferType::L0A>;
    using L0BType = BuffersPolicyDB<BufferType::L0B>;
    using L0CType = BuffersPolicyDB<BufferType::L0C>;
    // using L0CType = typename std::conditional<
    //     (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
    //      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
    //     BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>::type;

    TPipe *tPipe;
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<KV_T> rawKeyGm;
    GlobalTensor<KV_T> rawValueGm;

    QSeqParserType *qSeqParserPtr = nullptr;
    KvSeqParserType *kvSeqParserPtr = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    L1QType l1QBuffers;
    L1KvType l1KvBuffers;
    // L1KvType l1VBuffers;

    L0AType mmL0ABuffers;
    L0BType mmL0BBuffers;
    L0CType mmL0CBuffers;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    const ConstInfoX &constInfo;

    __aicore__ inline FAMixedQuantGqaBlockCube(ConstInfoX &constInfo)
        : MixedQuantGqaCubeBlockBase<QINPUT_T, ACC_T, KV_TYPE, inputLayoutType, s1TemplateType, s2TemplateType,
                                      dTemplateType, KvLayoutType>(constInfo),
          constInfo(constInfo){};

    template <typename QParserRef, typename KvParserRef>
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         QParserRef &qParser, KvParserRef &kvParser)
    {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        this->qSeqParserPtr = &qParser;
        this->kvSeqParserPtr = &kvParser;
        InitCubeInput(query, key, value, blockTable);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);

        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize, queryGm,
                    query);

        keyPtr = key;
        valuePtr = value;
        if (constInfo.isKvContinuous) {
            InitKVBuffer(keyGm, key, constInfo.dSize);
            InitKVBuffer(valueGm, value, constInfo.dSizeV);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr);
        }
    }

    __aicore__ inline void InitKVBuffer(FaGmTensorKV &kvGmTensor, __gm__ uint8_t *gm, uint32_t headDim)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.blockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t mm1LeftSize = mBaseSize * dBaseSize * sizeof(Q_T);
        constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(KV_T);
        // constexpr uint32_t mm2RightSize = dVBaseSize * s2BaseSize * sizeof(KV_T);
        l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
        l1KvBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
        // l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);

        l0aBufferManager.Init(tPipe, 65536);
        l0bBufferManager.Init(tPipe, 65536);
        l0cBufferManager.Init(tPipe, 262144);

        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);

        if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
        } else {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        }
    }

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit((*l1BufferManagerPtr));
        l1KvBuffers.Uninit((*l1BufferManagerPtr));
        // l1VBuffers.Uninit((*l1BufferManagerPtr));
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    __aicore__ inline void InitAntiquantCrossCore()
    {
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[0]);
            CrossCoreSetFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[0] + 16);
            CrossCoreSetFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[1]);
            CrossCoreSetFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[1] + 16);
        }
    }

    __aicore__ inline void UninitAntiquantCrossCore()
    {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_VC_L1_EVENT[0]);
            CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_VC_L1_EVENT[0] + 16);
            CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_VC_L1_EVENT[1]);
            CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_VC_L1_EVENT[1] + 16);
        }
    }

    template <bool isAntiquantKey>
    __aicore__ inline void WaitAntiquantL1Ready(uint32_t idx)
    {
        CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_VC_L1_EVENT[idx]);
        CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_VC_L1_EVENT[idx] + 16);
    }

    template <bool isAntiquantKey>
    __aicore__ inline void SetAntiquantL1Consumed(uint32_t idx)
    {
        CrossCoreSetFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[idx]);
        CrossCoreSetFlag<MQFA_SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[idx] + 16);
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

    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;
        CopyKeySlice(dstTensor, 0, dSize, runInfo);
    }

    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        GmKvCoord gmCoord{.bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};

        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U)};
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, 0, constInfo.dSizeV, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        keyGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            keyGm.offsetCalculator.Init(constInfo.n2Size, constInfo.blockSize, constInfo.dSize, blockTableGm,
                                        constInfo.maxBlockNumPerBatch);
        }
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        valueGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            valueGm.offsetCalculator.Init(constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV, blockTableGm,
                                          constInfo.maxBlockNumPerBatch);
        }
    }

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &mm1ResBuf, KV_ABUF_T &inputBufB, RunInfoX &runInfo)
    {
        // if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        //     if (runInfo.isChangeBatch) {
        //         UpdateKey(runInfo.bIdx);
        //     }
        // }

#ifdef SKIP_C1
        mm1ResBuf.WaitCrossCore();
        mm1ResBuf.SetCrossCore();
        return;
#endif

        IterateBmm1NdL0Split(mm1ResBuf, inputBufB, runInfo);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                      RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4;
        fixpipeParams.dstStride = s2BaseSize;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<MM_T, MM_T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm1NdL0Split(MM1_DBUF_T &mm1ResBuf, KV_ABUF_T &inputBufB, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1> mm1A;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A = l1QBuffers.Get();
            mm1A.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
            CopyQueryTile(mm1ATensor, runInfo);
            mm1A.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1A = l1QBuffers.GetPre();
            mm1A.Set<HardEvent::MTE2_MTE1>();
        }
        KV_ABUF_T mm1B = inputBufB;
        //等待antiquantkey结果搬运到L1完成
        CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE3>(MQFA_VC_L1_EVENT[runInfo.KvL1loop & 3]); //应该mod 4
        CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE3>(MQFA_VC_L1_EVENT[runInfo.KvL1loop & 3] + 16);
        // mm1B.WaitCrossCore();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        // mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                    (uint32_t)(constInfo.dSize), false, true);

        // printf("runInfo.actMSize:%u\n",runInfo.actMSize);
        // printf("runInfo.actSingleLoopS2Size:%u\n",runInfo.actSingleLoopS2Size);
        // printf("constInfo.dSize:%u\n",constInfo.dSize);
        // if constexpr (dBaseSize <= 128) {
        //     MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
        //         mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers,
        //         mm1ResL0C.GetTensor<T>(), param);
        // } else {
        //     if constexpr (s2BaseSize == 256) {
        //         MatmulN<Q_T, KV_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
        //             mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers,
        //             mm1ResL0C.GetTensor<T>(), param);
        //     } else {
        //         MatmulK<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
        //             mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers,
        //             mm1ResL0C.GetTensor<T>(), param);
        //     }
        // }
        MatmulK<Q_T, Q_T, MM_T, mBaseSize, s2BaseSize, 64, ABLayout::MK, ABLayout::KN>(
                    mm1A.GetTensor<Q_T>(), mm1B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers,
                    mm1ResL0C.GetTensor<MM_T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

        //set 告诉antiquant阶段 L1已使用完
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[runInfo.KvL1loop & 3]); // 2 is double buffer
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + MQFA_CV_L1_EVENT[runInfo.KvL1loop & 3]); // 16 is Vec num, 2 is double buffer

        //等待Ub上结果使用完成
        mm1ResBuf.WaitCrossCore();

        FixpipeMm1(mm1ResBuf.template GetTensor<MM_T>(), mm1ResL0C.GetTensor<MM_T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        //告诉V1 C1结果搬运完成
        mm1ResBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &mm2ResBuf, MM2_ABUF_POLICY_T &inputBufA, KV_ABUF_T &inputBufB, RunInfoX &runInfo)
    {
#ifdef SKIP_C2
        // Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBufA.Get();
        // mm2A.WaitCrossCore();
        // mm2ResBuf.WaitCrossCore();
        // mm2ResBuf.SetCrossCore();
        // return;
#endif
        IterateBmm2l0Split(mm2ResBuf, inputBufA, inputBufB, runInfo);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<MM_T> &l0C,
                                              uint32_t realN, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;  //TODO: 验证是runInfo.actMSize还是s1BaseSize
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4;
        fixpipeParams.dstStride = ((uint32_t)dVBaseSize + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2l0Split(mm2ResPos &mm2ResBuf, MM2_ABUF_POLICY_T &inputBufA, KV_ABUF_T &inputBufB, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> mm2A = inputBufA.Get();
        KV_ABUF_T mm2B = inputBufB;
        // wait V1Res
        mm2A.WaitCrossCore();
        // wait VAntiquantRes
        // mm2B.WaitCrossCore();
        CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE3>(MQFA_VC_L1_EVENT[runInfo.KvL1loop & 3]); //应该mod 4
        CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE3>(MQFA_VC_L1_EVENT[runInfo.KvL1loop & 3] + 16);
        LocalTensor<Q_T> mm2BTensor = mm2B.GetTensor<Q_T>();

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();
        MMParam param = MakeMMParam(mBaseSize, (uint32_t)constInfo.dSizeV,
                                    (uint32_t)runInfo.actSingleLoopS2Size, false, false,
                                    true, true, 0, (uint32_t)runInfo.actMSize);

        MatmulN<Q_T, Q_T, MM_T, mBaseSize, dVBaseSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<MM_T>(), param);
        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();

        // set Vantiquant
        // mm2B.SetCrossCore();
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(MQFA_CV_L1_EVENT[runInfo.KvL1loop & 3]); // 2 is double buffer
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + MQFA_CV_L1_EVENT[runInfo.KvL1loop & 3]); // 16 is Vec num, 2 is double buffer


        //等待V2用完上一轮C2结果
        // CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(VC_MM2RES_EVENT[runInfo.loop % 2]);
        // CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + VC_MM2RES_EVENT[runInfo.loop % 2]); // 16 is Vec num
        mm2ResBuf.WaitCrossCore();

        FixpipeMm2PartialN(mm2ResBuf.template GetTensor<MM_T>(), mm2ResL0C.GetTensor<MM_T>(), constInfo.dSizeV,
                           runInfo);

        mm2ResL0C.Set<HardEvent::FIX_M>();
        //告诉本轮V2C2结果搬运完成
        // CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CV_MM2RES_EVENT[runInfo.loop % 2]);
        // CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + CV_MM2RES_EVENT[runInfo.loop % 2]); // 16 is Vec num
        mm2ResBuf.SetCrossCore();
    }
};

} // namespace BaseApi

#endif // FLASH_ATTN_BLOCK_CUBE_MIXEDQUANT_GQA_H_
