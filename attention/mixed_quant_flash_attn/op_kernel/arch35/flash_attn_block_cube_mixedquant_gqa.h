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
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"         // PFA_CFG_ROW_MAJOR_UB
#include "../../../common/op_kernel/memory_copy_arch35.h" // SeqLenFromTensorList
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/mixed_quant_flash_attn_common_def.h"

namespace BaseApi {

using AscendC::TPipe;
using namespace fa_base_matmul;

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
    } else {
        return GmFormat::PA_BnBsND;
    }
}

template <typename INPUT_T, typename ACC_T, inferFaLayOutTypeEnum inputLayoutType, inferS1TemplateType s1TemplateType,
          inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, uint8_t KvLayoutType>
class MixedQuantGqaCubeBlockBase {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using Q_T = INPUT_T;
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
    static constexpr GmFormat KV_FORMAT_MQ = KV_FORMAT; // alias for kernel base access

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);
    static constexpr bool Q_NEEDS_WZH = IS_TND_LAYOUT;

    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;
    static constexpr ActualSeqLensMode KV_MODE = ActualSeqLensMode::BY_BATCH;

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;
    using KvSeqParserType = ActualSeqLensParser<KV_MODE, int32_t>;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<Q_T, KV_FORMAT, int32_t>;

    __aicore__ inline MixedQuantGqaCubeBlockBase(ConstInfoX &constInfo) : constInfo(constInfo){};

    const ConstInfoX &constInfo;
};

template <typename INPUT_T, typename ACC_T, inferFaLayOutTypeEnum inputLayoutType, inferS1TemplateType s1TemplateType,
          inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, uint8_t KvLayoutType>
class FAMixedQuantGqaBlockCube : public MixedQuantGqaCubeBlockBase<INPUT_T, ACC_T, inputLayoutType, s1TemplateType,
                                                                   s2TemplateType, dTemplateType, KvLayoutType> {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = ACC_T;

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;
    static constexpr bool PAGE_ATTENTION = true;
    static constexpr bool BMM2_TOUB = true;

    static constexpr inferFaLayOutTypeEnum LAYOUT_Q_MQ = inputLayoutType;
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormatMQ<inputLayoutType>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormatMQ<KvLayoutType>();

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);
    static constexpr bool Q_NEEDS_WZH = IS_TND_LAYOUT;

    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;
    static constexpr ActualSeqLensMode KV_MODE = ActualSeqLensMode::BY_BATCH;

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;
    using KvSeqParserType = ActualSeqLensParser<KV_MODE, int32_t>;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<Q_T, KV_FORMAT, int32_t>;

    TPipe *tPipe;
    BufferManager<BufferType::L1> *l1BufferManagerPtr;

    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    GlobalTensor<int32_t> blockTableGm;

    QSeqParserType *qSeqParserPtr = nullptr;
    KvSeqParserType *kvSeqParserPtr = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    const ConstInfoX &constInfo;

    __aicore__ inline FAMixedQuantGqaBlockCube(ConstInfoX &constInfo)
        : MixedQuantGqaCubeBlockBase<INPUT_T, ACC_T, inputLayoutType, s1TemplateType, s2TemplateType, dTemplateType,
                                     KvLayoutType>(constInfo),
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
        l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);

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
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        GmCoord gmCoord{.bIdx = runInfo.bIdx, .n2Idx = runInfo.n2Idx, .gS1Idx = runInfo.gS1Idx, .dIdx = 0,
                        .gS1DealSize = runInfo.actMSize, .dDealSize = dSize};
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
    }

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &mm1ResBuf, MM2_ABUF_POLICY_T &l1KAntiResBuf, RunInfoX &runInfo,
                                       int32_t vcL1EventId, int32_t cvL1EventId)
    {
#ifdef SKIP_C1
        mm1ResBuf.WaitCrossCore();
        mm1ResBuf.SetCrossCore();
        return;
#endif
        IterateBmm1NdL0Split(mm1ResBuf, l1KAntiResBuf, runInfo, vcL1EventId, cvL1EventId);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C, RunInfoX &runInfo)
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

    __aicore__ inline void IterateBmm1NdL0Split(MM1_DBUF_T &mm1ResBuf, MM2_ABUF_POLICY_T &l1KAntiResBuf, RunInfoX &runInfo,
                                                   int32_t vcL1EventId, int32_t cvL1EventId)
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

        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(vcL1EventId);
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm1B = l1KAntiResBuf.Get();
        mm1B.WaitCrossCore();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                    (uint32_t)(constInfo.dSize), false, true);
        MatmulBase<Q_T, Q_T, MM_T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<Q_T>(), mm1B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers,
            mm1ResL0C.GetTensor<MM_T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(cvL1EventId);
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        mm1ResBuf.WaitCrossCore();
        FixpipeMm1(mm1ResBuf.template GetTensor<MM_T>(), mm1ResL0C.GetTensor<MM_T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        mm1ResBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(MM1_DBUF_T &mm2ResBuf, MM2_ABUF_POLICY_T &inputBuf,
                                       MM2_ABUF_POLICY_T &l1VAntiResBuf, RunInfoX &runInfo,
                                       int32_t vcL1EventId, int32_t cvL1EventId)
    {
#ifdef SKIP_C2
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();
        mm2ResBuf.WaitCrossCore();
        mm2ResBuf.SetCrossCore();
        return;
#endif
        IterateBmm2l0Split(mm2ResBuf, inputBuf, l1VAntiResBuf, runInfo, vcL1EventId, cvL1EventId);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<MM_T> &l0C,
                                              uint32_t realN, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4;
        fixpipeParams.dstStride = ((uint32_t)dVBaseSize + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2l0Split(MM1_DBUF_T &mm2ResBuf, MM2_ABUF_POLICY_T &inputBuf,
                                               MM2_ABUF_POLICY_T &l1VAntiResBuf, RunInfoX &runInfo,
                                               int32_t vcL1EventId, int32_t cvL1EventId)
    {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();

        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(vcL1EventId);
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2B = l1VAntiResBuf.Get();
        mm2B.WaitCrossCore();

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();
        MMParam param = MakeMMParam(mBaseSize, (uint32_t)constInfo.dSizeV,
                                    (uint32_t)runInfo.actSingleLoopS2Size, false, false,
                                    true, true, 0, (uint32_t)runInfo.actMSize);

        MatmulBase<Q_T, Q_T, MM_T, 128, (uint32_t)dVBaseSize, 128, ABLayout::MK, ABLayout::KN>(
            mm2A.GetTensor<Q_T>(), mm2B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers,
            mm2ResL0C.GetTensor<MM_T>(), param);

        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(cvL1EventId);
        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();

        mm2ResBuf.WaitCrossCore();
        FixpipeMm2PartialN(mm2ResBuf.template GetTensor<MM_T>(), mm2ResL0C.GetTensor<MM_T>(),
                           constInfo.dSizeV, runInfo);

        mm2ResL0C.Set<HardEvent::FIX_M>();
        mm2ResBuf.SetCrossCore();
    }

private:
    BuffersPolicyDB<BufferType::L1> l1QBuffers;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
};

} // namespace BaseApi

#endif // FLASH_ATTN_BLOCK_CUBE_MIXEDQUANT_GQA_H_
