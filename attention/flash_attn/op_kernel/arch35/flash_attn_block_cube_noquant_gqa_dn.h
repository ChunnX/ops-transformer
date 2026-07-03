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
 * \file flash_attn_block_cube_noquant_gqa_dn.h
 * \brief FANoQuantGqaBlockCubeDn —— 仅 useDn=true 路径的 Cube Block 模板（sync-only）。
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_DN_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_DN_H_

#include "../utils/flash_attn_block_cube_noquant_gqa_comm.h"

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = true,
          bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockCubeDn {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;
    static constexpr bool HAS_PREFIX = false;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

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

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

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

    using QSeqParserType =
        typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>,
                                  ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>>::type;

    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();
    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;

    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    FaGmTensorQ queryRopeGm;
    FaGmTensorKV keyRopeGm;
    GlobalTensor<int32_t> blockTableGm;

    QSeqParserType *qSeqParserPtr = nullptr;
    KvSeqParserType *kvSeqParserPtr = nullptr;

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

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    const ConstInfoX &constInfo;

    __aicore__ inline FANoQuantGqaBlockCubeDn(ConstInfoX &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable,
                                         QSeqParserType &qParser, KvSeqParserType &kvParser)
    {
        this->qSeqParserPtr = &qParser;
        this->kvSeqParserPtr = &kvParser;
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

        keyPtr = key;
        valuePtr = value;
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
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr);
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
            kvGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr);
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

    // Cube 侧仅保留核间/核内同步，不执行 GM 搬运 / Matmul / Fixpipe。
    __aicore__ inline void Bmm1SyncOnly(uint32_t ccId0, uint32_t ccId1)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
        SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);

        WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
        SetFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);

        WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);

        SetFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
        qBufId = (qBufId + 1) % L1_Q_BUFCNT;
        SetFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
        kBufId = (kBufId + 1) % L1_KV_BUFCNT;

        SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
        WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);

        SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        cBufId = (cBufId + 1) % L0C_BUFCNT;

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1)
    {
        (void)outputTensor;
        Bmm1SyncOnly(ccId0, ccId1);
    }

    __aicore__ inline void Bmm2SyncOnly(uint32_t outCcId0, uint32_t outCcId1, uint32_t l1PCcId)
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);

        WaitFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0 + vBufId);
        SetFlag<HardEvent::MTE2_MTE1>(l1VBufFreeEventId0 + vBufId);

        WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1VBufFreeEventId0 + vBufId);

        SetFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0 + vBufId);
        vBufId = (vBufId + 1) % L1_KV_BUFCNT;
        SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
        WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

        if constexpr (bmm2Write2Ub) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
        }

        SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        cBufId = (cBufId + 1) % L0C_BUFCNT;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                        LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId)
    {
        (void)outputTensor;
        (void)l1PTensor;
        Bmm2SyncOnly(outCcId0, outCcId1, l1PCcId);
    }

}; // FANoQuantGqaBlockCubeDn

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = true,
          bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockCubeDummyDn {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_PREFIX = false;

    using Q_T = INPUT_T;
    using MM_T = T;
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline FANoQuantGqaBlockCubeDummyDn(ConstInfoX &constInfo){};
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_DN_H_
