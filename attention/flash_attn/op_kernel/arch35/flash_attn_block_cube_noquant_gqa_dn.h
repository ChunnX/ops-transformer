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
 * \brief FANoQuantGqaBlockCubeDn —— 仅 useDn=true 路径的 Cube Block 模板（参考 quant_flash_attn 风格）。
 *        IterateBmm1 仅含 IterateBmm1Dn / IterateBmm1DnSplitK；IterateBmm2 系列仅保留 useDn=true 分支。
 *        Nd 路径（IterateBmm1NdL0Split / IterateBmm1NdL1SplitK / FixpipeMm1）由独立的 _nd.h 提供。
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_DN_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_DN_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "fa_static_buffer.h"  // 自包含的Buffer/BufferManager/BuffersPolicy，替代common下的buff文件
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"         // PFA_CFG_ROW_MAJOR_UB
#include "../../../common/op_kernel/memory_copy_arch35.h" // SeqLenFromTensorList
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;

namespace BaseApi {

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

/* ============确定L0A的类型（仅供matmul使用）============= */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>, BuffersPolicySingleBuffer<BufferType::L0A>,
                                    BuffersPolicyDB<BufferType::L0A>>;
};
/* ============确定L0B的类型（仅供matmul使用）============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float> ||
                                        (s2BaseSize == 256 && dBaseSize > 128 &&
                                         !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                           std::is_same_v<INPUT_T, fp8_e5m2_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
                                    BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
};


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

    // ====== L1/L0C 缓冲区布局常量 ======
    static constexpr bool L1Q_IS_SINGLE = (dBaseSize > 256);
    static constexpr bool L1KV_IS_SINGLE = (s2BaseSize == 256 && dBaseSize > 128);
    static constexpr bool L0C_IS_4BUFF = (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                          mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4);
    static constexpr uint32_t L0C_SLOT_COUNT = L0C_IS_4BUFF ? 4 : 2;

    // ====== 核间同步常量 ======
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    // ====== 静态事件ID常量 ======
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

    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    FaGmTensorQ queryRopeGm;
    FaGmTensorKV keyRopeGm;
    GlobalTensor<int32_t> blockTableGm;

    QSeqParserType *qSeqParserPtr = nullptr;
    KvSeqParserType *kvSeqParserPtr = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    /* =====================LocalBuffer变量====================*/
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

    /*============================================================================== */
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

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr, bIdx);
        keyGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm.offsetCalculator.Init(0, constInfo.n2Size, s2Size, constInfo.dSize);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr, bIdx);
        valueGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm.offsetCalculator.Init(0, constInfo.n2Size, s2Size, constInfo.dSize);
    }

    // ─── IterateBmm1：Dn 专用 dispatch（dBaseSize>256 的 Nd L1SplitK 路径在此分支不会触发，已移除）──
    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateKey(runInfo.bIdx);
            }
        }

#ifdef SKIP_C1
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
        return;
#endif

        if constexpr (dBaseSize > 128) {
            IterateBmm1DnSplitK(outputTensor, ccId0, ccId1, runInfo);
        } else {
            IterateBmm1Dn(outputTensor, ccId0, ccId1, runInfo);
        }
    }

    /* 针对useDn=true, S1Base=128, S2Base = 128, 128 < D <= 256场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1DnSplitK(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
    {
        // ===== L1 Q（DN 中作为右矩阵）：仅 isFirstS2Loop 搬运，qBufId 在 isLastS2Loop 后推进 =====
        LocalTensor<Q_T> mm1QTensor = l1QTensor[qBufId * L1_Q_SLOT_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
            {
                CopyQueryTile(mm1QTensor, runInfo);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
        } else {
            SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
        }

        // ===== L1 K（DN 中作为左矩阵）：每轮 S2 内层载入 =====
        LocalTensor<KV_T> mm1ATensor = l1KTensor[kBufId * L1_K_SLOT_BYTES].template ReinterpretCast<KV_T>();
        WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
        {
            CopyKeyTile(mm1ATensor, runInfo);
        }
        SetFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);

        WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        {
            LocalTensor<T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<T>();
            MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                        (uint32_t)(constInfo.dSize), false, true);
            MatmulK<KV_T, Q_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
                mm1ATensor, mm1QTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);

            if (unlikely(runInfo.isLastS2Loop)) {
                SetFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
                qBufId = (qBufId + 1) % L1_Q_BUFCNT;
            }
            SetFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
            kBufId = (kBufId + 1) % L1_KV_BUFCNT;

            SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
            WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);
            FixpipeMm1Dn(outputTensor, l0CSubTensor, runInfo);
        }
        SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        cBufId = (cBufId + 1) % L0C_BUFCNT;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = runInfo.actSingleLoopS2Size;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1Dn(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
    {
        LocalTensor<Q_T> mm1QTensor = l1QTensor[qBufId * L1_Q_SLOT_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
            {
                CopyQueryTile(mm1QTensor, runInfo);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
        } else {
            SetFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);
        }

        LocalTensor<KV_T> mm1ATensor = l1KTensor[kBufId * L1_K_SLOT_BYTES].template ReinterpretCast<KV_T>();
        WaitFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
        {
            CopyKeyTile(mm1ATensor, runInfo);
        }
        SetFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1KBufFreeEventId0 + kBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(l1QBufFreeEventId0 + qBufId);

        WaitFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        {
            LocalTensor<T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<T>();
            MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                        (uint32_t)(constInfo.dSize), false, true);
            MatmulBase<KV_T, Q_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
                mm1ATensor, mm1QTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);

            if (unlikely(runInfo.isLastS2Loop)) {
                SetFlag<HardEvent::MTE1_MTE2>(l1QBufFreeEventId0 + qBufId);
                qBufId = (qBufId + 1) % L1_Q_BUFCNT;
            }
            SetFlag<HardEvent::MTE1_MTE2>(l1KBufFreeEventId0 + kBufId);
            kBufId = (kBufId + 1) % L1_KV_BUFCNT;

            SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
            WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);
            FixpipeMm1Dn(outputTensor, l0CSubTensor, runInfo);
        }
        SetFlag<HardEvent::FIX_M>(l0cBufFreeEventId0 + cBufId);
        cBufId = (cBufId + 1) % L0C_BUFCNT;
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                        LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoX &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateValue(runInfo.bIdx);
            }
        }

#ifdef SKIP_C2
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);
        if constexpr (bmm2Write2Ub) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
        return;
#endif

        if constexpr ((uint32_t)dVTemplateType > 256 || (uint32_t)dTemplateType > 256) {
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

        if constexpr (bmm2Write2Ub) {
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
                LocalTensor<T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<T>();
                // useDn=true 分支：useDn 直接传 true，且不设 param.realM
                MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)realN,
                                            (uint32_t)(runInfo.actSingleLoopS2Size), true, false);
                WaitFlag<HardEvent::MTE2_MTE1>(l1VBufFreeEventId0 + vBufId);
                MatmulFull<Q_T, KV_T, T, 128, baseN, 128, ABLayout::MK, ABLayout::KN>(
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
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        if constexpr (bmm2Write2Ub) {
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = realN;
        }

        // useDn=true 分支
        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;

        if constexpr (bmm2Write2Ub || splitD) {
            fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
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
            LocalTensor<T> l0CSubTensor = l0CTensor[cBufId * L0C_SLOT_BYTES].template ReinterpretCast<T>();
            // useDn=true 分支：isLeftTranspose=true，且不设 param.realM
            MMParam param = {
                (uint32_t)mBaseSize,                   // singleM 128
                (uint32_t)constInfo.dSizeV,            // singleN 128
                (uint32_t)runInfo.actSingleLoopS2Size, // singleK
                true,                                  // isLeftTranspose（useDn=true）
                false                                  // isRightTranspose
            };

            if constexpr ((uint32_t)dVTemplateType > 128) {
                MatmulN<Q_T, KV_T, T, mBaseSize, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                    l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
            } else {
                if constexpr (s2BaseSize == 128) {
                    MatmulFull<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                        l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
                } else {
                    MatmulBase<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                        l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CSubTensor, param);
                }
            }

            SetFlag<HardEvent::MTE1_MTE2>(l1VBufFreeEventId0 + vBufId);
            vBufId = (vBufId + 1) % L1_KV_BUFCNT;
            SetFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);
            WaitFlag<HardEvent::M_FIX>(l0cBufFreeEventId0 + cBufId);

            if constexpr (bmm2Write2Ub) {
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

}; // FANoQuantGqaBlockCubeDn

// AIC/AIV 分编译占位（Mix kernel 在 AIV 侧重编译时使用）
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
