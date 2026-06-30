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
 * \file flash_attn_block_cube_noquant_gqa.h
 * \brief
 */
#ifndef FLASH_ATTEN_BLOCK_CUBE_NOQUANT_GQA_H_
#define FLASH_ATTEN_BLOCK_CUBE_NOQUANT_GQA_H_

#include "../../../common/op_kernel/offset_calculator.h"
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

/* ============确定Query的L1类型============= */
template <typename INPUT_T, uint32_t dBaseSize>
struct QL1BuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float> || (!(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                                                         std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                                                         std::is_same_v<INPUT_T, hifloat8_t>) &&
                                                                       dBaseSize > 256),
                                    BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>;
};

/* ============确定Key的L1类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct KVL1BuffSel {
    constexpr static bool isFP8DType = std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                       std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                       std::is_same_v<INPUT_T, hifloat8_t>;
    using Type = std::conditional_t<
        (isFP8DType && s2BaseSize == 128 && dBaseSize == 576), BuffersPolicy4buff<BufferType::L1>,
        std::conditional_t<(!(isFP8DType) && s2BaseSize == 256 && dBaseSize > 128),
                           BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>>;
};

/* ============确定L0A的类型============= */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>, BuffersPolicySingleBuffer<BufferType::L0A>,
                                    BuffersPolicyDB<BufferType::L0A>>;
};
/* ============确定L0B的类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float> ||
                                        (s2BaseSize == 256 && dBaseSize > 128 &&
                                         !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                           std::is_same_v<INPUT_T, fp8_e5m2_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
                                    BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
};
/* ============确定L0C的类型============= */
template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct L0CBuffSel {
    using Type = std::conditional_t<(s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                     s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                    BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
};

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false>
class CubeBlockBase {
public:
    template <uint32_t dBaseSize>
    struct QL1BuffSel {
        using Type = std::conditional_t<(dBaseSize > 256),
                                        BuffersPolicySingleBuffer<BufferType::L1>,
                                        BuffersPolicyDB<BufferType::L1>>;
    };

    /* ============确定Key的L1类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct KVL1BuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L1>,
                                        BuffersPolicyDB<BufferType::L1>>;
    };

    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type = BuffersPolicyDB<BufferType::L0A>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L0B>,
                                        BuffersPolicyDB<BufferType::L0B>>;
    };

    /* ============确定L0C的类型============= */
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = std::conditional_t<(mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                        BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool SPLITD = splitD;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using L1KvType = typename KVL1BuffSel<s2BaseSize, dBaseSize>::Type;
    using L1QType = typename QL1BuffSel<dBaseSize>::Type;
    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSel<mBaseSize, s2BaseSize, dVBaseSize>::Type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline CubeBlockBase(ConstInfoX &constInfo){};
};

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockCube : public CubeBlockBase<INPUT_T, T, layout, s1TemplateType, s2TemplateType, dTemplateType,
                                                   dVTemplateType, KvLayoutType, useDn, bmm2Write2Ub, splitD> {
public:
    template <uint32_t dBaseSize>
    struct QL1BuffSel {
        using Type = std::conditional_t<(dBaseSize > 256), 
                                        BuffersPolicySingleBuffer<BufferType::L1>,
                                        BuffersPolicyDB<BufferType::L1>>;
    };

    /* ============确定Key的L1类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct KVL1BuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L1>,
                                        BuffersPolicyDB<BufferType::L1>>;
    };

    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type = BuffersPolicyDB<BufferType::L0A>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L0B>,
                                        BuffersPolicyDB<BufferType::L0B>>;
    };

    /* ============确定L0C的类型============= */
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = std::conditional_t<(mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                        BuffersPolicy4buff<BufferType::L0C>,
                                        BuffersPolicyDB<BufferType::L0C>>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using L1KvType = typename KVL1BuffSel<s2BaseSize, dBaseSize>::Type;
    using L1QType = typename QL1BuffSel<dBaseSize>::Type;
    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSel<mBaseSize, s2BaseSize, dVBaseSize>::Type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    using QSeqParserType =
        typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>,
                                  ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>>::type;

    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();
    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;
    TPipe *tPipe;

    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    GlobalTensor<int32_t> blockTableGm;

    QSeqParserType *qSeqParserPtr = nullptr;
    KvSeqParserType *kvSeqParserPtr = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    L1QType l1QBuffers;
    L1KvType l1KBuffers;
    L1KvType l1VBuffers;

    L0AType mmL0ABuffers;
    L0BType mmL0BBuffers;
    L0CType mmL0CBuffers;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    const ConstInfoX &constInfo;

    /*============================================================================== */
    __aicore__ inline FANoQuantGqaBlockCube(ConstInfoX &constInfo)
        : CubeBlockBase<INPUT_T, T, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, KvLayoutType,
                        useDn, bmm2Write2Ub, splitD>(constInfo),
          constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         QSeqParserType &qParser, KvSeqParserType &kvParser)
    {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        this->qSeqParserPtr = &qParser;
        this->kvSeqParserPtr = &kvParser;
        InitCubeInput(query, key, value, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        if constexpr (dBaseSize > 256) {
            /* D大于256的其他dtype场景，Bmm1左矩阵不开DB + 驻留 + 复用 + L1切K
            Bmm2左矩阵3 Buffer循环，Bmm1右矩阵和Bmm2右矩阵在L1上切D轴，并开启DoubleBuffer
            唯一不同的是D=256场景下的D轴只能切分到96，其余都可以切分到128，原因是D=256场景S1Base是128
            */
            constexpr uint32_t mm1LeftSize = mBaseSize * dBaseSize * sizeof(Q_T);
            constexpr uint32_t mm1RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
            constexpr uint32_t mm2RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
            l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
            l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
            l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
        } else {
            constexpr uint32_t mm1LeftSize = mBaseSize * dBaseSize * sizeof(Q_T);
            constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(KV_T);
            constexpr uint32_t mm2RightSize = (uint32_t)dVTemplateType * s2BaseSize * sizeof(KV_T);
            l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
            l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
            l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
        }

        // L0A B C 当前写死，能否通过基础api获取
        l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
        // L0A B C当前写死，要改成通过计算获取
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);

        if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
        } else {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        }
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
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit((*l1BufferManagerPtr));
        l1KBuffers.Uninit((*l1BufferManagerPtr));
        l1VBuffers.Uninit((*l1BufferManagerPtr));
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    // copy query with full s1g and split D
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

    // copy key with full s2 and split D
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

    // copy key with full s2 and split D
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

    // 全量拷贝
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

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &mm1ResBuf, RunInfoX &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateKey(runInfo.bIdx);
            }
        }

#ifdef SKIP_C1
        mm1ResBuf.WaitCrossCore();
        mm1ResBuf.SetCrossCore();
        return;
#endif

        if constexpr (dBaseSize > 256) {
            IterateBmm1NdL1SplitK(mm1ResBuf, runInfo);
            return;
        }

        if constexpr (useDn) {
            if constexpr (dBaseSize > 128) {
                IterateBmm1DnSplitK(mm1ResBuf, runInfo);
            } else {
                IterateBmm1Dn(mm1ResBuf, runInfo);
            }
            return;
        }
        IterateBmm1NdL0Split(mm1ResBuf, runInfo);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4;
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1;         // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 128, D > 128场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    __aicore__ inline void IterateBmm1NdL0Split(MM1_DBUF_T &mm1ResBuf, RunInfoX &runInfo)
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

        Buffer<BufferType::L1> mm1B = l1KBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();
        CopyKeyTile(mm1BTensor, runInfo);
        mm1B.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                    (uint32_t)(constInfo.dSize), false, true);
        if constexpr (dBaseSize <= 128) {
            MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(), param);
        } else {
            if constexpr (s2BaseSize == 256) {
                MatmulN<Q_T, KV_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
                    mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers,
                    mm1ResL0C.GetTensor<T>(), param);
            } else {
                MatmulK<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
                    mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers,
                    mm1ResL0C.GetTensor<T>(), param);
            }
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

        mm1ResBuf.WaitCrossCore();
        FixpipeMm1(mm1ResBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        mm1ResBuf.SetCrossCore();
    }

    /* 针对S1Base=128, S2Base = 128, D > 256场景，L1层面切K，且左矩阵单Buffer+驻留，右矩阵每次重新搬运。*/
    __aicore__ inline void IterateBmm1NdL1SplitK(MM1_DBUF_T &mm1ResBuf, RunInfoX &runInfo)
    {
        constexpr uint32_t baseKSize = l1BaseD;
        uint32_t kLoops = (constInfo.dSize + baseKSize - 1) / baseKSize;

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        uint32_t dstNzC0Stride = ((runInfo.actMSize + 15) >> 4 << 4);
        uint64_t l1BaseKOffset = baseKSize * dstNzC0Stride;

        Buffer<BufferType::L1> mm1A = l1QBuffers.Get();
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A.Wait<HardEvent::MTE1_MTE2>();
        }

        uint32_t realK = baseKSize;
        for (uint32_t kIdx = 0; kIdx < kLoops; kIdx++) {
            Buffer<BufferType::L1> mm1B; // 左矩阵复用, 但是每次只加载realK列
            if (kIdx == kLoops - 1) {
                realK = constInfo.dSize - kIdx * baseKSize;
            }
            if (unlikely(runInfo.isFirstS2Loop)) { // sOuter循环第一个基本快：搬运0
                LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
                CopyQuerySlice(mm1ATensor[kIdx * l1BaseKOffset], kIdx * baseKSize, realK, runInfo);
                mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
            } else {                              // 非s2的第一次循环直接复用Q
                mm1A = l1QBuffers.GetPre();
                mm1A.Set<HardEvent::MTE2_MTE1>();
            }
            // 加载当前轮的右矩阵到L1
            mm1B = l1KBuffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();
            CopyKeySlice(mm1BTensor, kIdx * baseKSize, realK, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();  // 通知
            mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
            mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

            MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size, realK, false,
                                        true, kIdx == 0, kIdx == 0);

            MatmulFull<Q_T, KV_T, T, 128, 128, baseKSize, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<Q_T>()[kIdx * l1BaseKOffset], mm1BTensor, mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(), param);

            mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
        mm1ResBuf.WaitCrossCore();

        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
        FixpipeMm1(mm1ResBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放
        mm1ResBuf.SetCrossCore();
    }

    /* 针对useDn=true, S1Base=128, S2Base = 128, 128 < D <= 256场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    __aicore__ inline void IterateBmm1DnSplitK(MM1_DBUF_T &mm1ResBuf, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1> mm1B;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1B = l1QBuffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();

        CopyKeyTile(mm1ATensor, runInfo);

        mm1A.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                    (uint32_t)(constInfo.dSize), false, true);

        MatmulK<KV_T, Q_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(mm1A.GetTensor<KV_T>(),
            mm1B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        mm1ResBuf.WaitCrossCore();
        FixpipeMm1Dn(mm1ResBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        mm1ResBuf.SetCrossCore();
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5; // 双目标模式, 按照N方向切分，N必须为32的倍数
        fixpipeParams.mSize = runInfo.actSingleLoopS2Size;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2; // 按照N方向切分
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams); // 将matmul结果从L0C搬运到UB
    }

    __aicore__ inline void IterateBmm1Dn(MM1_DBUF_T &mm1ResBuf, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1> mm1B;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1B = l1QBuffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();
        CopyKeyTile(mm1ATensor, runInfo);

        mm1A.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size,
                                    (uint32_t)runInfo.actMSize,
                                    (uint32_t)(constInfo.dSize), false, true);

        MatmulBase<KV_T, Q_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<KV_T>(), mm1B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers,
            mm1ResL0C.GetTensor<T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();

        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        mm1ResBuf.WaitCrossCore();
        FixpipeMm1Dn(mm1ResBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);
        mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
        mm1ResBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &mm2ResBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateValue(runInfo.bIdx);
            }
        }

#ifdef SKIP_C2
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();
        if constexpr (bmm2Write2Ub) {
            mm2ResBuf.WaitCrossCore();
        }
        mm2ResBuf.SetCrossCore();
        return;
#endif

        if constexpr ((uint32_t)dVTemplateType > 256 || (uint32_t)dTemplateType > 256) {
            IterateBmm2L1SplitN(mm2ResBuf, inputBuf, runInfo);
        } else {
            IterateBmm2l0Split(mm2ResBuf, inputBuf, runInfo);
        }
    }

    __aicore__ inline void IterateBmm2L1SplitN(mm2ResPos &mm2ResBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        MM2_ABUF_T mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();

        // dVTemplateType > 256, L1切N, 左矩阵不变，右矩阵每次循环搬运S2*128
        if constexpr (bmm2Write2Ub) {
            mm2ResBuf.WaitCrossCore();
        }
        constexpr uint32_t baseNSize = l1BaseD;
        uint32_t nLoops = ((uint32_t)constInfo.dSizeV + baseNSize - 1) / baseNSize; // 尾块处理
        uint32_t realN = baseNSize;
        for (uint32_t nIdx = 0; nIdx < nLoops; ++nIdx) {
            if (nIdx == nLoops - 1) {
                uint32_t tailSize = (uint32_t)constInfo.dSizeV % baseNSize;
                realN = tailSize ? tailSize : baseNSize;
            }
            Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
            mm2B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<KV_T> mm2BTensor = mm2B.GetTensor<KV_T>();
            CopyValueSlice(mm2BTensor, nIdx * baseNSize, realN, runInfo);
            mm2B.Set<HardEvent::MTE2_MTE1>();

            Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
            mm2ResL0C.Wait<HardEvent::FIX_M>();

            MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)realN, (uint32_t)(runInfo.actSingleLoopS2Size),
                                        useDn, false);
            if constexpr (!useDn) {
                param.realM = (uint32_t)runInfo.actMSize;
            }
            mm2B.Wait<HardEvent::MTE2_MTE1>();
            MatmulFull<Q_T, KV_T, T, 128, baseNSize, 128, ABLayout::MK, ABLayout::KN>(
                mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);

            mm2B.Set<HardEvent::MTE1_MTE2>();
            mm2ResL0C.Set<HardEvent::M_FIX>();
            mm2ResL0C.Wait<HardEvent::M_FIX>();

            FixpipeMm2PartialN(mm2ResBuf.template GetTensor<T>()[nIdx * baseNSize],
                               mm2ResL0C.GetTensor<T>(), realN, runInfo);
            mm2ResL0C.Set<HardEvent::FIX_M>();
        }
        mm2ResBuf.SetCrossCore();
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        if constexpr (bmm2Write2Ub) {
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = realN;
        }

        if constexpr (!useDn) {
            fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4;
        } else {
            fixpipeParams.mSize = mBaseSize;
            fixpipeParams.srcStride = ((mBaseSize + 15) >> 4) << 4;
        }

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

    __aicore__ inline void IterateBmm2l0Split(mm2ResPos &mm2ResBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
        Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
        mm2A.WaitCrossCore();
        mm2B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
        LocalTensor<KV_T> mm2BTensor = mm2B.GetTensor<KV_T>();
        CopyValueTile(mm2BTensor, runInfo);
        mm2B.Set<HardEvent::MTE2_MTE1>(); // 通知

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
        MMParam param = {
            (uint32_t)mBaseSize,                   // singleM 128
            (uint32_t)constInfo.dSizeV,            // singleN 128
            (uint32_t)runInfo.actSingleLoopS2Size, // singleK
            useDn,                                 // isLeftTranspose
            false                                  // isRightTranspose
        };
        if constexpr (!useDn) {
            param.realM = (uint32_t)runInfo.actMSize;
        }
        mm2B.Wait<HardEvent::MTE2_MTE1>(); // 等待

        if constexpr ((uint32_t)dVTemplateType <= 128) {
            if constexpr (s2BaseSize == 128) {
                MatmulFull<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
            } else {
                MatmulBase<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
            }
        } else {
            MatmulN<Q_T, KV_T, T, mBaseSize, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
        }

        mm2B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

        if constexpr (bmm2Write2Ub) {
            mm2ResBuf.WaitCrossCore();
        }

        FixpipeMm2PartialN(mm2ResBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), constInfo.dSizeV, runInfo);

        mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
        mm2ResBuf.SetCrossCore();
    }

};
} // namespace BaseApi

#endif // FLASH_ATTEN_BLOCK_CUBE_NOQUANT_GQA_H_
