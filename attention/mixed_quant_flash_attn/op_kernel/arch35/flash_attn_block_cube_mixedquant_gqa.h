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

#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"

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
        // TODO: 待后续填充实现
    }

    __aicore__ inline void AllocEventID()
    {
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit((*l1BufferManagerPtr));
        l1KBuffers.Uninit((*l1BufferManagerPtr));
        l1VBuffers.Uninit((*l1BufferManagerPtr));
    }

private:
    BuffersPolicyDB<BufferType::L1> l1QBuffers;
    BuffersPolicyDB<BufferType::L1> l1KBuffers;
    BuffersPolicyDB<BufferType::L1> l1VBuffers;
};

} // namespace BaseApi

#endif // FLASH_ATTN_BLOCK_CUBE_MIXEDQUANT_GQA_H_
