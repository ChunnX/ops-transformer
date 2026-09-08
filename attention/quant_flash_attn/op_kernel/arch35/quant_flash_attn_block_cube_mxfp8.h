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
 * \file quant_flash_attn_block_cube_mxfp8.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_H_
#if __has_include("../../../common/op_kernel/offset_calculator.h")
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#else
#include "../../common/op_kernel/offset_calculator.h"
#include "../../common/op_kernel/matmul.h"
#include "../../common/op_kernel/FixpipeOut.h"
#include "../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/arch35/util_regbase.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "quant_flash_attn_type.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (kvLayoutType == 0) {
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
    } else if constexpr (kvLayoutType == 1) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) {
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

template <typename FA_T>
class QuantFlashAttnBlockCubeMxfp8 {
public:
    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type =
            BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type =
            BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    };

    // fa式核内同步（ADR 0003）：Mutex ID全局单一空间（0-27）须全局互异，AIC侧独立。
    // Q_L1 0-1（驻留双缓冲）、KV_L1 2-5（4buff）、L0A 6/7、L0B 8/9、L0C 10-13（条件4buff）或10/11（DB）。
    // L0A/L0B由matmul内部Wait/Set调用点经policy映射为Lock/Unlock（序列每迭代严格配对，fa同款）。
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 0U;
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 2U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 6U;
    static constexpr uint32_t L0A_BUFFER_ID1 = 7U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 8U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 9U;
    static constexpr uint32_t L0C_BUFFER_ID0 = 10U;

    /* ============FA_T派生的基本块信息（对齐fa）============= */
    using INPUT_T = typename FA_T::inputType;
    using Q_T = typename FA_T::inputType;
    using KV_T = typename FA_T::inputType;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = float;
    using T = float;
    using QSeqLensToolType = QFASeqLensTool<FA_T::qLayout, false, false, int32_t, GetQActSeqMode<FA_T::qLayout>()>;
    using KvSeqLensToolType = QFASeqLensTool<FA_T::kvLayout, FA_T::pageAttention, false, int32_t,
                                             GetKvActSeqMode<FA_T::kvLayout, FA_T::pageAttention>()>;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr DTemplateType dTemplateType = FA_T::dBaseSize;
    static constexpr DTemplateType dVTemplateType = FA_T::dVBaseSize;
    static constexpr bool isDAligned = FA_T::isDAligned;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t s2SplitSize =
        (dBaseSize == static_cast<uint32_t>(DTemplateType::Aligned256)) ? 128U : 256U;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    static constexpr LayOutTypeEnum LAYOUT = FA_T::qLayout;
    static constexpr LayOutTypeEnum layout = FA_T::qLayout; // 内部小写引用兼容（GetKVGmFormat等）
    static constexpr uint8_t KvLayoutType = FA_T::kvLayoutType;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool USE_DN = FA_T::useDn;
    static constexpr InnerMLayout Q_M_LAYOUT = USE_DN ? InnerMLayout::S1_ONLY_LAYOUT : InnerMLayout::GS1_MERGE_LAYOUT;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();
    static constexpr GmFormat V_SCALE_FORMAT = GetValueScaleGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    // fa式L1/L0C布局：裸数组+显式bufId_轮转。L1的P区由kernel侧l1BufferManager_分配（起始0，
    // 保证AIC/AIV跨核同址），Q/KV从P区末尾起算（kernel侧mm2LeftSize公式在cube侧重算，同vec侧VEC_UB_BASE模式）。
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES =
        mBaseSize * s2BaseSize * sizeof(INPUT_T) + mBaseSize * s2BaseSize / MXFP_GROUP_SIZE;
    static constexpr uint32_t L1_P_REGION_BYTES = L1_P_BUFCNT * L1_P_BUF_BYTES;
    // fa式跨核共享区（cube/vec双侧各自持有同布局裸数组，跨核flag=slotID，AIC侧±16双AIV镜像）：
    // UB: [bmm2Res 1槽][bmm1Res 2槽] 从0起（vec的VEC_UB_BASE紧随其后）；L1: P区3槽从0起（Q/KV紧随其后）
    static constexpr uint32_t MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T) / 2;
    static constexpr uint32_t MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);
    static constexpr uint32_t MM2_RES_BUF_BYTES_TOTAL = MM2_RES_BUF_BYTES + MM1_RES_BUFCNT * MM1_RES_BUF_BYTES;
    // 跨核flag ID（mode 4 独立编号空间；AIC侧每flag须Set/Wait id与id+16两个）
    static constexpr uint32_t CC_MM_BMM1_0 = 0U;
    static constexpr uint32_t CC_MM_BMM1_1 = 1U;
    static constexpr uint32_t CC_MM_BMM2_0 = 2U;
    static constexpr uint32_t CC_L1P_0 = 3U;
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_Q_BUF_BYTES =
        mBaseSize * dBaseSize * sizeof(INPUT_T) + mBaseSize * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T);
    static constexpr uint32_t L1_KV_BUFCNT = 4U;
    static constexpr uint32_t L1_KV_BUF_BYTES =
        (dBaseSize == static_cast<uint32_t>(DTemplateType::Aligned256)) ?
            (dBaseSize * s2BaseSize * sizeof(INPUT_T) + s2BaseSize * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T)) :
            (dBaseSize * (s2BaseSize >> 1) * sizeof(INPUT_T) +
             (s2BaseSize >> 1) * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T));
    static constexpr bool L0C_IS_SMALL = mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4;
    static constexpr uint32_t L0C_BUFCNT = L0C_IS_SMALL ? 4U : 2U;
    static constexpr uint32_t L0C_BUF_BYTES =
        L0C_IS_SMALL ? (L0C_SIZE / NUM_4) * KB_TO_BYTES : (L0C_SIZE / NUM_2) * KB_TO_BYTES;

    /* ============确定key scale的类型============= */
    template <uint8_t kvLayoutType>
    struct KeyScaleGmToL1Sel {
        using Type = std::conditional_t<(kvLayoutType == 3), // 3: PA_NZ
                                        CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ, ScaleTrans::NO_TRANS>,
                                        CopyKeyScaleGmToL1<SCALE_T, K_SCALE_FORMAT, L1Format::NZ, ScaleTrans::DN2NZ>>;
    };

    /* ============确定value scale的类型============= */
    template <uint8_t kvLayoutType>
    struct ValueScaleGmToL1Sel {
        using Type =
            std::conditional_t<(kvLayoutType == 3), // 3: PA_NZ
                               CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::NO_TRANS>,
                               CopyValueScaleGmToL1<SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::ND2NZ>>;
    };

    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using KeyScaleGmToL1Type = typename KeyScaleGmToL1Sel<KvLayoutType>::Type;
    using ValueScaleGmToL1Type = typename ValueScaleGmToL1Sel<KvLayoutType>::Type;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    static constexpr ActualSeqLensMode Q_PARSER_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;
    using FaGmTensorQScale = FaGmTensor<SCALE_T, Q_SCALE_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKScale = FaGmTensor<SCALE_T, K_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;
    using FaGmTensorVScale = FaGmTensor<SCALE_T, V_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;

    using ConstInfoX = ConstInfo_t;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    FaGmTensorQScale queryScaleGm_;
    FaGmTensorKScale keyScaleGm_;
    FaGmTensorVScale valueScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    QSeqLensToolType &qSeqLensTool_;
    KvSeqLensToolType &kvSeqLensTool_;

    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, Q_M_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT> copyQueryScaleGmToL1_;
    KeyScaleGmToL1Type copyKeyScaleGmToL1_;
    ValueScaleGmToL1Type copyValueScaleGmToL1_;

    /* =====================LocalBuffer变量（fa式：裸数组+bufId_手工轮转）=====================*/
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;

    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    uint32_t mm1ResBufId_ = 0U;

    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> ubBmm2ResBuffers_;

    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;

    L0AType mmL0ABuffers_;
    L0BType mmL0BBuffers_;

    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    const ConstInfoX &constInfo_;

    /*============================================================================== */
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8(ConstInfoX &constInfo, QSeqLensToolType &qSeqLensTool,
                                                   KvSeqLensToolType &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery,
                                         __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue)
    {
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        // fa式L1布局：P区(跨核共享,AIV写/AIC读)从0起，Q/KV裸数组从P区末尾手工排布
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, L1_P_REGION_BYTES, L1_Q_BUFCNT * L1_Q_BUF_BYTES);
        l1KvBuffers_ = LocalTensor<uint8_t>(TPosition::A1, L1_P_REGION_BYTES + L1_Q_BUFCNT * L1_Q_BUF_BYTES,
                                            L1_KV_BUFCNT * L1_KV_BUF_BYTES);
        // L0A/L0B保留policy（LOCK_UNLOCK经matmul内部Wait/Set调用点映射），policy Init在AllocEventID（fa位置）
        l0aBufferManager_.Init(65536); // 64 * 1024
        l0bBufferManager_.Init(65536); // 64 * 1024
        // L0C裸数组（fa式）
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
        // 跨核共享UB区（bmm2Res单槽 + bmm1Res双槽，与vec侧同布局）
        ubBmm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, 0U, MM2_RES_BUF_BYTES);
        ubBmm1ResBuffers_ =
            LocalTensor<uint8_t>(TPosition::VECIN, MM2_RES_BUF_BYTES, MM1_RES_BUFCNT * MM1_RES_BUF_BYTES);
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        // AIC收尾drain：等两个AIV读完全部mmRes槽（free flag翻转到位）
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_1 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery,
                                         __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);

        InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size,
                         dBaseSize / MXFP_GROUP_SIZE, queryScaleGm_, dequantScaleQuery);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueStrides.bnStride, constInfo_.valueStrides.n2Stride);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize,
                         dBaseSize / MXFP_GROUP_SIZE, keyScaleGm_, dequantScaleKey);
        InitVScaleBuffer(constInfo_.bSize, constInfo_.s2Size / MXFP_DIVISOR_SIZE, constInfo_.n2Size,
                         constInfo_.blockSize / MXFP_DIVISOR_SIZE, constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE,
                         valueScaleGm_, dequantScaleValue);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, this->qSeqLensTool_.cuSeqLensParser);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim,
                                            this->qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, FaGmTensorQScale &qScaleGmTensor, __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, this->qSeqLensTool_.cuSeqLensParser);
        } else {
            qScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim,
                                                 this->qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm, uint64_t bnStride, uint64_t n2Stride)
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
        } else if constexpr (KV_NEEDS_WZH) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, this->kvSeqLensTool_.cuSeqLensParser);
        } else {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(this->kvSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorKScale &kScaleGmTensor, __gm__ uint8_t *gm)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                                 constInfo_.maxBlockNumPerBatch, constInfo_.kDescaleStrides.bnStride,
                                                 constInfo_.kDescaleStrides.n2Stride);
        } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_K_SCALE_PA_NZ) {
            uint32_t bs0 = 32 / sizeof(KV_T);
            uint32_t kvCacheBlockSize1 = kvCacheBlockSize / bs0 * MXFP_MULTI_BASE_SIZE;
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize1, headDim / MXFP_MULTI_BASE_SIZE, bs0,
                                                 blockTableGm_, constInfo_.maxBlockNumPerBatch,
                                                 constInfo_.kDescaleStrides.bnStride,
                                                 constInfo_.kDescaleStrides.n2Stride);
        } else if constexpr (KV_NEEDS_WZH) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, headDim, this->kvSeqLensTool_.cuSeqLensParser);
        } else {
            kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kScaleGmTensor.offsetCalculator.Init(this->kvSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorVScale &vScaleGmTensor, __gm__ uint8_t *gm)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                                 constInfo_.maxBlockNumPerBatch, constInfo_.vDescaleStrides.bnStride,
                                                 constInfo_.vDescaleStrides.n2Stride);
        } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            vScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                                 constInfo_.maxBlockNumPerBatch, constInfo_.vDescaleStrides.bnStride,
                                                 constInfo_.vDescaleStrides.n2Stride);
        } else if constexpr (KV_NEEDS_WZH) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, headDim, this->kvSeqLensTool_.cuSeqLensParser);
        } else {
            vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            vScaleGmTensor.offsetCalculator.Init(this->kvSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // fa位置：L0A/B policy Init（LOCK_UNLOCK，mutex ID经EXTERNAL注入）
        mmL0ABuffers_.Init(l0aBufferManager_, 32 * 1024, L0A_BUFFER_ID0, L0A_BUFFER_ID1);
        mmL0BBuffers_.Init(l0bBufferManager_, 32 * 1024, L0B_BUFFER_ID0, L0B_BUFFER_ID1);
    }

    __aicore__ inline void FreeEventID()
    {
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        if constexpr (USE_DN) {
            // DN分支下realGSize=1，gS1Idx即s1Idx，gIdx恒为0
            GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                                  .n2Idx = runInfo.realN2Idx,
                                  .gIdx = 0,
                                  .s1Idx = runInfo.gS1Idx,
                                  .dIdx = dOffset,
                                  .s1DealSize = runInfo.actMSize,
                                  .dDealSize = nopeDealSize};
            copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
        } else {
            GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                    .n2Idx = runInfo.realN2Idx,
                                    .gS1Idx = runInfo.gS1Idx,
                                    .dIdx = dOffset,
                                    .gS1DealSize = runInfo.actMSize,
                                    .dDealSize = nopeDealSize};
            copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyQuerySlice(dstTensor, 0, constInfo_.dSize, runInfo);
    }

    // copy query scale with full s1g
    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<SCALE_T> &dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.realN2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = dOffset,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dRealSize};
        copyQueryScaleGmToL1_(l1Tensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<SCALE_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, dBaseSize / MXFP_GROUP_SIZE, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                        uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(KV_T);
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;

        uint32_t nopeDealSize = dRealSize;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        // CopyKey的控制参数
        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = nopeDealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo, uint32_t s2RealSize,
                                       uint32_t subLoop)
    {
        uint32_t s2Offset = (subLoop % 2 == 0) ? runInfo.s2Idx : runInfo.s2Idx + s2BaseSize / 2;
        CopyKeySlice(dstTensor, s2Offset, s2RealSize, 0, constInfo_.dSize, runInfo);
    }

    // copy key scale with full s2
    __aicore__ inline void CopyKeyScaleSlice(const LocalTensor<SCALE_T> &dstTensor, uint32_t s2Offset,
                                             uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                             RunInfoX &runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKeyScaleGmToL1_(l1Tensor, keyScaleGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<SCALE_T> &dstTensor, RunInfoX &runInfo,
                                            uint32_t s2RealSize, uint32_t subLoop)
    {
        uint32_t s2Offset = subLoop % 2 == 0 ? runInfo.s2Idx : runInfo.s2Idx + s2BaseSize / 2;
        CopyKeyScaleSlice(dstTensor, s2Offset, s2RealSize, 0, dBaseSize / MXFP_GROUP_SIZE, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                          uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t dstStride = (s2RealSize + 63) >> 6 << 6;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size, 0, constInfo_.dSizeV, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyValueScaleSlice(const LocalTensor<SCALE_T> &dstTensor, uint32_t s2Offset,
                                               uint32_t s2RealSize, uint32_t dOffset, uint32_t dRealSize,
                                               RunInfoX &runInfo)
    {
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = s2RealSize};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyValueScaleGmToL1_(l1Tensor, valueScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyValueScaleTile(const LocalTensor<SCALE_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueScaleSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size / MXFP_DIVISOR_SIZE, 0,
                            constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr_));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr_, bIdx);
        keyGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSize);
        keyGm_.offsetCalculator.Init(this->kvSeqLensTool_.cuSeqLensParser);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr_));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr_, bIdx);
        valueGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSizeV);
        valueGm_.offsetCalculator.Init(this->kvSeqLensTool_.cuSeqLensParser);
    }

    __aicore__ inline void IterateBmm1(RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t mm1ResBufId = mm1ResBufId_;
        mm1ResBufId_ = (mm1ResBufId_ + 1) % MM1_RES_BUFCNT;
        LocalTensor<T> mm1ResUbTensor =
            ubBmm1ResBuffers_[mm1ResBufId * MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        uint32_t mmSyncIdx = CC_MM_BMM1_0 + mm1ResBufId;
        // 等两个AIV读完该槽（free），fixpipe写入后向两个AIV置ready
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        if constexpr (USE_DN) {
            IterateBmm1Dn(mm1ResUbTensor, runInfo, subLoop);
        } else {
            IterateBmm1Nd(mm1ResUbTensor, runInfo, subLoop);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, 左矩阵驻留. GS1<=80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Nd(LocalTensor<T> &mm1ResUb, RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t s2CurSize = s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size < s2BaseSize)) {
            if (subLoop == 0) {
                s2CurSize = AttentionCommon::Min((int32_t)s2SplitSize, (int32_t)runInfo.actSingleLoopS2Size);
            } else {
                s2CurSize = runInfo.actSingleLoopS2Size - s2SplitSize;
            }
        }
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        LocalTensor<SCALE_T> qL1ScaleTensor =
            l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES + qScaleOffset].template ReinterpretCast<SCALE_T>();
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            // Q驻留（fa式锁交接）：MTE2写入后交MTE1持有至本任务末（取代GetPre+裸Set惯用法）
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(qL1Tensor, mBaseSize, dBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyQueryTile(qL1Tensor, runInfo);
            CopyQueryScaleTile(qL1ScaleTensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        uint32_t kScaleOffset = ((s2CurSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> kL1ScaleTensor =
            l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES + kScaleOffset].template ReinterpretCast<SCALE_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        if constexpr (!isDAligned) {
            InitValueL1BufferNAxis(kL1Tensor, s2SplitSize, dBaseSize);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        CopyKeyTile(kL1Tensor, runInfo, s2CurSize, subLoop);
        CopyKeyScaleTile(kL1ScaleTensor, runInfo, s2CurSize, subLoop);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CurSize, dBaseSize, false, true);
            MatmulFullMX<Q_T, KV_T, T, 128, (s2BaseSize >> 1), dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                qL1Tensor, kL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param, qL1ScaleTensor, kL1ScaleTensor);
            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

            FixpipeMm1(mm1ResUb, l0CTensor, runInfo, s2CurSize);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                              (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 31) >> 5 << 5;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2SplitSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                        uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = 64; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 2; // 双目标模式，按M维度拆分， M * N / 2写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Dn(LocalTensor<T> &mm1ResUb, RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t s2CalcSize = s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size < s2BaseSize)) { // unlikely告诉编译器大概率走不到
            if (subLoop == 0) {
                s2CalcSize = AttentionCommon::Min((int32_t)s2SplitSize, (int32_t)runInfo.actSingleLoopS2Size);
            } else {
                s2CalcSize = runInfo.actSingleLoopS2Size - s2SplitSize;
            }
        }
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        LocalTensor<SCALE_T> qL1ScaleTensor =
            l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES + qScaleOffset].template ReinterpretCast<SCALE_T>();
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            // Q驻留（fa式锁交接）：MTE2写入后交MTE1持有至本任务末（取代GetPre+裸Set惯用法）
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(qL1Tensor, mBaseSize, dBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyQueryTile(qL1Tensor, runInfo);
            CopyQueryScaleTile(qL1ScaleTensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        uint32_t kScaleOffset = ((s2CalcSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> kL1ScaleTensor =
            l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES + kScaleOffset].template ReinterpretCast<SCALE_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        if constexpr (!isDAligned) {
            InitValueL1BufferNAxis(kL1Tensor, s2SplitSize, dBaseSize);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        CopyKeyTile(kL1Tensor, runInfo, s2CalcSize, subLoop);
        CopyKeyScaleTile(kL1ScaleTensor, runInfo, s2CalcSize, subLoop);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            MMParam param = MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, dBaseSize, false, true);
            MatmulFullMX<Q_T, KV_T, T, (s2BaseSize >> 1), 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                         SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                kL1Tensor, qL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param, kL1ScaleTensor, qL1ScaleTensor);
            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

            FixpipeMm1Dn(mm1ResUb, l0CTensor, runInfo, s2CalcSize);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                              (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm2(RunInfoX &runInfo)
    {
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<uint8_t> pL1Buf = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        LocalTensor<T> mm2ResUbTensor = ubBmm2ResBuffers_.template ReinterpretCast<T>();
        // 等两个AIV写完该P槽（ready，MTE3置位）；bmm2Res单槽等free
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0 + AIV0_AIV1_OFFSET);
        if constexpr (USE_DN) {
            IterateBmm2Dn(mm2ResUbTensor, pL1Buf, runInfo);
        } else {
            IterateBmm2Nd(mm2ResUbTensor, pL1Buf, runInfo);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void IterateBmm2Nd(LocalTensor<T> &mm2ResUbTensor, LocalTensor<uint8_t> &pL1Buf,
                                         RunInfoX &runInfo)
    {
        LocalTensor<INPUT_T> pL1Tensor = pL1Buf.template ReinterpretCast<INPUT_T>();
        uint32_t pScaleOffset =
            mBaseSize * s2BaseSize; // PScale在L1P的偏移量（单位：元素）; // VScale在mm1B的偏移量（单位：元素）
        LocalTensor<SCALE_T> mm2AScaleFakeTensor = pL1Buf[pScaleOffset].template ReinterpretCast<SCALE_T>();

        LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
        Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);

        constexpr uint32_t baseK = s2SplitSize;
        uint64_t l1BaseKOffset = baseK * mBaseSize;
        uint64_t l1ScaleOffset = baseK / 32 * runInfo.actMSizeAlign32;
        uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1) / baseK;
        uint32_t realK = baseK;
        for (uint32_t kIdx = 0; kIdx < kLoops; kIdx++) {
            LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
            if (kIdx == kLoops - 1) {
                realK = runInfo.actSingleLoopS2Size - kIdx * baseK;
            }
            Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            if (unlikely(runInfo.isLastS2Loop)) {
                InitValueL1BufferNoTrans(vL1Tensor, realK, (uint32_t)constInfo_.dSizeV);
            }
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(vL1Tensor, baseK + baseK / MXFP_GROUP_SIZE, dVBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyValueSlice(vL1Tensor, runInfo.s2Idx + kIdx * baseK, realK, 0, constInfo_.dSizeV, runInfo);
            uint32_t vScaleOffset = (((realK) + 63) >> 6 << 6) * dVBaseSize;
            LocalTensor<SCALE_T> vL1ScaleTensor =
                l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES + vScaleOffset].template ReinterpretCast<SCALE_T>();
            CopyValueScaleSlice(vL1ScaleTensor, (runInfo.s2Idx + kIdx * baseK) / MXFP_DIVISOR_SIZE,
                                (realK + 63) / MXFP_DIVISOR_SIZE, 0, constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
            Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            {
                MMParam param =
                    MakeMMParam((uint32_t)mBaseSize, dVBaseSize, (realK + 63) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE,
                                false, false, kIdx == 0, kIdx == 0);
                if constexpr (!USE_DN) {
                    param.realM = (uint32_t)runInfo.actMSize;
                }
                MatmulFullMX<INPUT_T, KV_T, T, 128, dVBaseSize, baseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                             SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                    pL1Tensor[kIdx * l1BaseKOffset], vL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param,
                    mm2AScaleFakeTensor[kIdx * l1ScaleOffset], vL1ScaleTensor);
            }
            Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
        }

        Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
        Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

        FixpipeMm2(mm2ResUbTensor, l0CTensor, runInfo); // 数据搬出

        Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
        l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = (constInfo_.dSizeV + 7) >> 3 << 3;

        if constexpr (!USE_DN) {
            fixpipeParams.mSize = (runInfo.actMSize + 31) >> 5 << 5;
            fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        } else {
            fixpipeParams.mSize = mBaseSize;
            fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        }

        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2Dn(LocalTensor<T> &mm2ResUbTensor, LocalTensor<uint8_t> &pL1Buf,
                                         RunInfoX &runInfo)
    {
        LocalTensor<INPUT_T> pL1Tensor = pL1Buf.template ReinterpretCast<INPUT_T>();
        constexpr uint32_t pScaleOffset = mBaseSize * s2BaseSize; // pScale在L1的偏移量
        constexpr uint32_t sizeOfInt16 = 2;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            LocalTensor<int16_t> mm2AScaleTensor =
                pL1Buf[pScaleOffset + mBaseSize * s2SplitSize / MXFP_GROUP_SIZE].template ReinterpretCast<int16_t>();
            InitConstValue(mm2AScaleTensor,
                           {1, mBaseSize * s2SplitSize / MXFP_GROUP_SIZE * sizeof(SCALE_T) / blockBytes, 0, 0x7f7f});
        } else if (runInfo.actSingleLoopS2Size <= s2SplitSize) {
            LocalTensor<int16_t> mm2AScaleTensor = pL1Buf[pScaleOffset].template ReinterpretCast<int16_t>();
            InitConstValue(mm2AScaleTensor,
                           {1, mBaseSize * s2SplitSize / MXFP_GROUP_SIZE * sizeof(SCALE_T) / blockBytes, 0, 0x7f7f});
        }
        LocalTensor<SCALE_T> mm2AScaleFakeTensor = pL1Buf[pScaleOffset].template ReinterpretCast<SCALE_T>();

        LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
        Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);

        constexpr uint32_t baseK = s2SplitSize;
        uint64_t l1BaseKOffset = baseK * mBaseSize;
        uint64_t l1ScaleOffset = l1BaseKOffset / MXFP_GROUP_SIZE;
        uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1) / baseK;
        uint32_t realK = baseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
            if (k == kLoops - 1) {
                realK = runInfo.actSingleLoopS2Size - k * baseK;
            }
            Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            if (unlikely(runInfo.isLastS2Loop)) {
                InitValueL1BufferNoTrans(vL1Tensor, realK, (uint32_t)constInfo_.dSizeV);
            }
            if constexpr (!isDAligned) {
                InitValueL1BufferNAxis(vL1Tensor, baseK + baseK / MXFP_GROUP_SIZE, dVBaseSize);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
            CopyValueSlice(vL1Tensor, runInfo.s2Idx + k * baseK, realK, 0, constInfo_.dSizeV, runInfo);
            uint32_t vScaleOffset = (((realK) + 63) >> 6 << 6) * dVBaseSize;
            LocalTensor<SCALE_T> vL1ScaleTensor =
                l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES + vScaleOffset].template ReinterpretCast<SCALE_T>();
            CopyValueScaleSlice(vL1ScaleTensor, (runInfo.s2Idx + k * baseK) / MXFP_DIVISOR_SIZE,
                                (realK + 63) / MXFP_DIVISOR_SIZE, 0, constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE, runInfo);
            Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            {
                MMParam param =
                    MakeMMParam((uint32_t)mBaseSize, dVBaseSize, (realK + 63) / MXFP_DIVISOR_SIZE * MXFP_DIVISOR_SIZE,
                                USE_DN, false, k == 0, k == 0);
                MatmulFullMX<INPUT_T, KV_T, T, 128, dVBaseSize, baseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                             SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                    pL1Tensor[k * l1BaseKOffset], vL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param,
                    mm2AScaleFakeTensor[k * l1ScaleOffset], vL1ScaleTensor);
            }
            Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
            kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;
        }

        Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
        Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

        FixpipeMm2(mm2ResUbTensor, l0CTensor, runInfo);

        Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
        l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
    }

    __aicore__ inline void InitValueL1BufferNAxis(const LocalTensor<KV_T> &valueL1, const uint32_t k, const uint32_t n)
    {
        InitConstValueParams<half> initConstValueParams;
        initConstValueParams.repeatTimes = n / 32U;
        initConstValueParams.blockNum = k;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>(), initConstValueParams);
    }

    __aicore__ inline void InitValueL1BufferNoTrans(const LocalTensor<KV_T> &valueL1, const uint32_t realK,
                                                    const uint32_t curN)
    {
        InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        uint32_t curPadK = (realK + 63) >> 6 << 6;
        // nd2nz pading to k 64 align, (k,n)->(n1,k1,k0,n0)
        uint32_t movK = realK;
        if (curPadK == movK) {
            return;
        }
        offset = movK * 16U;
        initConstValueParams.repeatTimes = AttentionCommon::Align(curN, 16U) / 32U;
        initConstValueParams.blockNum = curPadK - movK;
        initConstValueParams.dstGap = movK;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>()[offset], initConstValueParams);
    }

    __aicore__ inline void InitValueL1BufferTrans(const LocalTensor<KV_T> &valueL1, const uint32_t realK,
                                                  const uint32_t curN)
    {
        InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        uint32_t curPadK = ((realK) + 63) >> 6 << 6;
        // nd2nz pading to 64 align, (n,k)->(k1,n1,n0,k0)
        if (curPadK - realK < 32U) {
            return;
        }
        // pad n1, n0, 16 for half
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = AttentionCommon::Align(curN, 16U);
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = 0;
        uint64_t kAlign = AttentionCommon::Align(realK, 16U) / 32U * 16U;
        offset = curN * kAlign;
        InitConstValue(valueL1.template ReinterpretCast<half>()[offset], initConstValueParams);
    }
}; // QuantFlashAttnBlockCubeMxfp8

template <typename FA_T>
class QuantFlashAttnBlockCubeMxfp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr LayOutTypeEnum LAYOUT = FA_T::qLayout;
    static constexpr LayOutTypeEnum layout = FA_T::qLayout; // 内部小写引用兼容（GetKVGmFormat等）
    static constexpr uint8_t KvLayoutType = FA_T::kvLayoutType;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool USE_DN = FA_T::useDn;

    using Q_T = typename FA_T::inputType;
    using KV_T = typename FA_T::inputType;
    using MM_T = float;

    using ConstInfoX = ConstInfo_t;
    using QSeqLensToolType = QFASeqLensTool<FA_T::qLayout, false, false, int32_t, GetQActSeqMode<FA_T::qLayout>()>;
    using KvSeqLensToolType = QFASeqLensTool<FA_T::kvLayout, FA_T::pageAttention, false, int32_t,
                                             GetKvActSeqMode<FA_T::kvLayout, FA_T::pageAttention>()>;
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8Dummy(ConstInfoX &constInfo, QSeqLensToolType &,
                                                        KvSeqLensToolType &){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_H_
