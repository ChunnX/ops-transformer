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
 * \file quant_flash_attn_block_cube_mxfp8_fp16.h
 * \brief quant_mode=3 (QFA_MXFP8_FP16) cube 层:
 *        基本块 128x512, CV 1:1, MM1 FIXPIPE 随路 FP32->FP16 并乘 softmaxscale;
 *        MM2 DN 方案 (V^T x P^T, A_L1 尾行全1随路得出 row_sum)
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_FP16_H_
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

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
class QuantFlashAttnBlockCubeMxfp8Fp16 {
public:
    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type = BuffersPolicyDB<BufferType::L0A>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = BuffersPolicyDB<BufferType::L0B>;
    };
    /* ============确定L0C的类型============= */
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = BuffersPolicyDB<BufferType::L0C>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr uint32_t s2SplitSize = 256U; // 仅支持d=128, 无需按d=256缩小split
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t MXFP_DIVISOR_SIZE = 64U;
    static constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2U;
    /* ============MM2 (FP16 softmax) DN方案参数============= */
    // 基本块128x512拆4次进mm2, 每次算(d+32)x128
    static constexpr uint32_t mm2BaseK = 128U;
    // A = V^T: m方向 = d + 32行全1带(随路得出row_sum, 各行全等), 直接32对齐免独立pad
    static constexpr uint32_t mm2VtMSize = dVBaseSize + 32U;
    // A数据区(fp8元素)与scale区
    static constexpr uint32_t mm2VtDataSize = mm2VtMSize * s2BaseSize;
    static constexpr uint32_t mm2VtScaleSize = mm2VtMSize * s2BaseSize / MXFP_GROUP_SIZE;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    // 仅DN方案(prefill): Q按S1_ONLY布局, realG=1, gIdx恒为0
    static constexpr InnerMLayout Q_M_LAYOUT = InnerMLayout::S1_ONLY_LAYOUT;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType>();
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, true>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<layout, KvLayoutType>();
    static constexpr GmFormat V_SCALE_FORMAT = GetValueScaleGmFormat<layout, KvLayoutType>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = T;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_PBUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_PBUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

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

    // K与V^T的L1 tile大小不同(V^T含全1带), 由原KV共享4buff改为各自独立2buff池
    using L1KType = BuffersPolicyDB<BufferType::L1>;
    // MM2 (FP16 softmax): A=V^T专用2buff乒乓(两bank全1带均在kernel启动时初始化一次, 数据搬运只写真实d通道)
    using L1VtType = BuffersPolicyDB<BufferType::L1>;
    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSel<mBaseSize, s2BaseSize, dVBaseSize>::Type;
    using KeyScaleGmToL1Type = typename KeyScaleGmToL1Sel<KvLayoutType>::Type;
    using ValueScaleGmToL1Type = typename ValueScaleGmToL1Sel<KvLayoutType>::Type;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    static constexpr ActualSeqLensMode Q_PARSER_MODE = GetQActSeqMode<LAYOUT>();
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, false>();

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_PARSER_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_PARSER_MODE, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;
    using FaGmTensorQScale = FaGmTensor<SCALE_T, Q_SCALE_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKScale = FaGmTensor<SCALE_T, K_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;
    using FaGmTensorVScale = FaGmTensor<SCALE_T, V_SCALE_FORMAT, int32_t, KV_NEEDS_WZH>;

    using ConstInfoX = ConstInfo_t;
    TPipe *tPipe_ = nullptr;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    FaGmTensorQScale queryScaleGm_;
    FaGmTensorKScale keyScaleGm_;
    FaGmTensorVScale valueScaleGm_;

    QSeqParserType *qSeqParserPtr_ = nullptr;
    KvSeqParserType *kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT, L1Format::NZ, Q_M_LAYOUT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;
    CopyQueryScaleGmToL1<SCALE_T, Q_SCALE_FORMAT> copyQueryScaleGmToL1_;
    KeyScaleGmToL1Type copyKeyScaleGmToL1_;
    ValueScaleGmToL1Type copyValueScaleGmToL1_;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::L1> *l1BufferManagerPtr_;
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;

    BuffersPolicyDB<BufferType::L1> l1QBuffers_;
    L1KType l1KBuffers_;
    L1VtType l1VtBuffers_;

    L0AType mmL0ABuffers_;
    L0BType mmL0BBuffers_;
    L0CType mmL0CBuffers_;

    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    const ConstInfoX &constInfo_;

    /*============================================================================== */
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8Fp16(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                         __gm__ uint8_t *dequantScaleValue, QSeqParserType &qParser,
                                         KvSeqParserType &kvParser)
    {
        tPipe_ = pipe;
        l1BufferManagerPtr_ = l1BuffMgr;
        this->qSeqParserPtr_ = &qParser;
        this->kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        constexpr uint32_t mm1QSize = mBaseSize * dBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1QScaleSize = mBaseSize * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T);
        constexpr uint32_t mmKVSize = dBaseSize * s2BaseSizeCur * sizeof(INPUT_T);
        constexpr uint32_t mmKVScaleSize = s2BaseSizeCur * dBaseSize / MXFP_GROUP_SIZE * sizeof(SCALE_T);
        l1QBuffers_.Init((*l1BufferManagerPtr_), mm1QSize + mm1QScaleSize);
        // K与V^T大小不同(且V^T需独立全1初始化), 拆分为各自独立的2buff池
        l1KBuffers_.Init((*l1BufferManagerPtr_), mmKVSize + mmKVScaleSize);
        // MM2 (FP16 softmax): A=V^T专用2buff乒乓, 数据区 + scale区
        constexpr uint32_t mm2VtBufSize = mm2VtDataSize * sizeof(INPUT_T) + mm2VtScaleSize * sizeof(SCALE_T);
        l1VtBuffers_.Init((*l1BufferManagerPtr_), mm2VtBufSize);
        // A(V^T)两个bank均在kernel启动时全置1(数据0x38/scale 0x7f), 此后各k段搬运只写真实d通道,
        // 全1带(m方向尾32行)永远保持1.0不被踩
        InitVtL1AllOne();
        // 开机fill与首个s2 loop的V^T搬运地址重叠(真实d通道), 同队列顺序执行不保证读写完成,
        // 需排空在飞fill后才能开始搬运
        AscendC::PipeBarrier<PIPE_MTE2>();

        // L0A B C 当前写死，能否通过基础api获取
        l0aBufferManager_.Init(tPipe_, 65536);  // 64 * 1024
        l0bBufferManager_.Init(tPipe_, 65536);  // 64 * 1024
        l0cBufferManager_.Init(tPipe_, 262144); // 256 * 1024
        // L0A B C当前写死，要改成通过计算获取
        mmL0ABuffers_.Init(l0aBufferManager_, 32 * 1024);
        mmL0BBuffers_.Init(l0bBufferManager_, 32 * 1024);

        if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
            mmL0CBuffers_.Init(l0cBufferManager_, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
        } else {
            mmL0CBuffers_.Init(l0cBufferManager_, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        }
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery,
                                         __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue)
    {
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
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, FaGmTensorQScale &qScaleGmTensor, __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
        } else {
            qScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm, uint64_t bnStride, uint64_t n2Stride)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);
        if constexpr (KV_NEEDS_WZH) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorKScale &kScaleGmTensor, __gm__ uint8_t *gm)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (KV_NEEDS_WZH) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                            uint32_t kvCacheBlockSize, uint32_t headDim,
                                            FaGmTensorVScale &vScaleGmTensor, __gm__ uint8_t *gm)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        if constexpr (KV_NEEDS_WZH) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr_);
        } else {
            vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            vScaleGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr_);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers_.Uninit((*l1BufferManagerPtr_));
        l1KBuffers_.Uninit((*l1BufferManagerPtr_));
        l1VtBuffers_.Uninit((*l1BufferManagerPtr_));
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
        mmL0CBuffers_.Uninit(l0cBufferManager_);
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        constexpr uint32_t blockNumDtype = 32 / sizeof(Q_T);
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        // DN方案下realGSize=1，gS1Idx即s1Idx，gIdx恒为0
        GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.realN2Idx,
                              .gIdx = 0,
                              .s1Idx = runInfo.gS1Idx,
                              .dIdx = dOffset,
                              .s1DealSize = runInfo.actMSize,
                              .dDealSize = nopeDealSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
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

    /* =====================MM2 (FP16 softmax) A位: V^T搬运与全1带===================== */
    // A = V^T ((d+32) x K): V GM (s2行, d列)经Dn2Nz转置到L1 A位, 只写真实d通道, 不踩全1带;
    // 用common层CopySingleMXScaleDNToNZ(fp8数据2列一16bit, 与scale搬运同构):
    // 源(s2行, d列)行主序 -> 转置(d, s2); nValue/srcDValue按2倍E8M0入参(内部/2转16bit)
    __aicore__ inline void CopyValueAsATile(LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t realK,
                                            RunInfoX &runInfo)
    {
        uint64_t offset = valueGm_.offsetCalculator.GetOffset(runInfo.bIdx, runInfo.n2Idx, s2Offset, 0);
        CopySingleMXScaleDNToNZ(dstTensor, valueGm_.gmTensor[offset], constInfo_.dSizeV, realK,
                                valueGm_.offsetCalculator.GetStrideS2(), mm2VtMSize);
    }

    // A位scale: VScale按DN存储(B,N,S//64,D,2), DN2NZ转置搬运, 用common层CopySingleMXScaleDNToNZ
    // (与K/Q scale的DN2NZ同一函数): nValue/srcDValue/dstNzC0Stride按2倍E8M0入参(内部/2转16bit),
    // dstNzC0Stride=2*mm2VtMSize对应A位m方向(含全1带)
    __aicore__ inline void CopyValueScaleAsATile(const LocalTensor<SCALE_T> &dstTensor, uint32_t s2Offset,
                                                 uint32_t s2ScaleRows, RunInfoX &runInfo)
    {
        uint64_t offset =
            valueScaleGm_.offsetCalculator.GetOffset(runInfo.bIdx, runInfo.n2Idx, s2Offset / MXFP_DIVISOR_SIZE, 0);
        CopySingleMXScaleDNToNZ(dstTensor, valueScaleGm_.gmTensor[offset], constInfo_.dSizeV * MXFP_MULTI_BASE_SIZE,
                                s2ScaleRows, valueScaleGm_.offsetCalculator.GetStrideS2(),
                                mm2VtMSize * MXFP_MULTI_BASE_SIZE);
    }

    // A位尾k段pad清零: Mmad的k按64对齐, realK后到64对齐边界未写区域清零
    __aicore__ inline void ClearVtKPad(const LocalTensor<KV_T> &vtL1, uint32_t kStart, uint32_t kEnd)
    {
        uint32_t padK = kEnd - kStart;
        if (padK == 0) {
            return;
        }
        // A的L1 NZ布局按k列组(32个fp8列)组织: 列组g起点 = g * mm2VtMSize * 32B(每列组内m行连续, 每行32B)
        // pad列组[gStart, gEnd)内清零m方向[0, dSizeV)行(真实d通道), 全1带[dSizeV, mm2VtMSize)不动;
        // gStart按ceil(realK/32)向上取整: 组内残尾由Dn2Nz硬件补零覆盖(与原mxfp8 InitValueL1BufferTrans
        // 的pad<32不清同构), realK任意取值均不会误清有效数据
        uint32_t gStart = (kStart + 31) >> 5;
        uint32_t gEnd = (kEnd + 31) >> 5;
        for (uint32_t g = gStart; g < gEnd; g++) {
            uint64_t offset = static_cast<uint64_t>(g) * mm2VtMSize * 32U;
            InitConstValueParams<half> initParams;
            initParams.repeatTimes = 1;
            initParams.blockNum = constInfo_.dSizeV; // dSizeV行 x 32B/行 = dSizeV个32B块
            initParams.dstGap = 0;
            initParams.initValue = 0;
            InitConstValue(vtL1.template ReinterpretCast<half>()[offset / 2], initParams);
        }
    }

    // A位一次性初始化: 两个bank的整块V^T L1(数据+scale)开机全置1(fp8数据0x38, scale 0x7f, 均为1.0);
    // 此后各k段V/VScale DN2NZ搬运只写真实d通道, 全1带(尾32行)及其scale永远保持1.0不被踩,
    // Mmad后L0C尾带行自然得row_sum。整块刷写免去band/scale区两级初始化的块几何拼装。
    // (DB池刚Init完无在飞事件, 连续两次Get()恰好遍历ping/pong且轮转flag复位回初始态,
    //  k循环首次Get()仍从ping开始, 行为确定)
    __aicore__ inline void InitVtL1AllOne()
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            Buffer<BufferType::L1> &vtBuf = l1VtBuffers_.Get();
            // 数据区: mm2VtDataSize个fp8全置1.0(fp8_e4m3 1.0 = 0x38, 两个打包0x3838)
            InitConstValueParams<uint16_t> vtDataInitParams(1, static_cast<uint16_t>(mm2VtDataSize / 32), 0, 0x3838);
            Fill(vtBuf.template GetTensor<KV_T>().template ReinterpretCast<uint16_t>(), vtDataInitParams);

            AscendC::PipeBarrier<PIPE_MTE2>();

            // scale区: mm2VtScaleSize个E8M0全置1.0(E8M0 1.0f = 0x7f, 两个打包0x7f7f)
            InitConstValueParams<uint16_t> vtScaleInitParams(1, static_cast<uint16_t>(mm2VtScaleSize / 32), 0, 0x7f7f);
            Fill(vtBuf.template GetTensor<SCALE_T>(mm2VtDataSize).template ReinterpretCast<uint16_t>(),
                 vtScaleInitParams);
        }
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr_));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr_, bIdx);
        keyGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSize);
        keyGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr_));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr_, bIdx);
        valueGm_.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm_.offsetCalculator.Init(0, constInfo_.n2Size, s2Size, constInfo_.dSizeV);
        valueGm_.offsetCalculator.Init(*this->kvSeqParserPtr_);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo, uint32_t subLoop)
    {
        uint32_t s2CalcSize = s2SplitSize;
        if (unlikely(runInfo.actSingleLoopS2Size < s2BaseSize)) { // unlikely告诉编译器大概率走不到
            if (subLoop == 0) {
                s2CalcSize = AttentionCommon::Min((int32_t)s2SplitSize, (int32_t)runInfo.actSingleLoopS2Size);
            } else {
                s2CalcSize = runInfo.actSingleLoopS2Size - s2SplitSize;
            }
        }
        Buffer<BufferType::L1> mm1B;
        uint32_t qScaleOffset = ((runInfo.actMSize + 31) >> 5 << 5) * dBaseSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0)) {
            mm1B = l1QBuffers_.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>(); // B矩阵常驻，两次Sub中，只有第一次搬B
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);

            LocalTensor<SCALE_T> mm1BScaleTensor = mm1B.GetTensor<SCALE_T>(qScaleOffset);
            CopyQueryScaleTile(mm1BScaleTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers_.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KBuffers_.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();
        CopyKeyTile(mm1ATensor, runInfo, s2CalcSize, subLoop);

        uint32_t kScaleOffset = ((s2CalcSize + 31) >> 5 << 5) * dBaseSize;
        LocalTensor<SCALE_T> mm1AScaleTensor = mm1A.GetTensor<SCALE_T>(kScaleOffset);
        CopyKeyScaleTile(mm1AScaleTensor, runInfo, s2CalcSize, subLoop);
        mm1A.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers_.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        MMParam param = MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, dBaseSize, false, true);
        MatmulFullMX<Q_T, KV_T, T, (s2BaseSize >> 1), 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                     SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
            mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
            mm1ResL0C.GetTensor<T>(), param, mm1A.GetTensor<SCALE_T>(kScaleOffset),
            mm1B.GetTensor<SCALE_T>(qScaleOffset));

        if (unlikely(runInfo.isLastS2Loop && (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                                              (runInfo.actSingleLoopS2Size <= s2SplitSize)))) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

        outputBuf.WaitCrossCore();

        FixpipeMm1(outputBuf.template GetTensor<half>(), mm1ResL0C.GetTensor<T>(), runInfo, s2CalcSize);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<half> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
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
        fixpipeParams.dstStride = mBaseSize; // mmResUb上两行之间的间隔，单位：element; CV 1:1单V核整行输出
        fixpipeParams.dualDstCtl = 0; // 单目标模式: (s2 x 128)行主序整块写入, V2按128元素寄存器整行读
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        // C1 FIXPIPE 随路: FP32 -> FP16 (QF322F16_PRE), 并乘以 softmaxscale (deqScalar)
        float softmaxScale = constInfo_.scaleValue;
        fixpipeParams.quantPre = QuantMode_t::QF322F16_PRE;
        fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&softmaxScale));

        Fixpipe<half, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* ============ MM2 (FP16 softmax) DN方案: A=V^T((d+32)xK) x B=P(Kx128) ============
     * 基本块128x512拆4次进mm2, 每次算(d+32)x128x128;
     * A的m方向尾32行为全1带(直接32对齐) -> Mmad(m=d+32)后L0C尾32行自然为row_sum(全等), 零额外开销;
     * A(V^T)在L1为专用2buff乒乓, 两bank全1带均在kernel启动时初始化一次, k循环数据搬运只写真实d通道;
     * B(P)由vec层写入l1PBuffers_(3buff轮转), isRightTranspose=true走LoadData转置分支;
     * C输出(d+32)x128 float, fixpipe到UB后由V2除以row_sum并cast bf16。
     */
    __aicore__ inline void IterateBmm2(MM2_DBUF_T &outputBuf, MM2_PBUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        // 1. 等待P就绪(vec层VF写完P已SetCrossCore)
        MM2_PBUF_T mm2P = inputBuf.Get();
        mm2P.WaitCrossCore();

        // 2. P的scale区域初始化0x7f(E8M0 1.0): vec层仅在subLoop%2==1时写入P0修正,
        //    其余k段(前半/单subLoop尾块)的scale必须为1.0, 每个P buffer(3buff轮转)每次使用前初始化
        InitPScaleRegion(mm2P, runInfo);

        // 3. k循环: baseK=128, 每轮 搬V^T段 -> 搬P scale修正(仅subLoop奇数次) -> Mmad累加
        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers_.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();
        IterateBmm2KLoop(mm2P, mm2ResL0C, runInfo);

        // 4. fixpipe搬出: (d+32)x128 float到UB, 供V2做row_sum除法
        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();
        outputBuf.WaitCrossCore();
        FixpipeMm2(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), runInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    // P的scale区域初始化(对齐原mxfp8逻辑): E8M0 1.0=0x7f7f, P buffer(3buff轮转)每次使用前初始化;
    // 双subLoop(>s2SplitSize): vec在subLoop1写入前半的P0修正(从scale区起点起), 此处初始化后半;
    // 单subLoop(<=s2SplitSize): vec不写修正, 此处初始化前半(即本loop实际占用的scale区域)
    __aicore__ inline void InitPScaleRegion(MM2_PBUF_T &mm2P, RunInfoX &runInfo)
    {
        constexpr uint32_t pScaleOffset = mBaseSize * s2BaseSize; // PScale在L1的偏移量(元素)
        constexpr uint32_t sizeOfInt16 = 2;
        // 前半scale个数(subLoop0的k列对应区域)
        constexpr uint32_t halfScaleNum = mBaseSize * s2SplitSize / MXFP_GROUP_SIZE;
        constexpr uint32_t initBlockNum = halfScaleNum * sizeof(SCALE_T) / blockBytes; // 1KB=32块
        uint32_t initOffsetInt16 = runInfo.actSingleLoopS2Size > s2SplitSize ?
                                       pScaleOffset / sizeOfInt16 + halfScaleNum / sizeOfInt16 : // 后半起点
                                       pScaleOffset / sizeOfInt16;                               // 前半起点
        LocalTensor<int16_t> mm2PScaleTensor = mm2P.GetTensor<int16_t>(initOffsetInt16);
        InitConstValue(mm2PScaleTensor, {1, initBlockNum, 0, 0x7f7f});
    }

    // MM2 k循环: 总K(本loopS2)拆baseK=128段, 每段V^T搬运(Dn2Nz直通)+P scale修正+Mmad累加;
    // V^T为2buff乒乓: k循环Get()按段轮转bank, CopyVtSegToL1/MmadBmm2Seg内部用GetPre()取同段bank,
    // 使段k+1的MTE2写入与段k的MTE1装载重叠; 每bank事件对做MTE2(写L1)/MTE1(LoadData读L1)同步:
    // Wait<MTE1_MTE2>等该bank上一轮读完 -> Copy写本段 -> Set/Wait<MTE2_MTE1>保证读前写完成 -> Mmad -> Set通知读完
    __aicore__ inline void IterateBmm2KLoop(MM2_PBUF_T &mm2P, Buffer<BufferType::L0C> &mm2ResL0C, RunInfoX &runInfo)
    {
        uint32_t kLoops = (runInfo.actSingleLoopS2Size + mm2BaseK - 1) / mm2BaseK;
        uint32_t realK = mm2BaseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            if (k == kLoops - 1) {
                realK = runInfo.actSingleLoopS2Size - k * mm2BaseK;
            }
            Buffer<BufferType::L1> &vtBuf = l1VtBuffers_.Get(); // 按段轮转bank
            vtBuf.Wait<HardEvent::MTE1_MTE2>(); // 等该bank上一段LoadData读完(首次由Init阶段置就绪)
            CopyVtSegToL1(k * mm2BaseK, realK, runInfo); // MTE2写本段V^T
            vtBuf.Set<HardEvent::MTE2_MTE1>();           // 通知MTE1可读
            vtBuf.Wait<HardEvent::MTE2_MTE1>();          // LoadData前等本段写完成
            // 本段Mmad: m=d+32, n=mBaseSize, k=realK(64对齐), 尾带32行=row_sum(全等)
            MmadBmm2Seg(mm2P, mm2ResL0C, k, realK, runInfo);
            vtBuf.Set<HardEvent::MTE1_MTE2>(); // 读完成, 下一段/下一loop可写
        }
    }

    // V^T单段(kStart起realK行)搬运到A位L1: 数据+scale DN2NZ转置, 尾k段pad清零
    // (GetPre: k循环已Get()轮转bank, 此处取同段bank不轮转)
    __aicore__ inline void CopyVtSegToL1(uint32_t kStart, uint32_t realK, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1> &vtBuf = l1VtBuffers_.GetPre();
        LocalTensor<KV_T> vtL1 = vtBuf.template GetTensor<KV_T>();
        LocalTensor<SCALE_T> vtScaleL1 = vtBuf.template GetTensor<SCALE_T>(mm2VtDataSize);

        // 数据: V GM(s2行,d列) -> A位NZ(d行,s2列), 只写真实d通道
        LocalTensor<KV_T> segTensor = vtL1[kStart * mm2VtMSize];
        CopyValueAsATile(segTensor, runInfo.s2Idx + kStart, realK, runInfo);
        // scale: VScale DN存储(B,N,S//64,D,2) -> A位scale NZ, 只写真实d通道;
        // scale段偏移 = 数据段偏移/32(每32个fp8数据1个E8M0, 与mm2VtScaleSize及LoadData寻址>>5自洽)
        LocalTensor<SCALE_T> segScaleTensor = vtScaleL1[kStart * mm2VtMSize / MXFP_GROUP_SIZE];
        CopyValueScaleAsATile(segScaleTensor, runInfo.s2Idx + kStart,
                              (realK + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE, runInfo);
        // 尾段k pad清零: Mmad的k按64对齐, realK后到边界未写区域清零
        uint32_t kAlign64End = (kStart + realK + 63) >> 6 << 6;
        ClearVtKPad(vtL1, kStart + realK, kStart + mm2BaseK < kAlign64End ? kStart + mm2BaseK : kAlign64End);
    }

    // 单段Mmad: A=V^T直通(ifTranspose=false), B=P转置(isRightTranspose=true), m=d+32尾带32行=row_sum(全等)
    // (GetPre: k循环已Get()轮转bank, 此处取同段bank不轮转)
    __aicore__ inline void MmadBmm2Seg(MM2_PBUF_T &mm2P, Buffer<BufferType::L0C> &mm2ResL0C, uint32_t k, uint32_t realK,
                                       RunInfoX &runInfo)
    {
        Buffer<BufferType::L1> &vtBuf = l1VtBuffers_.GetPre();
        LocalTensor<KV_T> vtL1 = vtBuf.template GetTensor<KV_T>();
        LocalTensor<SCALE_T> vtScaleL1 = vtBuf.template GetTensor<SCALE_T>(mm2VtDataSize);

        // k段在A位L1的偏移(fp8元素): 每k列组一个m方向stride; scale段偏移=数据偏移/32(每32个数据1个E8M0)
        uint32_t kElemOffset = k * mm2BaseK * mm2VtMSize;
        uint32_t kScaleElemOffset = kElemOffset / MXFP_GROUP_SIZE;

        // B位(P)的scale: P的E8M0按行scale(P0修正, vec层已写入L1), 按k段偏移寻址
        // (与原mxfp8的 mm2AScaleFakeTensor[k*l1ScaleOffset] 同构: l1ScaleOffset = baseK*mBaseSize/32)
        constexpr uint32_t pScaleOffset = mBaseSize * s2BaseSize;                     // pScale在L1的偏移量(元素)
        constexpr uint32_t pScaleKSegOffset = mm2BaseK * mBaseSize / MXFP_GROUP_SIZE; // 每k段scale偏移
        LocalTensor<SCALE_T> pScaleFakeTensor = mm2P.GetTensor<SCALE_T>(pScaleOffset);

        // 单段Mmad参数: A=V^T直通(isLeftTranspose=false), B=P走LoadDataToL0B转置分支
        // (isRightTranspose=false -> ifTranspose=true, P在L1为(k x n)DN列主序, 与V的NZ加载同构)
        MMParam param = MakeMMParam(mm2VtMSize, mBaseSize, (realK + 63) >> 6 << 6, false, false, k == 0, k == 0);
        MatmulFullMX<KV_T, INPUT_T, T, mm2VtMSize, mBaseSize, mm2BaseK, ABLayout::MK, ABLayout::KN, L0AType, L0BType,
                     SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
            vtL1[kElemOffset], mm2P.GetTensor<INPUT_T>()[k * mm2BaseK * mBaseSize], mmL0ABuffers_, mmL0BBuffers_,
            mm2ResL0C.GetTensor<T>(), param, vtScaleL1[kScaleElemOffset], pScaleFakeTensor[k * pScaleKSegOffset]);
    }

    // MM2 DN方案fixpipe: L0C (d+32)x128 float -> UB, 保留完整m方向(尾带32行全等row_sum, V2取首行)
    __aicore__ inline void FixpipeMm2(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;   // L0C→UB
        fixpipeParams.mSize = mm2VtMSize;                        // d+32行全保留(尾带row_sum)
        fixpipeParams.nSize = mBaseSize;                         // n=128
        fixpipeParams.srcStride = ((mm2VtMSize + 15) >> 4 << 4); // L0C相邻Z排布偏移, 16对齐
        fixpipeParams.dstStride = mBaseSize;                     // UB行距=n
        fixpipeParams.dualDstCtl = 0;                            // 单目标: 输出完整(d+32)x128
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

}; // QuantFlashAttnBlockCubeMxfp8Fp16

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
class QuantFlashAttnBlockCubeMxfp8Fp16Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_PBUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_PBUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockCubeMxfp8Fp16Dummy(ConstInfoX &constInfo){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_FP16_H_
