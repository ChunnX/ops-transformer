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
 * \file quant_flash_mla_block_cube_fp8.h
 * \brief QuantFlashMlaWithKvcache cube block（自FIA MLA block_cube裁剪适配, rope方案A: 576维含rope布局）
 */

#ifndef QUANT_FLASH_MLA_BLOCK_CUBE_FP8_H_
#define QUANT_FLASH_MLA_BLOCK_CUBE_FP8_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "memory_copy_arch35_quant_flash_mla_with_kvcache.h"

#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "quant_flash_mla_with_kvcache_public_def.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

/* ============确定Query/Key的L1类型============= */
template <typename INPUT_T, uint32_t dBaseSize>
struct QmlaQL1BuffSel {
    using Type =
        std::conditional_t<std::is_same_v<INPUT_T, float> ||
                               (!(std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                  std::is_same_v<INPUT_T, int8_t> || std::is_same_v<INPUT_T, hifloat8_t>) &&
                                dBaseSize > 256),
                           BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>;
};

template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct QmlaKVL1BuffSel {
    constexpr static bool isFP8DType = std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                       std::is_same_v<INPUT_T, hifloat8_t>;
    constexpr static bool isINT8DType = std::is_same_v<INPUT_T, int8_t>;
    using Type = std::conditional_t<
        ((isFP8DType || isINT8DType) && s2BaseSize == 128 && dBaseSize == 576), BuffersPolicy4buff<BufferType::L1>,
        std::conditional_t<(!(isFP8DType || isINT8DType) && s2BaseSize == 256 && dBaseSize > 128),
                           BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>>;
};

template <typename INPUT_T>
struct QmlaL0ABuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>, BuffersPolicySingleBuffer<BufferType::L0A>,
                                    BuffersPolicyDB<BufferType::L0A>>;
};

template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct QmlaL0BBuffSel {
    using Type =
        std::conditional_t<std::is_same_v<INPUT_T, float> ||
                               (s2BaseSize == 256 && dBaseSize > 128 &&
                                !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
                                  std::is_same_v<INPUT_T, int8_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
                           BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
};

template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct QmlaL0CBuffSel {
    using Type = std::conditional_t<(s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                     s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                    BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
};

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, uint8_t KvLayoutType = 0,
          bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class QuantFlashMlaBlockCubeFp8 {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = true; // MLA固定PA场景
    // rope方案A: q/k_cache的D维=576(nope 512 + rope 64)拼接, bmm1单次576维matmul
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;

    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr TPosition bmm2OutPos =
        GetC2Position(dVTemplateType,
                      UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, PseTypeEnum::PSE_NONE_TYPE, false,
                                              false, false, mBaseSize == 64),
                      (s2BaseSize == 256 && mBaseSize == 64), true);
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, BMM2_TOUB};

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    // 新接口KvLayoutType编码: 0=PA_BBND 1=PA_BNBD 2=PA_NZ
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MLA_FULLQUANT_MM2_T = std::conditional_t<isInt8, int32_t, T>;
    using mm2ResPos = typename std::conditional<BMM2_TOUB, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = mm2ResPos;

    // Q: TND, cu_seq(含前导0)+seq_used, int32
    using QSeqParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>;
    // KV: PA, cache_seqlens按batch, int32
    using KvSeqParserType = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;

    static constexpr bool Q_NEEDS_WZH = true;
    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t>;

    using ConstInfoX = QmlaConstInfo;
    TPipe *tPipe_ = nullptr;

    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<float> deScaleQGm_;
    GlobalTensor<float> deScaleKGm_;
    GlobalTensor<float> deScaleVGm_;

    QSeqParserType *qSeqParserPtr_ = nullptr;
    KvSeqParserType *kvSeqParserPtr_ = nullptr;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;

    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L1> *l1BufferManagerPtr_;
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;

    typename QmlaQL1BuffSel<INPUT_T, dBaseSize>::Type l1QBuffers_;
    typename QmlaKVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type l1KBuffers_;

    typename QmlaL0ABuffSel<INPUT_T>::Type mmL0ABuffers_;
    typename QmlaL0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type mmL0BBuffers_;
    typename QmlaL0CBuffSel<INPUT_T, mBaseSize, s2BaseSize, dVBaseSize>::Type mmL0CBuffers_;

    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    const ConstInfoX &constInfo_;

    __aicore__ inline QuantFlashMlaBlockCubeFp8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                         __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                         __gm__ uint8_t *dequantScaleValue, QSeqParserType &qParser,
                                         KvSeqParserType &kvParser)
    {
        tPipe_ = pipe;
        l1BufferManagerPtr_ = l1BuffMgr;
        qSeqParserPtr_ = &qParser;
        kvSeqParserPtr_ = &kvParser;
        InitCubeInput(query, key, value, blockTable, dequantScaleQuery, dequantScaleKey, dequantScaleValue);
    }

    __aicore__ inline void InitBuffers()
    {
        // rope方案A: Q/K的L1均按完整D=576(nope+rope拼接)分配
        constexpr uint32_t mm1QSize = mBaseSize * dBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1KSize = dBaseSize * s2BaseSize * sizeof(INPUT_T);

        l1QBuffers_.Init((*l1BufferManagerPtr_), mm1QSize);
        l1KBuffers_.Init((*l1BufferManagerPtr_), mm1KSize);

        // L0A B C 当前写死
        l0aBufferManager_.Init(tPipe_, 65536);  // 64 * 1024
        l0bBufferManager_.Init(tPipe_, 65536);  // 64 * 1024
        l0cBufferManager_.Init(tPipe_, 262144); // 256 * 1024
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
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);

        // rope方案A: q的D维=576(nope 512+rope 64拼接), 整维拷贝
        InitQBuffer(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSize, queryGm_, query);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize, keyGm_, key,
                     constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);
        InitKVBuffer(constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize, valueGm_, value,
                     constInfo_.valueStrides.bnStride, constInfo_.valueStrides.n2Stride);

        // MLA全量化dequantScale: Q per-token-head, KV per-tensor
        if (dequantScaleQuery != nullptr) {
            deScaleQGm_.SetGlobalBuffer((__gm__ float *)dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            deScaleKGm_.SetGlobalBuffer((__gm__ float *)dequantScaleKey);
        }
        if (dequantScaleValue != nullptr) {
            deScaleVGm_.SetGlobalBuffer((__gm__ float *)dequantScaleValue);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t n2Size, uint32_t gSize, uint32_t headDim, FaGmTensorQ &qGmTensor,
                                       __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr_);
    }

    __aicore__ inline void InitKVBuffer(uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                         FaGmTensorKV &kvGmTensor, __gm__ uint8_t *gm, uint64_t bnStrides,
                                         uint64_t n2Strides)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else { // GM_KV_PA_BBND
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
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
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
        mmL0CBuffers_.Uninit(l0cBufferManager_);
    }

    // rope方案A: 整维拷贝q(含rope, D=576)到L1 NZ布局
    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoord gmCoord{.bIdx = runInfo.bIdx,
                        .n2Idx = runInfo.realN2Idx,
                        .gS1Idx = runInfo.gS1Idx,
                        .dIdx = 0,
                        .gS1DealSize = runInfo.actMSize,
                        .dDealSize = constInfo_.dSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    // rope方案A: 整维拷贝k(含rope, D=576)到L1 NZ布局
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo, uint32_t s2RealSize)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0,
                          .s2DealSize = s2RealSize,
                          .dDealSize = constInfo_.dSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    // MLA中V = K_nope, bmm2复用K的L1 buffer, 无需独立搬运V

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        // MLA全量化bmm1(rope方案A): Q[64,576] @ K[576,128]单次matmul
        // K维576=2*256+64, MatmulK内部按256分块并处理尾块
        // Q(含rope)首次全载L1，后续s2循环复用
        Buffer<BufferType::L1> mm1A;

        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A = l1QBuffers_.Get();
            mm1A.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
            CopyQueryTile(mm1ATensor, runInfo);
            mm1A.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1A = l1QBuffers_.GetPre();
            mm1A.Set<HardEvent::MTE2_MTE1>();
        }

        // 加载当前轮的K(含rope)到L1
        Buffer<BufferType::L1> mm1B = l1KBuffers_.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();
        uint32_t s2CurSize = runInfo.actSingleLoopS2Size;
        CopyKeyTile(mm1BTensor, runInfo, s2CurSize);
        mm1B.Set<HardEvent::MTE2_MTE1>();

        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers_.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        // MatMul: Q[64,576] @ K[576,128], K维含rope(方案A单次matmul)
        MMParam param =
            MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)s2CurSize, (uint32_t)constInfo_.dSize, false, true);
        MatmulK<INPUT_T, INPUT_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
            mm1ResL0C.GetTensor<T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1B.Set<HardEvent::MTE1_MTE2>();
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();

        FixpipeMm1(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo, s2CurSize);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                      uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (s2RealSize + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((runInfo.actMSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        // MLA全量化bmm2: P @ V_nope, V复用K的nope部分(MLA中V=K_nope)
        MM2_ABUF_T mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();

        if constexpr (BMM2_TOUB) {
            outputBuf.WaitCrossCore();
        }

        Buffer<BufferType::L1> mm2B = l1KBuffers_.GetReused();
        // V读取需与bmm1的K buffer轮转握手：消费bmm1(T)入队的释放标志，
        // 并在V读取完成后释放该buffer，防止bmm1(T+4)的MTE2在MTE1读V未完成时覆写
        mm2B.Wait<HardEvent::MTE1_MTE2>();
        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers_.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo_.dSizeV,
                                    (uint32_t)runInfo.actSingleLoopS2Size, false, false);
        MatmulN<INPUT_T, INPUT_T, MLA_FULLQUANT_MM2_T, 128, 256, 128, ABLayout::MK, ABLayout::KN>(
            mm2A.GetTensor<INPUT_T>(), mm2B.GetTensor<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
            mm2ResL0C.GetTensor<MLA_FULLQUANT_MM2_T>(), param);
        mm2B.Set<HardEvent::MTE1_MTE2>();

        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();

        FixpipeMm2(outputBuf.template GetTensor<MLA_FULLQUANT_MM2_T>(),
                  mm2ResL0C.GetTensor<MLA_FULLQUANT_MM2_T>(), runInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T &dstTensor, const LocalTensor<MLA_FULLQUANT_MM2_T> &l0C,
                                      RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        if constexpr (BMM2_TOUB) {
            fixpipeParams.nSize = ((uint32_t)constInfo_.dSizeV + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = constInfo_.dSizeV;
        }
        fixpipeParams.mSize = mBaseSize;
        fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        if constexpr (BMM2_TOUB) {
            fixpipeParams.dstStride = ((uint32_t)dVBaseSize + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo_.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MLA_FULLQUANT_MM2_T, MLA_FULLQUANT_MM2_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }
};

// AIV编译单元使用的Cube空壳: kernel在AIC分支外不会实例化Cube接口, 仅需类型/常量成员
template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, uint8_t KvLayoutType = 0,
          bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class QuantFlashMlaBlockCubeFp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = true;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;

    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using MLA_FULLQUANT_MM2_T = std::conditional_t<isInt8, int32_t, T>;
    using mm2ResPos = typename std::conditional<BMM2_TOUB, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using ConstInfoX = QmlaConstInfo;
    __aicore__ inline QuantFlashMlaBlockCubeFp8Dummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_MLA_BLOCK_CUBE_FP8_H_
