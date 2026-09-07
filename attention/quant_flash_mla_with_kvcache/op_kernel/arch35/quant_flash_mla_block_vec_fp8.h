/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_block_vec_fp8.h
 * \brief QuantFlashMlaWithKvcache FA阶段vec block（自FIA MLA block_vec裁剪适配）
 */

#ifndef QUANT_FLASH_MLA_BLOCK_VEC_FP8_H_
#define QUANT_FLASH_MLA_BLOCK_VEC_FP8_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/vector_common.h"
#include "memory_copy_arch35_quant_flash_mla_with_kvcache.h"
#include "quant_flash_mla_with_kvcache_public_def.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, bool hasAtten = false,
          uint8_t KvLayoutType = 0, bool isFd = false, bool bmm2Write2Ub = true>
class QuantFlashMlaBlockVecFp8 {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2

    static constexpr uint32_t initOutputEventId = 0U;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND ||
         layout == LayOutTypeEnum::LAYOUT_SBH) ?
            MaskFormat::SG :
            MaskFormat::GS;

    static constexpr bool USE_DN = false;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool IS_PER_TOKEN_HEAD = true;
    // MLA全量化: Q per-token-head descale, TND下[T, N1]排布(TNG)
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN, IS_PER_TOKEN_HEAD, true>();

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using ConstInfoX = QmlaConstInfo;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // Q: TND, cu_seq(含前导0)+seq_used, int32
    using QSeqParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>;

    // gm
    TPipe *tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    QSeqParserType *qSeqParserPtr_ = nullptr;
    GlobalTensor<uint8_t> attenMaskGmInt_;
    GlobalTensor<float> deScaleQGm_;
    GlobalTensor<float> deScaleKGm_;
    GlobalTensor<float> deScaleVGm_;
    FaGmTensor<float, Q_SCALE_FORMAT, int32_t, true> queryScaleGm_;
    CopyQueryScaleGmToUb<float, Q_SCALE_FORMAT> copyQueryScaleGmToUb_;
    flashdecodeGmType accumOutGm_;
    flashdecodeGmType softmaxFDSumGm_;
    flashdecodeGmType softmaxFDMaxGm_;

    // ub
    TBuf<> commonTBuf_;
    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue_[2];
    TBuf<> stage2OutBuf_;
    TEventID mte3ToVId_[2];
    TEventID vToMte3Id_[2];
    TEventID mte2ToVId_[2];
    TEventID vToMte2Id_[2];
    TBuf<> softmaxMaxBuf_[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf_[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf_[PRELOAD_N + 1];
    TBuf<> preLoopMaxBuf_;
    TBuf<> preLoopSumBuf_;
    TBuf<> firstLoopSumBuf_;
    TBuf<> vselrIndexesBuf_[4];
    TBuf<> lseTmpBuff_;
    TBuf<> queryAntiqScaleInputQue_[2];
    TBuf<> pScaleBuf_[3];
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue_;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域, FLASH_DECODE 写 softmaxFDSumGm_/softmaxFDMaxGm_ */
    TQue<QuePosition::VECOUT, 1> maxBrdcst_;
    TQue<QuePosition::VECOUT, 1> sumBrdcst_;

    const ConstInfoX &constInfo_;
    T negativeFloatScalar_;
    float deScaleKValue_{1.0f};
    float deScaleVValue_{1.0f};
    uint32_t minValue_{NEGATIVE_MIN_VALUE_FP32};

    __aicore__ inline QuantFlashMlaBlockVecFp8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *dequantScaleQuery,
                                        __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace, QSeqParserType &qParser)
    {
        tPipe_ = pipe;
        qSeqParserPtr_ = &qParser;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        InitVecInput(dequantScaleQuery, dequantScaleKey, dequantScaleValue, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                        __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        // MLA全量化dequantScale: Q per-token-head, KV per-tensor
        if (dequantScaleQuery != nullptr) {
            deScaleQGm_.SetGlobalBuffer((__gm__ float *)dequantScaleQuery);
            InitQScaleBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, 1,
                             queryScaleGm_, dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            deScaleKGm_.SetGlobalBuffer((__gm__ float *)dequantScaleKey);
            deScaleKValue_ = this->deScaleKGm_.GetValue(0);
        }
        if (dequantScaleValue != nullptr) {
            deScaleVGm_.SetGlobalBuffer((__gm__ float *)dequantScaleValue);
            deScaleVValue_ = this->deScaleVGm_.GetValue(0);
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, FaGmTensor<float, Q_SCALE_FORMAT, int32_t, true> &qScaleGmTensor,
                                            __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_TN) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, *qSeqParserPtr_);
        }
    }

    // MLA q_descale: 复用common层 CopyQueryScaleGmToUb
    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo)
    {
        FaUbTensor<float> ubTensor{
            .tensor = dstTensor,
            .rowCount = runInfo.actVecMSize,
            .colCount = dRealSize,
        };

        GmCoord gmCoord{
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actVecMSize,
        };
        copyQueryScaleGmToUb_(ubTensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, 1, runInfo);
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoX runInfo)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            outputBuf.SetCrossCore();
            return;
        }

        // MLA全量化走ND路径
        ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo_.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qSeqParserPtr_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum);
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(attentionOutGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qSeqParserPtr_->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo_.coreNum) - 1) / (2 * constInfo_.coreNum);
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 3e+99);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        tPipe_->InitBuffer(softmaxSumBuf_[0], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxSumBuf_[1], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxSumBuf_[2], 256); // [64, 1]

        tPipe_->InitBuffer(softmaxMaxBuf_[0], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxMaxBuf_[1], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxMaxBuf_[2], 256); // [64, 1]

        tPipe_->InitBuffer(softmaxExpBuf_[0], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxExpBuf_[1], 256); // [64, 1]
        tPipe_->InitBuffer(softmaxExpBuf_[2], 256); // [64, 1]

        if constexpr (FLASH_DECODE) {
            tPipe_->InitBuffer(maxBrdcst_, 1, 2048); // [64, 8]
            tPipe_->InitBuffer(sumBrdcst_, 1, 2048); // [64, 8]
        }
    }

    __aicore__ inline void InitBuffers()
    {
        SoftmaxInitBuffer();
        tPipe_->InitBuffer(preLoopMaxBuf_, 256);
        tPipe_->InitBuffer(preLoopSumBuf_, 256);
        tPipe_->InitBuffer(firstLoopSumBuf_, 256);
        tPipe_->InitBuffer(stage2OutBuf_, 32 * dTemplateAlign64 * sizeof(T));
        // (mBaseSize / CV_RATIO + 1) * s2BaseSize * sizeof(INPUT_T), fp8占1字节
        tPipe_->InitBuffer(stage1OutQue_[0], 1, 4224);
        tPipe_->InitBuffer(stage1OutQue_[1], 1, 4224);
        tPipe_->InitBuffer(commonTBuf_, 512);

        if constexpr (HAS_MASK) {
            // GS1方向需要循环处理，一个vector计算softmax的数据量最大为32*128，对应mask的数据量为4096Bytes
            tPipe_->InitBuffer(attenMaskInQue_[0], 1, 4096);
            tPipe_->InitBuffer(attenMaskInQue_[1], 1, 4096);
        }

        // MLA全量化: queryAntiqScaleInputQue(per-token Q scale)和pScaleBuf
        constexpr uint32_t softmaxRowmaxBufSize = 256; // s1 baseSize * 4b(fp32)
        tPipe_->InitBuffer(queryAntiqScaleInputQue_[0], (mBaseSize / CV_RATIO) * sizeof(float));
        tPipe_->InitBuffer(queryAntiqScaleInputQue_[1], (mBaseSize / CV_RATIO) * sizeof(float));
        tPipe_->InitBuffer(pScaleBuf_[0], softmaxRowmaxBufSize);
        tPipe_->InitBuffer(pScaleBuf_[1], softmaxRowmaxBufSize);
        tPipe_->InitBuffer(pScaleBuf_[2], softmaxRowmaxBufSize);

        if (constInfo_.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            this->tPipe_->InitBuffer(softmaxLseQueue_, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
        }
        // ND
        tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)],
                           128); // s2realsize (64, 128]
        tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)],
                           64); // s2realsize (0, 64]

        LocalTensor<uint8_t> vselrIndexesTensor =
            vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        vselrIndexesTensor.SetValue(0, 0x7f);
        for (int i = 0; i < 128; i++) {
            vselrIndexesTensor.SetValue(i, i << 1);
        }
        vselrIndexesTensor =
            vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        for (int i = 0; i < 64; i++) {
            vselrIndexesTensor.SetValue(i, i << 2);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        mte2ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        mte2ToVId_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        vToMte2Id_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(vToMte2Id_[0]);
        SetFlag<HardEvent::V_MTE2>(vToMte2Id_[1]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[1]);
        WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Id_[0]);
        WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Id_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVId_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Id_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Id_[1]);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX &runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize;

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.realGSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSizeCur;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = 0; // 新接口无batch维度mask张量, causal在UB内动态构造
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSizeCur - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool isSkipMask = IsSkipAttentionmask(maskInfo);
        if (unlikely(!isSkipMask)) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSizeCur>(attenMaskUb, attenMaskGmInt_, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
        }
    }

    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;

        if constexpr (HAS_MASK) {
            attenMaskUb = this->attenMaskInQue_[0].template AllocTensor<uint8_t>();
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
        LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf_.template Get<uint8_t>();

        int64_t stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0f;

        LocalTensor<float> queryScaleUb;
        LocalTensor<T> pScaleUb;
        // MLA全量化: 首个S2 loop加载per-token Q scale, 取pScale buffer
        if (unlikely(runInfo.isFirstS2Loop)) {
            queryScaleUb = queryAntiqScaleInputQue_[runInfo.mloop % DB].template Get<float>();
            WaitFlag<HardEvent::V_MTE2>(vToMte2Id_[runInfo.mloop % DB]);
            CopyQueryScaleTile(queryScaleUb, runInfo);
            SetFlag<HardEvent::MTE2_V>(mte2ToVId_[runInfo.mloop % DB]);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVId_[runInfo.mloop % DB]);
        }
        pScaleUb = pScaleBuf_[runInfo.loop % 3].template Get<T>();

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();

        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        // MLA s2BaseSize=128固定, 仅需EQ_128/GT_0_AND_LTE_64/GT_64_AND_LTE_128三段
        // ProcessVec1Vf模板参数: <T, T2, pseShiftType, isUpdate, s1BaseSize, s2BaseSize, oriNRange,
        //                          hasAtten, pseMode, hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>
        constexpr bool isMlaSgdTrue = true;
        constexpr bool isMlaSgdFalse = false;
        if (unlikely(runInfo.isFirstS2Loop)) {
            // 首 loop: isUpdate=false
            if (likely(s2CalcSize == s2BaseSize)) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                                           PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdFalse, true, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                    posShift, static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 1.0F,
                    queryScaleUb, deScaleKValue_);
            } else if (s2CalcSize <= 64) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                                           HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdFalse, true, false,
                                           false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                    posShift, static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 1.0F,
                    queryScaleUb, deScaleKValue_);
            } else {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                                           HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdFalse, true, false,
                                           false>(
                    stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                    posShift, static_cast<T>(constInfo_.scaleValue), descaleQK, negativeFloatScalar_, 1.0F,
                    queryScaleUb, deScaleKValue_);
            }
        } else {
            // 非首 loop: isUpdate=true
            bool isMlaSgdRuntime = (constInfo_.gSize >= 32) && (constInfo_.gSize % 32 == 0);
            if (likely(s2CalcSize == s2BaseSize)) {
                if (isMlaSgdRuntime) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                                               PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdTrue, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo_.scaleValue), descaleQK,
                        negativeFloatScalar_, 1.0F, queryScaleUb, deScaleKValue_);
                } else {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                                               PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdFalse, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo_.scaleValue), descaleQK,
                        negativeFloatScalar_, 1.0F, queryScaleUb, deScaleKValue_);
                }
            } else if (s2CalcSize <= 64) {
                if (isMlaSgdRuntime) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                                               HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdTrue, true, false,
                                               false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo_.scaleValue), descaleQK,
                        negativeFloatScalar_, 1.0F, queryScaleUb, deScaleKValue_);
                } else {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                                               HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdFalse, true, false,
                                               false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo_.scaleValue), descaleQK,
                        negativeFloatScalar_, 1.0F, queryScaleUb, deScaleKValue_);
                }
            } else {
                if (isMlaSgdRuntime) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                                               HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdTrue, true, false,
                                               false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo_.scaleValue), descaleQK,
                        negativeFloatScalar_, 1.0F, queryScaleUb, deScaleKValue_);
                } else {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                                               HAS_MASK, PseTypeEnum::PSE_NONE_TYPE, false, isMlaSgdFalse, true, false,
                                               false>(
                        stage1CastTensor, this->vselrIndexesBuf_, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                        attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize,
                        pseStride, slopes, posShift, static_cast<T>(constInfo_.scaleValue), descaleQK,
                        negativeFloatScalar_, 1.0F, queryScaleUb, deScaleKValue_);
                }
            }
        }
        bmm1ResBuf.SetCrossCore();

        if constexpr (HAS_MASK) {
            this->attenMaskInQue_[0].template FreeTensor(attenMaskUb);
        }

        // ===================DataCopy to L1 ====================
        this->stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = constInfo_.subBlockIdx * (mBaseSize * 16);
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
                     {s2BaseSize / 32, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        this->stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        outputBuf.SetCrossCore();
        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::V_MTE2>(vToMte2Id_[runInfo.mloop % DB]);
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
            if (!runInfo.isS2SplitCore && constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMSize = runInfo.actVecMSize;
        uint32_t gmDealRowCount = runInfo.actVecMSize;

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseQueue_.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue_);
        softmaxLseQueue_.template EnQue(lseUb);
        softmaxLseQueue_.DeQue<float>();

        // 新接口LSE输出统一为NT格式[N2*G, T]
        uint32_t prefixBS1 = qSeqParserPtr_->GetTBase(runInfo.bIdx);
        uint64_t bN2Offset = runInfo.realN2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
        DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                                 gmDealRowCount, constInfo_);
        softmaxLseQueue_.FreeTensor(lseUb);
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst_.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst_.template EnQue(sumOutTensor);
        sumBrdcst_.template DeQue<float>();
        DataCopy(softmaxFDSumGm_[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst_.template FreeTensor(sumOutTensor);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst_.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst_.template EnQue(maxOutTensor);
        maxBrdcst_.template DeQue<float>();
        DataCopy(softmaxFDMaxGm_[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst_.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        // MLA全量化vec2
        bmm2ResBuf.WaitCrossCore();
        if constexpr (bmm2Write2Ub) {
            ProcessVec2OnUb(bmm2ResBuf, runInfo);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }
        uint32_t vecMSize = runInfo.actVecMSize;
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf_.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;
            pScaleUb = pScaleBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            float deSCalePreVValue = this->deScaleVValue_;
            if (likely(!runInfo.isLastS2Loop)) {
                if (unlikely(runInfo.s2LocalIdx == s2BaseSize)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, true>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, deScaleVValue_,
                        deSCalePreVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, true>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, deScaleVValue_,
                        deSCalePreVValue);
                }
            } else {
                if (unlikely(runInfo.s2LocalIdx == s2BaseSize)) {
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, true>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, deScaleVValue_,
                        deSCalePreVValue);
                } else {
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, true>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, deScaleVValue_,
                        deSCalePreVValue);
                }
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(vec2ResUb, vec2ResUb, sumUb, vecMSize,
                                                                         (uint16_t)dTemplateAlign64, deScaleVValue_);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = (constInfo_.sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow;
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo_.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                    int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            s1StartTdx = vecMStartIdx / constInfo_.realGSize;
            s1EndTdx = vecMEndIdx / constInfo_.realGSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[runInfo.loop % DB]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[runInfo.loop % DB]);
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm_[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
                                      .rowCount = dealRowCount,
                                      .colCount = (uint32_t)dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.realN2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qSeqParserPtr_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qSeqParserPtr_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else { // BNSD
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
};

// AIC编译单元使用的Vec空壳: kernel在AIV分支外不会实例化Vec接口, 仅需类型/常量成员
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512, bool hasAtten = false,
          uint8_t KvLayoutType = 0, bool isFd = false, bool bmm2Write2Ub = true>
class QuantFlashMlaBlockVecFp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = QmlaConstInfo;
    __aicore__ inline QuantFlashMlaBlockVecFp8Dummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_MLA_BLOCK_VEC_FP8_H_
