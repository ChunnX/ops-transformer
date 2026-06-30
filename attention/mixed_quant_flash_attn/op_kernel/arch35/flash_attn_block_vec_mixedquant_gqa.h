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
 * \file flash_attn_block_vec_mixedquant_gqa.h
 * \brief MixedQuantFlashAttn Vec Block定义
 */

#ifndef FLASH_ATTN_BLOCK_VEC_MIXEDQUANT_GQA_H_
#define FLASH_ATTN_BLOCK_VEC_MIXEDQUANT_GQA_H_

#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"

namespace BaseApi {

template <typename INPUT_T, typename ACC_T, typename OUT_T, inferFaLayOutTypeEnum inputLayoutType,
          inferFaLayOutTypeEnum outputLayoutType, inferS1TemplateType s1TemplateType,
          inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, bool hasAttenMask, uint8_t KvLayoutType>
class MixedQuantGqaVecBlockBase {
public:
    using ConstInfoX = MixedQuantConstInfoX;

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;

    static constexpr bool HAS_MASK = hasAttenMask;

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);

    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;

    __aicore__ inline MixedQuantGqaVecBlockBase(ConstInfoX &constInfo) : constInfo(constInfo){};

    const ConstInfoX &constInfo;
};

template <typename INPUT_T, typename ACC_T, typename OUT_T, inferFaLayOutTypeEnum inputLayoutType,
          inferFaLayOutTypeEnum outputLayoutType, inferS1TemplateType s1TemplateType,
          inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, bool hasAttenMask, uint8_t KvLayoutType>
class FAMixedQuantGqaBlockVec
    : public MixedQuantGqaVecBlockBase<INPUT_T, ACC_T, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                       s2TemplateType, dTemplateType, hasAttenMask, KvLayoutType> {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using INPUT_T2 = INPUT_T;
    using T = ACC_T;
    using OUTPUT_T = OUT_T;

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;
    static constexpr bool HAS_MASK = hasAttenMask;

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);
    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;
    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;

    static constexpr uint32_t initOutputEventId = 0U;

    TPipe *tPipe;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<float> softmaxFDSumGm;
    GlobalTensor<float> softmaxFDMaxGm;
    GlobalTensor<int32_t> actualSeqLengthsGmQ;

    QSeqParserType *qActSeqLensParser = nullptr;

    TEventID mte3ToVId[2];
    TEventID vToMte3Id[2];

    bool isFd = false;

    const ConstInfoX &constInfo;
    T negativeFloatScalar;

    using attenMaskGmType = typename std::conditional<hasAttenMask, GlobalTensor<uint8_t>, int8_t>::type;
    attenMaskGmType attenMaskGmInt;
    GlobalTensor<INPUT_T> sinkGm;

    __aicore__ inline FAMixedQuantGqaBlockVec(ConstInfoX &constInfo)
        : MixedQuantGqaVecBlockBase<INPUT_T, ACC_T, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                    s2TemplateType, dTemplateType, hasAttenMask, KvLayoutType>(constInfo),
          constInfo(constInfo){};

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType &qParser)
    {
        this->qActSeqLensParser = &qParser;
    }

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        tPipe = pipe;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, attenMask, learnableSink, softmaxLse, attentionOut,
                     workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        uint64_t actualLenQSize =
            (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) ? constInfo.cuSeqLensQSize : constInfo.seqUsedQSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, actualLenQSize);

        if constexpr (hasAttenMask) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if (learnableSink != nullptr) {
            sinkGm.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
        }

        if (this->isFd) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            InitOutputSingleCore();
            if (constInfo.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (MQFA_CV_RATIO * constInfo.coreNum) - 1) / (MQFA_CV_RATIO * constInfo.coreNum);
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize;
        int64_t singleCoreSize =
            (totalOutputSize + (MQFA_CV_RATIO * constInfo.coreNum) - 1) / (MQFA_CV_RATIO * constInfo.coreNum);
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 3e+99);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitBuffers(TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[], TBuf<> &SharedBuffer3)
    {
        // TODO: 待后续填充实现
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
    }
};

} // namespace BaseApi

#endif // FLASH_ATTN_BLOCK_VEC_MIXEDQUANT_GQA_H_
