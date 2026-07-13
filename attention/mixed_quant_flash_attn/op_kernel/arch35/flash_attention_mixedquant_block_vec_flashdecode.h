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
 * \file flash_attention_mixedquant_block_vec_flashdecode.h
 * \brief MixedQuantFlashAttn FlashDecode Vec Block定义
 */

#ifndef FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_
#define FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_

#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#include "../utils/attenmask_gs1.h"

template <inferFaLayOutTypeEnum LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_TND) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <inferFaLayOutTypeEnum LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_TND) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <inferFaLayOutTypeEnum LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BSND || LAYOUT == inferFaLayOutTypeEnum::LAYOUT_TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BNSD) {
        return UbFormat::GS1;
    }
}

template <inferFaLayOutTypeEnum LAYOUT>
__aicore__ inline uint64_t SeqLenFromTensorList(__gm__ uint8_t *keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4];
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT == inferFaLayOutTypeEnum::LAYOUT_BSND) {
        return keyTensorDesc.GetShape(1);
    } else {
        return keyTensorDesc.GetShape(2);
    }
}

namespace BaseApi {
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <typename INPUT_T, typename ACC_T, typename OUT_T, inferFaLayOutTypeEnum inputLayoutType,
          inferFaLayOutTypeEnum outputLayoutType, inferS1TemplateType s1TemplateType,
          inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, bool hasAttenMask, uint8_t KvLayoutType>
class FAMixedQuantGqaBlockVecFlashDecode {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using T = ACC_T;
    using OUTPUT_T = OUT_T;
    using SINK_T = INPUT_T;

    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;
    static constexpr inferFaLayOutTypeEnum layout = inputLayoutType;
    static constexpr inferFaLayOutTypeEnum outLayout = outputLayoutType;

    // static constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
    // static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

protected:
    GlobalTensor<float> lseSumFdGm;
    GlobalTensor<float> lseMaxFdGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<SINK_T> sinkGm;
    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();
    static constexpr bool isS1G = (UB_FORMAT == UbFormat::S1G);

    // =======================获取实际Act_S===========================
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, true>();
    // tensorlist
    __gm__ uint8_t *keyPtr = nullptr;

    using QSeqParserType =
        typename std::conditional<(layout == inferFaLayOutTypeEnum::LAYOUT_TND),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;

    using KvSeqParserType =
        typename std::conditional<(layout == inferFaLayOutTypeEnum::LAYOUT_TND),
                                  ActualSeqLensParser<KV_MODE, int32_t, true>,
                                  ActualSeqLensParser<KV_MODE, int32_t>>::type;

    QSeqParserType *qActSeqLensParser = nullptr;
    KvSeqParserType *kvActSeqLensParser = nullptr;

    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    bool learnableSinkFlag = false;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    // ================================类成员变量====================================
    const ConstInfoX &constInfo;
    TaskInfo taskInfo{};

private:
    LocalTensor<T> fdSumBuf1;
    LocalTensor<T> fdSumBuf2;
    LocalTensor<T> fdMaxBuf1;
    LocalTensor<T> fdMaxBuf2;
    LocalTensor<T> fdLseExpBuf;
    LocalTensor<T> fdMm2ResBuf1;
    LocalTensor<T> fdMm2ResBuf2;
    LocalTensor<T> fdReduceBuf;
    LocalTensor<OUTPUT_T> fdOutputBuf;
    LocalTensor<T> fdSinkCopyInBuf;
    LocalTensor<T> fdSinkValueBuf;
    LocalTensor<T> fdSinkExpBuf;
    LocalTensor<T> fdSinkTmpBuf;

    LocalTensor<T> fdLseMaxUbBuf1;
    LocalTensor<T> fdLseMaxUbBuf2;
    LocalTensor<T> fdLseUbBuf;
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 6;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 7;
    static constexpr uint64_t SYNC_SINK_BUF1_FLAG = 12;
    static constexpr uint64_t SYNC_SINK_BUF2_FLAG = 13;

    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align;
    static constexpr bool attenMaskFlag = hasAttenMask;

public:
    __aicore__ inline FAMixedQuantGqaBlockVecFlashDecode(ConstInfoX &constInfo) : constInfo(constInfo){};

    template <typename U> // 避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            __gm__ uint8_t *key)
    {
        this->lseMaxFdGm = lseMaxFdGm;
        this->lseSumFdGm = lseSumFdGm;
        this->accumOutGm = accumOutGm;
        this->attentionOutGm = attentionOutGm;
        this->keyPtr = key;
    }

    __aicore__ inline void SetCuSeqLensParsers(QSeqParserType &qParser, KvSeqParserType &kvParser)
    {
        this->qActSeqLensParser = &qParser;
        this->kvActSeqLensParser = &kvParser;
    }

    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink)
    {
        learnableSinkFlag = true;
        this->sinkGm = learnableSink;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm = softmaxLseGm;
    }

        __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = this->Align(constInfo.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }


    // __aicore__ inline void InitDecodeParams()
    // {
    //     this->blockIdx = GetBlockIdx();
    // }


    __aicore__ inline void InitBuffers(TPipe *pipe, TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[], TBuf<> &SharedBuffer3)
    {
        if ASCEND_IS_AIV {
            constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
            constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
            constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
            constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
            fdMm2ResBuf1 = SharedBuffer1[0].template AllocTensor<T>();
            fdMm2ResBuf2 = fdMm2ResBuf1[BUFFER_SIZE_BYTE_16K / sizeof(T)];
            fdReduceBuf = SharedBuffer1[1].template AllocTensor<T>();
            fdOutputBuf = fdReduceBuf[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<OUTPUT_T>();
            fdSumBuf1 = SharedBuffer3.template Get<T>();
            fdSumBuf2 = fdSumBuf1[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdMaxBuf1 = fdSumBuf2[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdMaxBuf2 = fdMaxBuf1[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdLseExpBuf = fdMaxBuf2[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdLseMaxUbBuf1 = SharedBuffer2[0].template AllocTensor<T>();
            fdLseMaxUbBuf2 = fdLseMaxUbBuf1[BUFFER_SIZE_BYTE_256B / sizeof(T)];
            fdLseUbBuf = fdLseMaxUbBuf2[BUFFER_SIZE_BYTE_256B / sizeof(T)];
            if (unlikely(learnableSinkFlag)) {
                fdSinkCopyInBuf = fdLseUbBuf[BUFFER_SIZE_BYTE_256B / sizeof(T)];
                fdSinkValueBuf = fdSinkCopyInBuf[BUFFER_SIZE_BYTE_2K / sizeof(T)];
                fdSinkExpBuf = fdSinkValueBuf[BUFFER_SIZE_BYTE_2K / sizeof(T)];
                fdSinkTmpBuf = fdSinkExpBuf[BUFFER_SIZE_BYTE_256B / sizeof(T)];
            }
        }
    }

    __aicore__ inline void FreeBuffers(TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[])
    {
        if ASCEND_IS_AIV {
            SharedBuffer1[0].template FreeTensor(fdMm2ResBuf1);
            SharedBuffer1[1].template FreeTensor(fdReduceBuf);
            SharedBuffer2[0].template FreeTensor(fdLseMaxUbBuf1);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        SetFlag<AscendC::HardEvent::V_MTE2>(1);
        SetFlag<AscendC::HardEvent::V_MTE2>(2);
        SetFlag<AscendC::HardEvent::V_MTE2>(3);
        SetFlag<AscendC::HardEvent::MTE3_V>(4);
        SetFlag<AscendC::HardEvent::MTE3_V>(5);
        SetFlag<AscendC::HardEvent::V_MTE2>(6);
        SetFlag<AscendC::HardEvent::V_MTE2>(7);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        WaitFlag<AscendC::HardEvent::V_MTE2>(2);
        WaitFlag<AscendC::HardEvent::V_MTE2>(3);
        WaitFlag<AscendC::HardEvent::MTE3_V>(4);
        WaitFlag<AscendC::HardEvent::MTE3_V>(5);
        WaitFlag<AscendC::HardEvent::V_MTE2>(6);
        WaitFlag<AscendC::HardEvent::V_MTE2>(7);
    }
protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align - constInfo.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align - constInfo.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo.dSizeV +                 // taskoffset + g轴offset
                                         splitKVIndex * s1BaseSize * constInfo.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1 : fdSumBuf2;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1 : fdMaxBuf2;

        uint64_t combineLseOffset = (baseOffset + startRow) * FP32_BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = s1BaseSize * FP32_BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * FP32_BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1 : fdSumBuf2;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1 : fdMaxBuf2;
        if (unlikely(learnableSinkFlag)) {
            SinkMax(startRow, dealRowCount);
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1 : fdLseMaxUbBuf2;

        LocalTensor<T> sinkExpBuf = fdSinkExpBuf;
        LocalTensor<T> maxLseUb = fdLseUbBuf;
        ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                actualCombineLoopSize, constInfo.isSoftmaxLseEnable, learnableSinkFlag);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoord gmCoord{.bIdx = taskInfo.bIdx,
                        .n2Idx = taskInfo.n2Idx,
                        .gS1Idx = taskInfo.gS1Idx + startRow,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};

        if constexpr (outLayout == inferFaLayOutTypeEnum::LAYOUT_BSND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == inferFaLayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == inferFaLayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align;
        FaVectorApi::ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf;
        AscendC::PipeBarrier<PIPE_V>();
        DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align);
        DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * this->dSizeV_Align);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                 dealRowCount * this->dSizeV_Align);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
        if (constInfo.seqUsedKvSize == 0 && !constInfo.isKvContinuous) {
            actSeqLensKv = SeqLenFromTensorList<layout>(keyPtr, taskInfo.bIdx);
        } else {
            actSeqLensKv = kvActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
        }
        int64_t safePreToken = constInfo.preTokens;
        int64_t safeNextToken = constInfo.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken, constInfo.sparseMode);

        if (constInfo.sparseMode == BAND) {
            preTokensPerBatch = safePreToken;
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
        } else if ((constInfo.sparseMode == DEFAULT_MASK) && attenMaskFlag) {
            nextTokensPerBatch = safeNextToken;
            preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
        } else {
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
            preTokensPerBatch = 0;
        }
    }

    __aicore__ inline void CopySinkIn(uint32_t cntM)
    {

    }
    __aicore__ inline void SinkMax(uint32_t startRow, uint32_t dealRowCount)
    {
        constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
        int64_t gIdx = 0;
        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf;
        LocalTensor<T> sinkExpBuf = fdSinkExpBuf;

        for (int64_t row = 0; row < dealRowCount; ++row) {
            if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { // 内存按照S1G排布
                gIdx = (taskInfo.gS1Idx + startRow + row) % constInfo.gSize;
            } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { // 内存按照GS1排布
                int64_t actS1Size = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
                gIdx = (taskInfo.gS1Idx + startRow + row) / actS1Size;
            }
            DataCopy(sinkExpBuf[row * BLOCK_ELEMENT_NUM], sinkBrcbBuf[gIdx * BLOCK_ELEMENT_NUM], BLOCK_ELEMENT_NUM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if (!attenMaskFlag) {
            return;
        }

        if (constInfo.sparseMode == ALL_MASK || constInfo.sparseMode == LEFT_UP_CAUSAL) {
            return;
        }

        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ,
            .gSize = static_cast<uint64_t>(constInfo.gSize),
            .gS1Idx = taskInfo.gS1Idx + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = preTokensPerBatch,
            .nextTokensPerBatch = nextTokensPerBatch,
        };

        fa_base_vector::InvalidRows<UBOUT_T, fa_base_vector::GeInputUbFormat<layout>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if (!attenMaskFlag) {
            return;
        }
        if (constInfo.sparseMode != DEFAULT_MASK && constInfo.sparseMode != ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1 : fdLseMaxUbBuf2;

        fa_base_vector::InvalidMaskRows<UBOUT_T, T, true>(0, dealRowCount, columnCount, lseMaxUb, negativeIntScalar,
                                                          attenOutUb);
    }

public:
    __aicore__ inline void FlashDecode(FDparamsX &fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        uint32_t fdBalanceMBaseSize = 8U;
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo.bIdx = fd.fdBN2Idx / constInfo.n2Size;
        taskInfo.n2Idx = fd.fdBN2Idx % constInfo.n2Size;
        taskInfo.gS1Idx = fd.fdMIdx * s1BaseSize;
        taskInfo.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * s1BaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf;
            LocalTensor<T> reduceOut = fdReduceBuf;
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            if (unlikely(learnableSinkFlag)) {
                CopySinkIn(reduceMLoop);
            }
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; preLoadIdx++) {
                LocalTensor<T> mm2Res = (((reduceGlobaLoop + preLoadIdx) & 1) == 0) ? fdMm2ResBuf1 : fdMm2ResBuf2;
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);
            CalcPreNextTokens();
            if (constInfo.isSoftmaxLseEnable) {
                LocalTensor<T> maxLseUb = fdLseUbBuf;
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo.gS1Idx + startRow;
                if constexpr (layout == inferFaLayOutTypeEnum::LAYOUT_TND) {
                    uint32_t prefixBS1 = qActSeqLensParser->GetTBase(taskInfo.bIdx);
                    uint64_t bN2Offset = taskInfo.n2Idx * constInfo.gSize * constInfo.t1Size + prefixBS1;
                    DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                   actualGSplitSize, constInfo);
                } else if constexpr (layout == inferFaLayOutTypeEnum::LAYOUT_BSND) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo);
                } else { // BNSD
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo, qActSeqLens);
                }
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1 : fdMm2ResBuf2;
                if (i >= preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_
