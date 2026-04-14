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
 * \file cast_softmax_grad.h
 */
#ifndef CAST_SOFTMAX_GRAD_
#define CAST_SOFTMAX_GRAD_
#include "../flash_attention_score_grad_common.h"
#include "./vf_softmax_grad_front_cast.h"
#include "./vf_anti_quant_softmax_grad_front_cast.h"

using namespace commondef;

TEMPLATES_DEF
__aicore__ inline void CopyInSoftmaxGrad(FagConstInfo &constInfo, FagRunInfo &runInfo, int32_t loopIdx,
                                         int32_t curLoopSize, int32_t loopSize, TQue<QuePosition::VECIN, 1> &yInQue,
                                         TQue<QuePosition::VECIN, 1> &dxInQue, GlobalTensor<OUTDTYPE> &dxGm,
                                         GlobalTensor<OUTDTYPE> &yGm)
{
    int64_t srcGmOffset = 0;
    int64_t transpose_stride = 0;
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    if (constInfo.commonConstInfo.layoutType == BNGSD) {
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1Dv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1Dv;
        s1Offset = runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType * constInfo.commonConstInfo.dSizeV +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.dSizeV +
                   loopIdx * loopSize * constInfo.commonConstInfo.dSizeV;
        transpose_stride = 0;
    } else if (constInfo.commonConstInfo.layoutType == SBNGD) {
        s1Offset = runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType * constInfo.commonConstInfo.bN2GDv +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.bN2GDv +
                   loopIdx * loopSize * constInfo.commonConstInfo.bN2GDv;
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GDv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        transpose_stride = (constInfo.commonConstInfo.bN2GDv - constInfo.commonConstInfo.dSizeV) * sizeof(OUTDTYPE);
    } else if (constInfo.commonConstInfo.layoutType == BSNGD) {
        bOffset = runInfo.commonRunInfo.boIdx * constInfo.commonConstInfo.n2GS1Dv;
        s1Offset = runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType * constInfo.commonConstInfo.n2GDv +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2GDv +
                   loopIdx * loopSize * constInfo.commonConstInfo.n2GDv;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        transpose_stride = (constInfo.commonConstInfo.n2GDv - constInfo.commonConstInfo.dSizeV) * sizeof(OUTDTYPE);
    } else if (constInfo.commonConstInfo.layoutType == TND) {
        int64_t tndS1PrefixSum = (runInfo.commonRunInfo.boIdx == 0 ?
                                      0 :
                                      ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
        bOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * constInfo.commonConstInfo.dSizeV;
        s1Offset = runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType * constInfo.commonConstInfo.n2GDv +
                   runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2GDv +
                   loopIdx * loopSize * constInfo.commonConstInfo.n2GDv;
        gOffset = runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.dSizeV;
        n2Offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gDv;
        transpose_stride = (constInfo.commonConstInfo.n2GDv - constInfo.commonConstInfo.dSizeV) * sizeof(OUTDTYPE);
    }
    srcGmOffset = bOffset + n2Offset + gOffset + s1Offset;

    LocalTensor<OUTDTYPE> yTensor = yInQue.AllocTensor<OUTDTYPE>();
    LocalTensor<OUTDTYPE> dxTensor = dxInQue.AllocTensor<OUTDTYPE>();

    if constexpr (IS_D_NO_EQUAL) {
        Duplicate<OUTDTYPE>(yTensor, 0, curLoopSize * (uint32_t)dTemplateType);
        Duplicate<OUTDTYPE>(dxTensor, 0, curLoopSize * (uint32_t)dTemplateType);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)));
    }

    uint32_t dstBlockStride = ((uint32_t)dTemplateType - constInfo.dAlignToBlockForFp8) * sizeof(OUTDTYPE) / 32;
    DataCopyPad(dxTensor, dxGm[srcGmOffset],
                {static_cast<uint16_t>(curLoopSize),
                 static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(OUTDTYPE)),
                 static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
                {true, 0, static_cast<uint8_t>((constInfo.dAlignToBlockForFp8 - constInfo.commonConstInfo.dSizeV)), 0});
    DataCopyPad(yTensor, yGm[srcGmOffset],
                {static_cast<uint16_t>(curLoopSize),
                 static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV * sizeof(OUTDTYPE)),
                 static_cast<uint32_t>(transpose_stride), dstBlockStride, 0},
                {true, 0, static_cast<uint8_t>((constInfo.dAlignToBlockForFp8 - constInfo.commonConstInfo.dSizeV)), 0});

    yInQue.EnQue(yTensor);
    dxInQue.EnQue(dxTensor);
}

TEMPLATES_DEF
__aicore__ inline void CalculateCastSoftmaxGrad(FagConstInfo &constInfo, int32_t curLoopSize,
                                                TQue<QuePosition::VECIN, 1> &yInQue,
                                                TQue<QuePosition::VECIN, 1> &dxInQue,
                                                const LocalTensor<CALC_TYPE> &dstTensor)
{
    LocalTensor<OUTDTYPE> yTensor = yInQue.DeQue<OUTDTYPE>();
    LocalTensor<OUTDTYPE> dxTensor = dxInQue.DeQue<OUTDTYPE>();
    if constexpr ((uint32_t)dTemplateType <= 256) {
        AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, (uint32_t)dTemplateType, (uint32_t)dTemplateType>(
            dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlockForFp8);
    } else {
        if (constInfo.dAlignToBlock <= 384 && !IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 384, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock <= 320 && IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 320, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 320 && constInfo.dAlignToBlock <= 384 &&
                   IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 384, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 384 && constInfo.dAlignToBlock <= 448 &&
                   IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 448, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 448 && constInfo.dAlignToBlock <= 512 &&
                   IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 512, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock <= 512 && !IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 512, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 512 && constInfo.dAlignToBlock <= 576 &&
                   IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 576, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 576 && constInfo.dAlignToBlock <= 640 &&
                   IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 640, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock <= 640 && !IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 640, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else if (constInfo.dAlignToBlock > 640 && constInfo.dAlignToBlock <= 704 &&
                   IsSameType<OUTDTYPE, float>::value) {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, 704, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        } else {
            AscendC::MySoftmaxGradFrontCast<OUTDTYPE, CALC_TYPE, (uint32_t)dTemplateType, (uint32_t)dTemplateType>(
                dstTensor, yTensor, dxTensor, curLoopSize, constInfo.dAlignToBlock);
        }
    }
    yInQue.FreeTensor(yTensor);
    dxInQue.FreeTensor(dxTensor);
}

#endif

