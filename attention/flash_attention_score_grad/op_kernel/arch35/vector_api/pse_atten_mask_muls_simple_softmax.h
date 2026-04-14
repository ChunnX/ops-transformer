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
 * \file pse_atten_mask_muls_simple_softmax.h
 */

#ifndef PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_
#define PSE_ATTEN_MASK_MULS_SIMPLE_SOFTMAX_
#include "../flash_attention_score_grad_common.h"
#include "../../../../common/op_kernel/arch35/pse.h"
#include "../../../../common/op_kernel/arch35/attenmask.h"
#include "vf_muls_sel_simple_softmax.h"
#include "vf_muls_sel_simple_softmax_aligned256.h"

using namespace commondef;

TEMPLATES_DEF
__aicore__ inline void CopyInPse(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo,
                                 TQue<QuePosition::VECIN, 1> &pseInQue, GlobalTensor<OUTDTYPE> &pseGm)
{
    // 调用公共函数，传入LocalTensor
    if constexpr (IS_PSE) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0 || pseInfo.pseType == 2 || pseInfo.pseType == 3) {
            return;
        }
        LocalTensor<OUTDTYPE> pseTensor = pseInQue.AllocTensor<OUTDTYPE>();
        // compute offset + copy
        PseCopyIn<CALC_TYPE, OUTDTYPE, IS_PSE>(pseTensor, pseGm, runInfo.commonRunInfo, constInfo.commonConstInfo,
                                               pseInfo);
        pseInQue.EnQue(pseTensor);
    }
}

TEMPLATES_DEF
__aicore__ inline void CopyInAttenMask(FagConstInfo &constInfo, FagRunInfo &runInfo, AttenMaskInfo &attenMaskInfo,
                                       TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                       TQue<QuePosition::VECIN, 1> &attenMaskInQuePre,
                                       GlobalTensor<uint8_t> &attenMaskGm)
{
    if constexpr (IS_ATTEN_MASK) {
        if (runInfo.commonRunInfo.halfS1RealSize == 0) {
            return;
        }
        AttenMaskCopyIn<IS_ATTEN_MASK>(attenMaskInQue, attenMaskInQuePre, attenMaskGm, runInfo.commonRunInfo,
                                       constInfo.commonConstInfo, attenMaskInfo);
    }
}

TEMPLATES_DEF
__aicore__ inline void CopyInMaxSum(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                    TQue<QuePosition::VECIN, 1> &maxSumInQue, GlobalTensor<CALC_TYPE> &maxGm,
                                    GlobalTensor<CALC_TYPE> &sumGm)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) { // TND + TND
        maxSumGmOffset =
            (runInfo.commonRunInfo.queryOffset / constInfo.commonConstInfo.dSize +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2G) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
    } else if (constInfo.commonConstInfo.layoutType == TND) { // TND + BNS8
        int64_t tndS1PrefixSum = (runInfo.commonRunInfo.boIdx == 0 ?
                                      0 :
                                      ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
        int64_t actualS1Len = 0;
        maxSumGmOffset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
        if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[0];
        } else {
            actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                          ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
        }
        maxSumGmOffset +=
            ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) *
                 actualS1Len +
             runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
    } else { // BNS8
        maxSumGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                               constInfo.commonConstInfo.gSize +
                           runInfo.commonRunInfo.goIdx) *
                              constInfo.commonConstInfo.s1Size +
                          runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType +
                          runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
                         MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
    }
    LocalTensor<CALC_TYPE> maxSumTensor = maxSumInQue.AllocTensor<CALC_TYPE>();
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        uint32_t srcStride = constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(
            maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
            maxGm[maxSumGmOffset],
            {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
             static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
            {false, 0, 0, 0});
    } else {
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(
            maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
            maxGm[maxSumGmOffset],
            {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
            {false, 0, 0, 0});
    }
    maxSumInQue.EnQue(maxSumTensor);
}

TEMPLATES_DEF
__aicore__ inline int64_t GetBNSOffset(FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    int64_t offset = 0;
    int64_t tndS1PrefixSum =
        (runInfo.commonRunInfo.boIdx == 0 ? 0 :
                                            ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1]);
    int64_t actualS1Len = 0;
    offset = tndS1PrefixSum * constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
    if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
        actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[0];
    } else {
        actualS1Len = ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx] -
                      ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1];
    }
    offset +=
        ((runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize + runInfo.commonRunInfo.goIdx) * actualS1Len +
         runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType +
         runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
        MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
    return offset;
}

TEMPLATES_DEF
__aicore__ inline void CopyInMaxSumWithSfmg(FagConstInfo &constInfo, FagRunInfo &runInfo,
                                            TQue<QuePosition::VECIN, 1> &maxSumInQue, GlobalTensor<CALC_TYPE> &maxGm,
                                            GlobalTensor<CALC_TYPE> &sumGm, GlobalTensor<CALC_TYPE> &sfmgWorkspaceGm)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    int64_t maxSumGmOffset = 0;
    int64_t sfmgGmOffset = 0;
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) { // TND + TND
        maxSumGmOffset =
            (runInfo.commonRunInfo.queryOffset / constInfo.commonConstInfo.dSize +
             runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx() * constInfo.commonConstInfo.n2G) *
            MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
        sfmgGmOffset = GetBNSOffset<TEMPLATE_ARGS>(constInfo, runInfo);
    } else if (constInfo.commonConstInfo.layoutType == TND) { // TND + BNS8
        maxSumGmOffset = GetBNSOffset<TEMPLATE_ARGS>(constInfo, runInfo);
        sfmgGmOffset = maxSumGmOffset;
    } else { // BNS8
        maxSumGmOffset = (((runInfo.commonRunInfo.boIdx * constInfo.n2Size + runInfo.commonRunInfo.n2oIdx) *
                               constInfo.commonConstInfo.gSize +
                           runInfo.commonRunInfo.goIdx) *
                              constInfo.commonConstInfo.s1Size +
                          runInfo.commonRunInfo.s1oIdx * (uint32_t)s1TemplateType +
                          runInfo.commonRunInfo.firstHalfS1RealSize * GetSubBlockIdx()) *
                         MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE);
        sfmgGmOffset = maxSumGmOffset;
    }
    LocalTensor<CALC_TYPE> maxSumTensor = maxSumInQue.AllocTensor<CALC_TYPE>();
    if (constInfo.tndMaxSumLayout == MAX_SUM_TND) {
        uint32_t srcStride = constInfo.commonConstInfo.n2G * MAX_SUM_REDUCE_AXIS_SIZE - MAX_SUM_REDUCE_AXIS_SIZE;
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
                     static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(
            maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
            maxGm[maxSumGmOffset],
            {static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize),
             static_cast<uint32_t>(MAX_SUM_REDUCE_AXIS_SIZE), static_cast<uint32_t>(srcStride), 0, 0},
            {false, 0, 0, 0});
    } else {
        DataCopyPad(maxSumTensor, sumGm[maxSumGmOffset],
                    {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(
            maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
            maxGm[maxSumGmOffset],
            {1, static_cast<uint16_t>(runInfo.commonRunInfo.halfS1RealSize * MAX_SUM_REDUCE_AXIS_SIZE), 0, 0},
            {false, 0, 0, 0});
    }
    DataCopyPad(
        maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE) << 1],
        sfmgWorkspaceGm[sfmgGmOffset / 8],
        {1, static_cast<uint16_t>((uint32_t)s1TemplateType / CV_CORE_RATIO * sizeof(CALC_TYPE)), 0, 0},
        {false, 0, 0, 0});
    maxSumInQue.EnQue(maxSumTensor);
}

/*************************
Function： VF计算函数，实现Pse + AttenMask + Muls + SimpleSoftmax计算
baseParams：循环定参，入参
runInfo: 循环变参，入参
attenMaskInfo：attenMask相关参数，入参
maxSumInQue：maxSum分配Que，入参
attenMaskInQue：attenMask分配Que，入参
pseInQue：pse分配Que，入参
dstTensor：返回计算结果，出参
srcTensor：VF输入，入参
*************************/
TEMPLATES_DEF
__aicore__ inline void CalculatePseMulsSelSimpleSoftMax(FagConstInfo &constInfo, FagRunInfo &runInfo, PseInfo &pseInfo,
                                                        AttenMaskInfo &attenMaskInfo,
                                                        TQue<QuePosition::VECIN, 1> &maxSumInQue,
                                                        TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                                        TQue<QuePosition::VECIN, 1> &pseInQue,
                                                        LocalTensor<CALC_TYPE> &dstTensor,
                                                        LocalTensor<CALC_TYPE> &srcTensor, __gm__ uint8_t *pseSlope)
{
    if (runInfo.commonRunInfo.halfS1RealSize == 0) {
        return;
    }
    constexpr bool IS_FP8_INPUT = IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_TYPE, hifloat8_t>::value;
    LocalTensor<uint8_t> attenMaskTensor;
    LocalTensor<OUTDTYPE> pseTensor;
    if constexpr (IS_ATTEN_MASK) {
        attenMaskTensor = attenMaskInQue.DeQue<uint8_t>();
    }
    // Compute
    LocalTensor<CALC_TYPE> maxSumTensor = maxSumInQue.DeQue<CALC_TYPE>();
    constexpr uint16_t CONVERT_VECTOR_BASEN = static_cast<uint16_t>(s2TemplateType);
    if constexpr (IS_PSE) {
        float posShift;
        float slopes;
        // sparse mode 8非batch = 0下不需要加偏移
        if ((pseInfo.pseType == 2 || pseInfo.pseType == 3) && constInfo.commonConstInfo.layoutType == TND &&
            attenMaskInfo.compressMode == static_cast<uint8_t>(BAND_LEFT_UP_CASUAL) &&
            runInfo.commonRunInfo.boIdx != 0) {
            pseInfo.qStartIdx = 0;
            pseInfo.kvStartIdx = 0;
        }
        ComputeInnerPseOffset<CALC_TYPE, OUTDTYPE, IS_PSE>(slopes, posShift, runInfo.commonRunInfo,
                                                           constInfo.commonConstInfo, pseInfo, pseSlope);
        LocalTensor<OUTDTYPE> pseTensor = pseInQue.DeQue<OUTDTYPE>();
        if (IS_FP8_INPUT) {
            AscendC::MulsSelSimpleSoftMaxAligned256<OUTDTYPE, CALC_TYPE, CONVERT_VECTOR_BASEN, IS_ATTEN_MASK, IS_PSE,
                                                    IS_DETER_OLD(DETER_SPARSE_TYPE)>(
                dstTensor, maxSumTensor,
                maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
                srcTensor, pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                pseInfo.pseLayoutType, posShift, slopes);
        } else if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, 128, IS_ATTEN_MASK, IS_PSE,
                                          IS_DETER_OLD(DETER_SPARSE_TYPE)>(
                dstTensor, maxSumTensor,
                maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
                srcTensor, pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                pseInfo.pseLayoutType, posShift, slopes);
        } else {
            AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, 64, IS_ATTEN_MASK, IS_PSE,
                                          IS_DETER_OLD(DETER_SPARSE_TYPE)>(
                dstTensor, maxSumTensor,
                maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
                srcTensor, pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize, pseInfo.pseType,
                pseInfo.pseLayoutType, posShift, slopes);
        }
        pseInQue.FreeTensor(pseTensor);
    } else {
        if (IS_FP8_INPUT) {
            AscendC::MulsSelSimpleSoftMaxAligned256<OUTDTYPE, CALC_TYPE, CONVERT_VECTOR_BASEN, IS_ATTEN_MASK, IS_PSE,
                                                    IS_DETER_OLD(DETER_SPARSE_TYPE)>(
                dstTensor, maxSumTensor,
                maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
                srcTensor, pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        } else if (runInfo.commonRunInfo.s2RealSize > 64) {
            AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, 128, IS_ATTEN_MASK, IS_PSE,
                                          IS_DETER_OLD(DETER_SPARSE_TYPE)>(
                dstTensor, maxSumTensor,
                maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
                srcTensor, pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        } else {
            AscendC::MulsSelSimpleSoftMax<OUTDTYPE, CALC_TYPE, 64, IS_ATTEN_MASK, IS_PSE,
                                          IS_DETER_OLD(DETER_SPARSE_TYPE)>(
                dstTensor, maxSumTensor,
                maxSumTensor[(uint32_t)s1TemplateType / CV_CORE_RATIO * MAX_SUM_REDUCE_AXIS_SIZE / sizeof(CALC_TYPE)],
                srcTensor, pseTensor, attenMaskTensor, constInfo.scaleValue, constInfo.attenMaskMinValue,
                runInfo.commonRunInfo.halfS1RealSize, runInfo.commonRunInfo.s2RealSize);
        }
    }
    // FreeTensor
    if constexpr (IS_ATTEN_MASK) {
        attenMaskInQue.FreeTensor(attenMaskTensor);
    }
    maxSumInQue.FreeTensor(maxSumTensor);
}

#endif