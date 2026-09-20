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
 * \file vf_flashupdate_trans_fp16.h
 * \brief FP16 softmax (quant_mode=3) V2阶段转置布局专用VF:
 *        O累加器保持MM2输出的转置布局(d, 128)行主序, softmax因子(exp/row_sum)是
 *        per-S1行(即转置布局的列j), 沿行内连续 -> 乘加/除法免转置, 行主序直接算;
 *        与common层FlashUpdateNew/LastDivNew的per-row广播方向相反, 故写专用VF
 */
#ifndef VF_FLASHUPDATE_TRANS_FP16_H_
#define VF_FLASHUPDATE_TRANS_FP16_H_

#include "kernel_tensor.h"

namespace FaVectorApi {

/* **************************************************************************************************
 * FlashUpdateTrans, O[i][j] = O[i][j]*exp[j] + M[i][j], fp32
 * ************************************************************************************************* */
// O[i][j] = O[i][j]*exp[j] + M[i][j]
template <typename T, uint16_t m, uint16_t n>
__simd_vf__ inline void FlashUpdateTransVF(__ubuf__ T *dstUb, __ubuf__ T *curUb, __ubuf__ float *expF32Ub)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateTrans, T must be float");
    RegTensor<float> vregExp;
    RegTensor<float> vregPre;
    RegTensor<float> vregCur;
    RegTensor<float> vregMul;
    RegTensor<float> vregAdd;
    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t nLoops = n / floatRepSize;
    for (uint16_t i = 0; i < m; ++i) {
        for (uint16_t j = 0; j < nLoops; ++j) {
            LoadAlign(vregExp, expF32Ub + j * floatRepSize); // 列j段因子, 行内连续
            LoadAlign(vregPre, dstUb + i * n + j * floatRepSize);
            LoadAlign(vregCur, curUb + i * n + j * floatRepSize);
            Mul(vregMul, vregPre, vregExp, pregAll);
            Add(vregAdd, vregMul, vregCur, pregAll);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dstUb + i * n + j * floatRepSize, vregAdd, pregAll);
        }
    }
}

template <typename T, uint16_t m, uint16_t n>
__aicore__ inline void FlashUpdateTrans(const LocalTensor<T> &dstTensor, const LocalTensor<T> &curTensor,
                                        const LocalTensor<float> &expF32Tensor)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *curUb = (__ubuf__ T *)curTensor.GetPhyAddr();
    __ubuf__ float *expUb = (__ubuf__ float *)expF32Tensor.GetPhyAddr();
    FlashUpdateTransVF<T, m, n>(dstUb, curUb, expUb);
}

/* **************************************************************************************************
 * LastDivTrans, O[i][j] = O[i][j] / rowSum[j], fp32
 * ************************************************************************************************* */
// O[i][j] = O[i][j] / rowSum[j]
template <typename T, uint16_t m, uint16_t n>
__simd_vf__ inline void LastDivTransVF(__ubuf__ T *dstUb, __ubuf__ float *sumUb)
{
    static_assert(IsSameType<T, float>::value, "VF LastDivTrans, T must be float");
    RegTensor<float> vregSum;
    RegTensor<float> vregSrc;
    RegTensor<float> vregDiv;
    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t nLoops = n / floatRepSize;
    for (uint16_t i = 0; i < m; ++i) {
        for (uint16_t j = 0; j < nLoops; ++j) {
            LoadAlign(vregSum, sumUb + j * floatRepSize);
            LoadAlign(vregSrc, dstUb + i * n + j * floatRepSize);
            Div(vregDiv, vregSrc, vregSum, pregAll);
            StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dstUb + i * n + j * floatRepSize, vregDiv, pregAll);
        }
    }
}

template <typename T, uint16_t m, uint16_t n>
__aicore__ inline void LastDivTrans(const LocalTensor<T> &dstTensor, const LocalTensor<float> &sumTensor)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ float *sumUb = (__ubuf__ float *)sumTensor.GetPhyAddr();
    LastDivTransVF<T, m, n>(dstUb, sumUb);
}

/* **************************************************************************************************
 * ComputeLseTrans, lse[j] = maxF32[j] + ln(rowSum[j] * pScaleRec), fp32
 * max==minValue(half域LN2对齐后的无效行max经cast到float)的无效行输出3e+99
 * ************************************************************************************************* */
// lse[j] = maxF32[j] + ln(rowSum[j] * pScaleRec); max==minValue的无效行输出3e+99
// minValue由调用侧经UpdateMinCheckValue计算(half域初始max -65504经LN2对齐后的float域值),
// 与V1 VF对无效行max的half域LN2对齐输出一致, 用于LSE阶段无效行检测
template <typename T>
__simd_vf__ inline void ComputeLseTransVF(__ubuf__ T *dstUb, __ubuf__ T *sumUb, __ubuf__ T *maxUb,
                                          const uint32_t dealCount, const float pScaleRec, const float minValue)
{
    static_assert(IsSameType<T, float>::value, "VF ComputeLseTrans, T must be float");
    RegTensor<float> vregSum;
    RegTensor<float> vregMax;
    RegTensor<float> vregRes;
    RegTensor<float> vregResFinal;
    RegTensor<float> vregMinValue;
    RegTensor<float> vregInfValue;
    RegTensor<float> vregScaleRec;
    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregCompare;
    constexpr uint16_t floatRepSize = 64;
    constexpr float infValue = 3e+99; // 3e+99 for float inf

    Duplicate(vregMinValue, minValue);
    Duplicate(vregInfValue, infValue);
    Duplicate(vregScaleRec, pScaleRec);

    uint16_t loops = (dealCount + floatRepSize - 1) / floatRepSize;
    for (uint16_t l = 0; l < loops; ++l) {
        uint32_t cnt = dealCount - l * floatRepSize;
        cnt = cnt < floatRepSize ? cnt : floatRepSize;
        MaskReg preg = UpdateMask<float>(cnt);
        LoadAlign(vregSum, sumUb + l * floatRepSize);
        LoadAlign(vregMax, maxUb + l * floatRepSize);
        Mul(vregSum, vregSum, vregScaleRec, preg); // rowSum/pScale
        Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSum, preg);
        Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, preg);
        Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, preg);
        Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dstUb + l * floatRepSize, vregResFinal, preg);
    }
}

template <typename T>
__aicore__ inline void ComputeLseTrans(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sumTensor,
                                       const LocalTensor<T> &maxTensor, const uint32_t dealCount, const float pScaleRec,
                                       const float minValue)
{
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *sumUb = (__ubuf__ T *)sumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    ComputeLseTransVF<T>(dstUb, sumUb, maxUb, dealCount, pScaleRec, minValue);
}

} // namespace FaVectorApi

#endif // VF_FLASHUPDATE_TRANS_FP16_H_
