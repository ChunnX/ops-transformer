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
 * \file fa_vec1_vf_static.h
 * \brief 静态Tensor编程兼容版本的 ProcessVec1Vf / ProcessVec1VfDn / ProcessVec1VfDnCausal。
 *
 * 与 FaVectorApi 原版的唯一区别：vselrIndexesBuf 参数类型从 TBuf<>* 改为
 * LocalTensor<uint8_t>*，消除对 TBuf 框架接口的依赖。
 * 下层实现函数（ProcessVec1DnNoUpdate / ProcessVec1DnUpdate / ProcessVec1DnCausal /
 * ProcessVec1NoUpdateImpl* / ProcessVec1UpdateImpl* 等）本身已接受 LocalTensor<uint8_t>，
 * 无需修改，直接复用原始头文件中的定义。
 */
#ifndef FA_VEC1_VF_STATIC_H_
#define FA_VEC1_VF_STATIC_H_

#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"

namespace fa_static {

using namespace AscendC;
using namespace FaVectorApi;

// ============================================================
// ProcessVec1NoUpdate — 静态版，LocalTensor<uint8_t>* 替代 TBuf<>*
// ============================================================
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1NoUpdate(
    const LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> *vselrIndexesBuf,
    const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f, const float pScale = 1.0f,
    float sinkValue = 0.0f)
{
    if constexpr (useNz) {
        if constexpr (oriNRange == GT_256_AND_LTE_512) {
            LocalTensor<uint8_t> indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)];
            ProcessVec1NoUpdateGeneralImpl512GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize,
                hasAtten, pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, expSumTensor, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        } else {
            LocalTensor<uint8_t> indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)];
            ProcessVec1NoUpdateGeneralImpl256GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize,
                hasAtten, pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, expSumTensor, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        }
    } else {
        if constexpr (oriNRange == GT_128_AND_LTE_256) {
            ProcessVec1NoUpdateGeneralImpl256<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop, hasSink>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb, sinkValue);
        } else if constexpr (oriNRange == EQ_128) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
            }
            ProcessVec1NoUpdateImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes,
                posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)];
            }
            ProcessVec1NoUpdateImpl64<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes,
                posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_256_AND_LTE_512) {
            ProcessVec1NoUpdateGeneralImpl512<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb);
        } else if constexpr (oriNRange == GT_512_AND_LTE_1024) {
            ProcessVec1NoUpdateGeneralImpl1024<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb);
        } else { // GT_64_AND_LTE_128
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
            }
            ProcessVec1NoUpdateGeneralImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride,
                slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue,
                sinkValue);
        }
    }
}

// ============================================================
// ProcessVec1Update — 静态版，LocalTensor<uint8_t>* 替代 TBuf<>*
// ============================================================
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1Update(
    const LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> *vselrIndexesBuf,
    const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& pScaleTensor, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const float dScaleQK, const T minValue, float keepProb, const LocalTensor<T>& queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float pScale = 1.0f, float sinkValue = 0.0f)
{
    if constexpr (useNz) {
        if constexpr (oriNRange == GT_256_AND_LTE_512) {
            LocalTensor<uint8_t> indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)];
            ProcessVec1UpdateGeneralImpl512GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, sharedTmpBuffer, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        } else {
            LocalTensor<uint8_t> indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)];
            ProcessVec1UpdateGeneralImpl256GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, sharedTmpBuffer, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        }
    } else {
        if constexpr (oriNRange == GT_128_AND_LTE_256) {
            ProcessVec1UpdateGeneralImpl256<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, hasSink>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb, sinkValue);
        } else if constexpr (oriNRange == EQ_128) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
            }
            ProcessVec1UpdateImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor, m, originN, pseStride,
                slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)];
            }
            ProcessVec1UpdateImpl64<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor, m, originN, pseStride,
                slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_256_AND_LTE_512) {
            ProcessVec1UpdateGeneralImpl512<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
                pseStride, slopes, posShift, scale, minValue, keepProb);
        } else if constexpr (oriNRange == GT_512_AND_LTE_1024) {
            ProcessVec1UpdateGeneralImpl1024<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
                pseStride, slopes, posShift, scale, minValue, keepProb);
        } else { // GT_64_AND_LTE_128
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)];
            }
            ProcessVec1UpdateGeneralImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor, m, originN,
                pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue,
                sinkValue);
        }
    }
}

// ============================================================
// ProcessVec1Vf — 静态版，LocalTensor<uint8_t>* 替代 TBuf<>*
// ============================================================
template <typename T, typename T2, typename pseShiftType, bool isUpdate = false, uint32_t s1BaseSize = 128,
          uint32_t s2BaseSize = 128, OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1Vf(const LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> *vselrIndexesBuf,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<T>& pScaleTensor, const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f,
    const float pScale = 1.0f, float sinkValue = 0.0f)
{
    if constexpr (!isUpdate) {
        fa_static::ProcessVec1NoUpdate<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer,
            m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb,
            deSCaleKValue, pScale, sinkValue);
    } else {
        fa_static::ProcessVec1Update<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor,
            m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb,
            deSCaleKValue, pScale, sinkValue);
    }
}

// ============================================================
// ProcessVec1VfDn — 静态版，LocalTensor<uint8_t>* 替代 TBuf<>*
// ============================================================
template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 128,
    bool hasSink = false>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor, LocalTensor<uint8_t> *vselrIndexesBuf,
                                       const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
                                       const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                       float keepProb, bool needAtten, const float pScale = 1.0f,
                                       const float sinkValue = 0.0f)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)];
        }
        ProcessVec1DnNoUpdate<T, T2, hasAtten, ubN, hasSink>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor,
            m, n, originN, scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)];
        }
        ProcessVec1DnUpdate<T, T2, hasAtten, ubN, hasSink>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor,
            m, n, originN, scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
    }
}

// ============================================================
// ProcessVec1VfDnCausal — 静态版，LocalTensor<uint8_t>* 替代 TBuf<>*
// ============================================================
template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
    bool hasSink = false, bool hasInvalidLine = false>
__aicore__ inline void ProcessVec1VfDnCausal(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor, LocalTensor<uint8_t> *vselrIndexesBuf,
                                       const LocalTensor<uint8_t>& maskTensor,
                                       const LocalTensor<uint8_t>& sharedTmpBuffer,
                                       const uint32_t m, const uint32_t n,
                                       const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                       float keepProb, bool needAtten, const float pScale = 1.0f,
                                       const float sinkValue = 0.0f)
{
    LocalTensor<uint8_t> indexesTensor;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
        IsSameType<T2, hifloat8_t>::value) {
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)];
    }
    ProcessVec1DnCausal<T, T2, isUpdate, hasAtten, ubN, hasSink, hasInvalidLine>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, sharedTmpBuffer,
            m, n, originN, scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
}

} // namespace fa_static

#endif // FA_VEC1_VF_STATIC_H_
