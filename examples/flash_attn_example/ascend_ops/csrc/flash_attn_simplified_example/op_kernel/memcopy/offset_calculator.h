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
 * \file offset_calculator.h
 * \brief
 */
#ifndef OFFSET_CALCULATOR_H
#define OFFSET_CALCULATOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using AscendC::GlobalTensor;
using AscendC::LocalTensor;

enum class GmFormat {
    BSNGD = 0,
    BNGSD = 1,
    NGBSD = 2,
    TNGD = 3,
    NGTD = 4,
    BSND = 5,
    BNSD = 6,
    TND = 7,
    NTD = 8,
    PA_BnBsND = 9,
    PA_BnNBsD = 10,
    PA_NZ = 11,
    NGD = 12, // post_quant
    ND = 13, //antiquant no PA
    BS2 = 14,
    BNS2 = 15,
    PA_BnBs = 16, //antiquant PA
    PA_BnNBs = 17,
    BN2GS1S2 = 18, //PSE_GmFormat
    SBNGD = 19,
    SBND = 20,
    NTGD = 21,
    TND2 = 22, // VSCALE, 尾轴2
    PA_NZ_K_SCALE = 23,
};

struct GmCoord {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t dIdx;
    uint32_t gS1DealSize;
    uint32_t dDealSize;
};

struct BsngdOffsetInfo {
    uint64_t strideB;
    uint64_t strideN2;
    uint64_t strideG;
    uint64_t strideS1;
    uint64_t strideD;
    uint64_t dimB;
    uint64_t dimN2;
    uint64_t dimG;
    uint64_t dimS1;
    uint64_t dimD;
};

struct BsndOffsetInfo {
    uint64_t strideB;
    uint64_t strideN2;
    uint64_t strideS2;
    uint64_t strideD;
    uint64_t dimB;
    uint64_t dimN2;
    uint64_t dimS2;
    uint64_t dimD;
};

template <GmFormat FORMAT>
struct OffsetInfoType {};

template <>
struct OffsetInfoType<GmFormat::BSNGD> {
    using Type = BsngdOffsetInfo;
};

template <>
struct OffsetInfoType<GmFormat::BSND> {
    using Type = BsndOffsetInfo;
};

template <typename Q_T, GmFormat FORMAT, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
struct FaGmTensor {
    GlobalTensor<Q_T> gmTensor;
    typename OffsetInfoType<FORMAT>::Type offsetInfo;

    __aicore__ inline FaGmTensor() {}

    __aicore__ inline FaGmTensor(__gm__ Q_T *gm, uint32_t b, uint32_t n2, uint32_t a, uint32_t c, uint32_t d) {
        gmTensor.SetGlobalBuffer(gm);
        if constexpr (FORMAT == GmFormat::BSNGD) {
            offsetInfo.dimB = b; offsetInfo.dimN2 = n2; offsetInfo.dimG = a;
            offsetInfo.dimS1 = c; offsetInfo.dimD = d;
            offsetInfo.strideD = 1; offsetInfo.strideG = d;
            offsetInfo.strideN2 = (uint64_t)a * d;
            offsetInfo.strideS1 = (uint64_t)n2 * a * d;
            offsetInfo.strideB = (uint64_t)c * n2 * a * d;
        } else {
            offsetInfo.dimB = b; offsetInfo.dimN2 = n2;
            offsetInfo.dimS2 = a; offsetInfo.dimD = c;
            offsetInfo.strideD = 1; offsetInfo.strideN2 = c;
            offsetInfo.strideS2 = (uint64_t)n2 * c;
            offsetInfo.strideB = (uint64_t)a * n2 * c;
        }
    }
};

template <typename OUT_T>
struct FaUbTensor {
    LocalTensor<OUT_T> tensor;
    uint32_t rowCount;
    uint32_t colCount;
};

template <typename Q_T>
struct FaL1Tensor {
    LocalTensor<Q_T> tensor;
    uint32_t rowCount;
};

template <typename OffsetInfoT>
__aicore__ inline uint64_t GetOffset(const OffsetInfoT &info,
    uint32_t bIdx, uint32_t n2Idx, uint32_t aIdx, uint32_t cIdx, uint32_t dIdx)
{
    if constexpr (std::is_same_v<OffsetInfoT, BsngdOffsetInfo>) {
        return bIdx * info.strideB + n2Idx * info.strideN2 + aIdx * info.strideG +
               cIdx * info.strideS1 + dIdx * info.strideD;
    } else {
        return bIdx * info.strideB + n2Idx * info.strideN2 + aIdx * info.strideS2 + cIdx * info.strideD;
    }
}

#endif
