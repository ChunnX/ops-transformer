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
 * \file gm_layout.h
 * \brief
 */
#ifndef GM_LAYOUT_H
#define GM_LAYOUT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

// ----------------------------------------------GmLayout--------------------------------
enum class GmFormat {
    TND = 0,
    TNGD = 1,
    PA_BnBsND = 2,
    PA_BnNBsD = 3,
    PA_BnNBs = 4
};

template <GmFormat FORMAT>
struct GmLayout {
};


template <>
struct GmLayout<GmFormat::TNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d) {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::TND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t d) {
        shape = AscendC::MakeShape(t, n, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnBsND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d) {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t bsStride = nStride * n;
        uint64_t bnStride = bsStride * blockSize;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnNBsD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d) {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t bsStride = dStride * d;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnNBs> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize) {
        shape = AscendC::MakeShape(n, blockSize);

        uint64_t bsStride = 1;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n; //blockSize * kvHeadNum
        stride = AscendC::MakeStride(bnStride, nStride, bsStride);
    }
};
#endif