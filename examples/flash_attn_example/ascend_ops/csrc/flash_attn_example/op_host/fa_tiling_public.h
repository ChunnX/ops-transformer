/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file fa_tiling_public.h
 * \brief Public tiling data structures for FIA.
 */

#ifndef FA_TILING_PUBLIC_H
#define FA_TILING_PUBLIC_H

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <ATen/Operators.h>
#include "tiling/platform/platform_ascendc.h"
#include "../fia_tiling_data.h"

namespace ascend_ops {
namespace fa_host {

enum class InputLayout {
    SH,
    BSH,
    BNSD,
    NSD,
    BSND,
    BNSD_BSND,
    TND,
    NTD_TND,
    NZ,
    BBH,
    BNBD,
    NONE,
};

enum class FaSparseEnum : uint8_t {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

enum class SplitCoreMode {
    SPLIT_NBS_VECTOR = 0,
    SPLIT_NBS_CUBE,
    SPLIT_ONEN_VECTOR,
    SPLIT_ONEN_CUBE,
    BALANCE_VECTOR,
    BALANCE_CUBE,
};

enum class LayoutType : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum FaAttnMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE,
    BAND_LEFT_UP_CAUSAL_MODE
};

struct ContextParamsForTiling {

    const at::Tensor *attentionMask = nullptr;
    const at::Tensor *actualSequenceLengthQ = nullptr;
    const at::Tensor *actualSequenceLengthKV = nullptr;

    at::ScalarType inputDataType = at::ScalarType::Float;
    at::ScalarType kDataType = at::ScalarType::Float;
    at::ScalarType vDataType = at::ScalarType::Float;

    at::ScalarType maskDataType = at::ScalarType::Bool;

    at::ScalarType outputDataType = at::ScalarType::Float;

    at::IntArrayRef queryInputShape;
    at::IntArrayRef keyInputShape;
    at::IntArrayRef valueInputShape;
    at::IntArrayRef attentionMaskShape;
    at::IntArrayRef outputShape;

    int32_t headsNumber;
    int32_t sparseMode;
    double softmaxScale;

    int32_t numKeyValueHeads;
    size_t workspaceSize;

    std::vector<const at::IntArrayRef *> kTensorList = {nullptr};
    std::vector<const at::IntArrayRef *> vTensorList = {nullptr};
};

constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

constexpr uint32_t SOUTER_FACTOR_SUB = 32;
constexpr uint32_t SOUTER_FACTOR_DEFAULT = 64;
constexpr uint32_t SINNER_FACTOR_SUB = 64;
constexpr uint32_t SINNER_FACTOR_DEFAULT = 128;
constexpr uint32_t SINNER_FACTOR_DOUBLE = 256;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t CV_RATIO = 2U;
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;
static const int64_t GM_ALIGN = 512;

constexpr uint32_t LOOP_BEGIN_NUM = 0;

constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SOUTER_128 = 128;
constexpr uint32_t SINNER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t SINNER_512 = 512;
constexpr uint32_t DSIZE_64 = 64;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_192 = 192;
constexpr uint32_t DSIZE_256 = 256;
constexpr uint32_t DSIZE_512 = 512;
constexpr uint32_t DSIZE_576 = 576;

} // namespace fa_host
} // namespace ascend_ops
#endif
