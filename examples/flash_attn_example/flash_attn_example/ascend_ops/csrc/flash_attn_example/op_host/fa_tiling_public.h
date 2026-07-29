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
 * \brief Public tiling data structures.
 */

#ifndef FA_TILING_PUBLIC_H
#define FA_TILING_PUBLIC_H

#include <cstdint>
#include <vector>

#include <ATen/Operators.h>
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

struct ContextParamsForTiling {
    const at::Tensor *attentionMask = nullptr;

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
    double softmaxScale;

    int32_t numKeyValueHeads;
};

constexpr uint32_t M_BASE_SIZE = 128;
constexpr uint32_t S2_BASE_SIZE = 128;

} // namespace fa_host
} // namespace ascend_ops
#endif
