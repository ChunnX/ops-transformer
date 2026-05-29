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
 * \file block_sparse_attention_common.h
 * \brief BlockSparseAttention kernel common declarations.
 * TODO: 待实现 - 后续补齐与正式 kernel dispatch 共享的常量、类型别名和调试开关。
 */

#ifndef BLOCK_SPARSE_ATTENTION_COMMON_H
#define BLOCK_SPARSE_ATTENTION_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

enum class BSALayout : uint32_t {
    BSND = 0,
    TND = 1,
};

enum class BSAKvLayout : uint32_t {
    CONTIGUOUS = 0,
    PA_ND = 1,
};

enum class BSADType : uint32_t {
    FP8_E4M3FN = 0,
    HIF8 = 1,
};

enum class BSACompressMode : uint32_t {
    NONE = 0,
    LEFT_UP_CAUSAL = 1,
    RIGHT_DOWN_CAUSAL = 2,
};

constexpr uint32_t BSA_BLOCK_SIZE = 128U;
constexpr uint32_t BSA_HEAD_DIM = 128U;
constexpr float BSA_EMPTY_LSE_VALUE = -3.4028234663852886e38F;

#endif // BLOCK_SPARSE_ATTENTION_COMMON_H
