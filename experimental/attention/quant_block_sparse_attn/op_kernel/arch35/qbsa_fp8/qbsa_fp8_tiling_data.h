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
 * \file qbsa_fp8_tiling_data.h
 * \brief Independent FP8 host/kernel tiling payload.
 */

#ifndef QBSA_FP8_TILING_DATA_H_
#define QBSA_FP8_TILING_DATA_H_

#include "../common/quant_block_sparse_attn_tiling_data.h"

namespace optiling {
constexpr uint32_t QBSA_CORE_SPLIT_NUM = QBSA_MAX_CORE_NUM + 1U;

#pragma pack(push, 8)

// FP8 RegBase initialization and task-boundary fields.
struct QuantBlockSparseAttnFp8Params {
    uint32_t bandIndex;
    uint32_t seqUsedQlenSize;
    uint32_t seqUsedKvlenSize;
    uint8_t layoutType;
    uint8_t isKvContinuous;
    uint8_t fromFused;
    uint8_t isGqa;
    int64_t s1OuterSize;
    uint32_t bnStartIdx[QBSA_CORE_SPLIT_NUM];
    uint32_t singleCoreSize;
    uint8_t needInit;
};

struct QuantBlockSparseAttnFp8TilingData {
    QuantBlockSparseAttnCommonParams commonParams;
    QuantBlockSparseAttnFp8Params fp8Params;
};

#pragma pack(pop)

static_assert(sizeof(QuantBlockSparseAttnFp8Params) == 184U, "QBSA FP8 parameters ABI changed");
static_assert(offsetof(QuantBlockSparseAttnFp8TilingData, commonParams) == 0U, "QBSA common tiling offset changed");
static_assert(offsetof(QuantBlockSparseAttnFp8TilingData, fp8Params) == 120U, "QBSA FP8 tiling offset changed");
static_assert(sizeof(QuantBlockSparseAttnFp8TilingData) == 304U, "QBSA FP8 tiling wire size changed");
static_assert(std::is_standard_layout<QuantBlockSparseAttnFp8TilingData>::value,
              "QBSA FP8 tiling data must have a standard layout");
static_assert(std::is_trivial<QuantBlockSparseAttnFp8TilingData>::value,
              "QBSA FP8 tiling data must remain a trivial POD");
static_assert(std::is_trivially_copyable<QuantBlockSparseAttnFp8TilingData>::value,
              "QBSA FP8 tiling data must support byte-wise serialization");
} // namespace optiling

#endif // QBSA_FP8_TILING_DATA_H_
