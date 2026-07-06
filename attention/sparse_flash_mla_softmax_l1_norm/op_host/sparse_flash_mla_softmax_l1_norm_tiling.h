/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_softmax_l1_norm_tiling.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_H_
#define SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {

// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t LSE_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INDEX = 3;
constexpr uint32_t CUM_SEQ_Q_INDEX = 4;
constexpr uint32_t CUM_SEQ_K_INDEX = 5;
constexpr uint32_t SEQ_USED_Q_INDEX = 6;
constexpr uint32_t SEQ_USED_K_INDEX = 7;
constexpr uint32_t CMP_RESIDUAL_K_INDEX = 8;
constexpr uint32_t TOPK_LENGTH_INDEX = 9;
constexpr uint32_t METADATA_INDEX = 10;
// Outputs Index
constexpr uint32_t SOFTMAX_L1_NORM_INDEX = 0;
// Attributes Index
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 0;
constexpr uint32_t ATTR_LAYOUT_K_INDEX = 1;
constexpr uint32_t ATTR_SCALE_VALUE_INDEX = 2;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 3;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 4;

enum class SmlaDataLayout : uint32_t {
    BSND = 0,
    TND = 1,
};

// TilingData 定义 (参考 DESIGN.md §0.4.3)
BEGIN_TILING_DATA_DEF(SmlaSoftmaxL1NormTilingData)
TILING_DATA_FIELD_DEF(int64_t, b)
TILING_DATA_FIELD_DEF(int64_t, n2)
TILING_DATA_FIELD_DEF(int64_t, g)
TILING_DATA_FIELD_DEF(int64_t, s1)
TILING_DATA_FIELD_DEF(int64_t, s2)
TILING_DATA_FIELD_DEF(int64_t, d)
TILING_DATA_FIELD_DEF(int64_t, selectedBlockCount)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum)
TILING_DATA_FIELD_DEF(int64_t, layout)
TILING_DATA_FIELD_DEF(int32_t, maskMode)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(int32_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint8_t, hasUsedSeqQ)
TILING_DATA_FIELD_DEF(uint8_t, hasUsedSeqK)
TILING_DATA_FIELD_DEF(uint8_t, hasTopK)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseFlashMlaSoftmaxL1Norm, SmlaSoftmaxL1NormTilingData)

struct SparseFlashMlaSoftmaxL1NormCompileInfo {};

} // namespace optiling

#endif // SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_TILING_H_
