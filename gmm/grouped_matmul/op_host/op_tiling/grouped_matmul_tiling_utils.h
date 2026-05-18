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
 * \file grouped_matmul_tiling_utils.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_UTILS_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_UTILS_H

#include "grouped_matmul_tiling.h"

#include <array>
#include <climits>
#include <map>
#include <sstream>
#include <vector>

#include "err/ops_err.h"
#include "../../op_kernel/grouped_matmul_tiling_key.h"

using namespace Ops::Transformer::OpTiling;
using namespace ge;
using namespace AscendC;
using namespace GroupedMatmul;

namespace optiling {
static inline uint32_t SixteenAlign(uint32_t a, bool up = false) {
    if (up) {
        a += 15U;  // 15: 16 bytes up-align
    }
    return a & ~15U;  // ~15: 16 bytes down-align
}

static inline int64_t SixteenAlign(int64_t a, bool up = false) {
    if (up) {
        a += 15;  // 15: 16 bytes up-align
    }
    return a & ~15;  // ~15: 16 bytes down-align
}

template <typename T>
static inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

constexpr uint32_t GROUP_LIST_SPARSE_M = 2U;
constexpr float EFFECTIVE_TASK_RATIO = 0.95f;
constexpr int64_t DOUBLE_VECTOT_THRESHOLD_K_LOWER = 1024L;
constexpr int64_t DOUBLE_VECTOT_THRESHOLD_K_UPPER = 2048L;
constexpr int32_t SMALL_TUNING_CONFIG_THRESHOLD = 128;
constexpr int32_t BIAS_REMAIN_SPACE = 2 * 1024;
constexpr int32_t MIN_BASE_M = 16;
constexpr int64_t FIXAXISMOVE_K1 = 2048L;
constexpr int64_t FIXAXISMOVE_K2 = 7168L;
constexpr int64_t FIXAXISMOVE_N1 = 7168L;
constexpr int64_t FIXAXISMOVE_N2 = 4096L;
constexpr int32_t FIXAXISMOVE_GROUP_NUM = 4;
constexpr int64_t FIXAXISMOVE_PERM_LOWER = 128L;
constexpr int64_t FIXAXISMOVE_PERM_UPPER = 512L;
constexpr int64_t FIXAXISMOVE_SPLIT_ITEM2 = 2L;
constexpr int64_t FIXAXISMOVE_SPLIT_ITEM3 = 3L;
constexpr int64_t FIXAXISMOVE_GROUP_LIST_TYPE = 0L;
constexpr int32_t FIXAXISMOVE_GROUP_TYPE = 0;
constexpr size_t TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX = 0;
constexpr size_t TUNING_CONFIG_A8W4_SPEC_SCENARIO_INDEX = 1;
constexpr size_t TUNING_CONFIG_ALLOW_WORKSPACE_INDEX = 2;
constexpr int64_t SPLITK_M_N_RATIO_THRESHOLD_2 = 2L;
constexpr int64_t A4W4OPTIMIZE_K_LOWER = 1024L;
constexpr int64_t A4W4OPTIMIZE_K_UPPER = 16384L;
constexpr int64_t A4W4OPTIMIZE_N_LOWER = 128L;
constexpr int64_t A4W4OPTIMIZE_N_UPPER = 16384L;
constexpr int32_t A4W4OPTIMIZE_GROUP_NUM_LOWER = 1;
constexpr int32_t A4W4OPTIMIZE_GROUP_NUM_UPPER = 256;
constexpr int64_t A4W4OPTIMIZE_PERM_LOWER = 16L;
constexpr int64_t A4W4OPTIMIZE_PERM_UPPER = 10240L;
constexpr int64_t A4W4OPTIMIZE_SPLIT_ITEM2 = 2L;
constexpr int64_t A4W4OPTIMIZE_SPLIT_ITEM3 = 3L;
constexpr int64_t A4W4OPTIMIZE_GROUP_LIST_TYPE = 0L;
constexpr int64_t A4W4OPTIMIZE_GROUP_TYPE = 0L;
constexpr int64_t A4W4OPTIMIZE_QUANT_GROUP_SIZE = 256L;

static inline uint32_t FindBestSingleNA8W4(uint32_t baseM_, uint32_t baseN_, uint32_t avg_m, uint32_t maxN_,
                                           uint32_t groupNum_, const uint32_t& aicNum) {
    int32_t mDim = CeilDiv(avg_m, baseM_);
    int32_t nDim = CeilDiv(maxN_, baseN_);
    int32_t taskNum = mDim * nDim * static_cast<int32_t>(groupNum_);
    int32_t taskNumPerCore = CeilDiv(taskNum, aicNum);
    if (taskNumPerCore <= 1) {
        return baseN_;
    }
    int32_t curNDim = 0;
    int32_t curTaskNum = 0;
    int32_t bestSingleN = baseN_;
    float ratio = 0;
    for (uint32_t i = 1; i <= aicNum; ++i) {
        bestSingleN = CeilDiv(static_cast<int32_t>(maxN_), i);
        if (baseN_ != 0 && static_cast<int64_t>(bestSingleN) != maxN_ && bestSingleN % baseN_ != 0) {
            continue;
        }
        curNDim = CeilDiv(maxN_, bestSingleN);
        curTaskNum = mDim * curNDim * static_cast<int32_t>(groupNum_);
        ratio = static_cast<float>(curTaskNum) / AlignUp(static_cast<uint32_t>(curTaskNum), aicNum);
        if (ratio >= EFFECTIVE_TASK_RATIO) {
            return bestSingleN;
        }
    }
    return baseN_;
}

static inline void InitPlatformInfo(const GMMCompileInfo* compileInfoPtr, matmul_tiling::PlatformInfo& platformInfo) {
    platformInfo.socVersion = compileInfoPtr->socVersion;
    platformInfo.l1Size = compileInfoPtr->l1Size;
    platformInfo.l0CSize = compileInfoPtr->l0CSize;
    platformInfo.ubSize = compileInfoPtr->ubSize;
    platformInfo.l0ASize = compileInfoPtr->l0ASize;
    platformInfo.l0BSize = compileInfoPtr->l0BSize;
}
}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_TILING_UTILS_H
