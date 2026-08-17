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
 * \file moe_tutel_combine_gates_tiling.h
 * \brief
 */
#ifndef OPS_TRANSFORMER_MOE_TUTEL_COMBINE_GATES_TILING_H
#define OPS_TRANSFORMER_MOE_TUTEL_COMBINE_GATES_TILING_H

#include "log/log.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/arch35/moe_tutel_combine_gates_tiling_data.h"

namespace optiling {

constexpr size_t MTCG_INPUT_X_INDEX = 0;
constexpr size_t MTCG_INPUT_Y_GRAD_INDEX = 1;
constexpr size_t MTCG_INPUT_INDICES_INDEX = 2;
constexpr size_t MTCG_INPUT_LOCATIONS_INDEX = 3;
constexpr size_t MTCG_OUTPUT_GATES_GRAD_INDEX = 0;

constexpr uint64_t MTCG_TILING_KEY_NOT_CUT_H = 10001UL;
constexpr uint64_t MTCG_TILING_KEY_CUT_H = 10002UL;

struct MoeTutelCombineGatesCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    uint32_t vectorLength = 0;
    uint32_t ubBlockSize = 0;
};

// 单模板 tiling：先按 NOT_CUT_H 试，UB 放不下再退到 CUT_H。
// 分核为二维（sCoreNum × kCoreNum），s 外层 k 内层，让 x 的一行在 k 窗口内复用。
class MoeTutelCombineGatesTiling {
public:
    explicit MoeTutelCombineGatesTiling(gert::TilingContext *context)
        : context_(context)
    {}

    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAndCheck();
    ge::graphStatus CheckDtype();
    ge::graphStatus SplitCore();
    ge::graphStatus SplitUb();
    ge::graphStatus PostTiling();

private:
    gert::TilingContext *context_ = nullptr;

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t vlFp32_ = 0;      // 一个 VREG 能装的 fp32 个数
    int64_t ubBlockSize_ = 0; // UB 对齐粒度（字节）
    int64_t dtypeSize_ = 0;   // 浮点张量的元素字节数

    int64_t samples_ = 0;  // S
    int64_t hidden_ = 0;   // H
    int64_t experts_ = 0;  // E
    int64_t capacity_ = 0; // C

    uint64_t tilingKey_ = MTCG_TILING_KEY_NOT_CUT_H;
    uint32_t blockDim_ = 1;
    MoeTutelCombineGatesTilingData td_;
};

} // namespace optiling
#endif // OPS_TRANSFORMER_MOE_TUTEL_COMBINE_GATES_TILING_H
