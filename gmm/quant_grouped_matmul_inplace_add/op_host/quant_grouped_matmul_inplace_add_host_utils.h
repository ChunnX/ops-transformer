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
 * \file quant_grouped_matmul_inplace_add_host_utils.h
 * \brief
 */

#ifndef QUANT_GROUPED_MATMUL_INPLACE_ADD_UTILS_H
#define QUANT_GROUPED_MATMUL_INPLACE_ADD_UTILS_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>

#include "../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"

namespace QuantGroupedMatmulInplaceAdd {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t GROUPLIST_INDEX = 3;
constexpr uint32_t PER_TOKEN_SCALE_INDEX = 5;
constexpr uint32_t Y_INDEX = 0;
constexpr uint64_t TILING_KEY = 0;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 0;

// QuantGroupedInplaceAddTiling 和 QGMMInplaceAddBasicApiTiling 共用的校验/解析函数。
// 抽成自由函数避免两条继承路径复制逻辑，同时保持两边继承链独立（前者继承 GroupedQmmTiling，
// 后者继承 GroupedQmmBasicApiTiling）。

bool AnalyzeAttrsForInplaceAdd(gert::TilingContext *context, optiling::GQmmInputInfo &inputParams);

// 只读取 GQmmInputInfo 的 dtype 字段，合法性校验由 CheckDtypeForInplaceAdd 负责
bool AnalyzeDtypeForInplaceAdd(gert::TilingContext *context, optiling::GQmmInputInfo &inputParams);

// allowFp8 为 false 时仅允许 HIFLOAT8+HIFLOAT8（BasicApi 的 T-T 路径）；
// 为 true 时额外允许 FP8_E4M3FN/FP8_E5M2 组合（非 BasicApi 路径）
bool CheckDtypeForInplaceAdd(const optiling::GQmmInputInfo &inputParams, bool allowFp8);

bool CheckCoreNumForInplaceAdd(gert::TilingContext *context, const optiling::GQmmInputInfo &inputParams);

// T-T 量化：scale1/scale2 为 (g,) 或 (g, 1)
bool CheckShapeForTTQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const optiling::GQmmInputInfo &inputParams);

// MX 量化：scale1/scale2 均为 3D，末轴 = MXFP_MULTI_BASE_SIZE
bool CheckShapeForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const optiling::GQmmInputInfo &inputParams);

// T-C 量化：scale1 为 (g,) 或 (g, 1)，scale2 为 (g, n)
bool CheckShapeForTCQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const optiling::GQmmInputInfo &inputParams);
} // namespace QuantGroupedMatmulInplaceAdd
#endif
