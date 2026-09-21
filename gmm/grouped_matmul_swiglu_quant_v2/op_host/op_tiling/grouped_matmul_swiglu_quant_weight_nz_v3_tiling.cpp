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
 * \file grouped_matmul_swiglu_quant_weight_nz_v3_tiling.cpp
 * \brief Tiling registration for the Ascend 950 MXFP8 weight-NZ V3 path.
 */

#include "grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h"
#include "op_host/tiling_templates_registry.h"

using namespace optiling::GroupedMatmulSwigluQuantV3Tiling;

namespace optiling {
namespace {
constexpr int32_t V3_950_TILING_TEMPLATE = 5;
} // namespace

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV3Tiling950, V3_950_TILING_TEMPLATE);
} // namespace optiling
