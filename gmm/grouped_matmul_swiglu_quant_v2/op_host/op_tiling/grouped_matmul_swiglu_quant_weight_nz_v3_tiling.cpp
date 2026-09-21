/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Tiling registration for the Ascend 950 MXFP8 weight-NZ V3 path.
 */

#include "grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h"
#include "op_host/tiling_templates_registry.h"

using namespace optiling::GroupedMatmulSwigluQuantV3Tiling;

namespace optiling {
namespace {
constexpr int32_t V3_950_TILING_TEMPLATE = 5;
} // namespace

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2,
                             GroupedMatmulSwigluQuantV3Tiling950,
                             V3_950_TILING_TEMPLATE);
} // namespace optiling
