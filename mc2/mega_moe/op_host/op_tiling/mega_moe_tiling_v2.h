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
 * \file mega_moe_tiling_v2.h
 * \brief
 */

#ifndef MEGA_MOE_TILING
#define MEGA_MOE_TILING

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"

namespace optiling {

struct MegaMoeConfig {
    uint32_t contextIndex = 0U;  // 0: 根据mega_moe算子原型标志位初始化context索引
    uint32_t xIndex = 0U; // 0: 根据mega_moe算子原型标志位初始化groupEp索引
    uint32_t expertIdsIndex = 1U; // 1: 根据mega_moe算子原型标志位初始化expertIds索引
    uint32_t weightOneIndex = 2U; // 2: 根据mega_moe算子原型标志位初始化weight1索引
    uint32_t weightScalesOneIndex = 3U; // 3: 根据mega_moe算子原型标志位初始化weightScales1索引
    uint32_t weightTwoIndex = 4U; // 4: 根据mega_moe算子原型标志位初始化weight2索引
    uint32_t weightScalesTwoIndex = 5U; // 5: 根据mega_moe算子原型标志位初始化weightScales2索引
    uint32_t attrGroupEpIndex = 0; // 0: 根据mega_moe算子原型标志位初始化groupEp索引
    uint32_t attrEpWorldSizeIndex = 1; // 1: 根据mega_moe算子原型标志位初始化epWorldSize索引
    uint32_t attrMaxOutputSizeIndex = 2; // 2: 根据mega_moe算子原型标志位初始化maxOutputSize索引
    bool isMc2Context = false;
};
using namespace Ops::Transformer::OpTiling;

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext* context, MegaMoeConfig& config);

}

#endif