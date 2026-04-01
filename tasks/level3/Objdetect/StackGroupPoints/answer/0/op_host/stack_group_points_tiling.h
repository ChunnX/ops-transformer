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
 * \file stack_group_points_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_STACK_GROUP_POINTS_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_STACK_GROUP_POINTS_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{

    struct StackGroupPointsCompileInfo
    {
        int32_t totalCoreNum = 0;
        int64_t ubSize = 0;
    };

    BEGIN_TILING_DATA_DEF(StackGroupPointsTilingData)
    TILING_DATA_FIELD_DEF(int64_t, b);
    TILING_DATA_FIELD_DEF(int64_t, m);
    TILING_DATA_FIELD_DEF(int64_t, c);
    TILING_DATA_FIELD_DEF(int64_t, nsample);
    TILING_DATA_FIELD_DEF(int64_t, res);
    TILING_DATA_FIELD_DEF(int64_t, featuresSize);
    TILING_DATA_FIELD_DEF(int64_t, indicesSize);
    TILING_DATA_FIELD_DEF(int64_t, fbcSize);
    TILING_DATA_FIELD_DEF(int64_t, ibcSize);
    TILING_DATA_FIELD_DEF(int64_t, reminder);
    TILING_DATA_FIELD_DEF(int64_t, outLength);
    TILING_DATA_FIELD_DEF(int64_t, n);
    TILING_DATA_FIELD_DEF(int64_t, standard);
    TILING_DATA_FIELD_DEF(int64_t, actCore);

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(StackGroupPoints, StackGroupPointsTilingData)
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_STACK_GROUP_POINTS_TILING_H_