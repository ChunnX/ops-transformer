/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TILING_DEF_H
#define MEGA_MOE_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../../op_kernel/mega_moe_tiling_def.h"

#define GET_TILING_DATA(tilingData, tilingArg)                                                        \
    MegaMoeTilingData tilingData;                                                                       \
    memcpy(&tilingData, tilingArg, sizeof(MegaMoeTilingData))

#define GET_TILING_DATA_WITH_STRUCT(TilingDataStru, tilingData, tilingArg)       \
    TilingDataStru tilingData;                                                 \
    memcpy(&tilingData, tilingArg, sizeof(TilingDataStru))

#endif