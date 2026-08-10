/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
 * PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_mx_matmul_ccu_tiling_data.h
 * \brief Tiling data for AllGather + QuantMatmul fusion kernel (CCU variant)
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/tiling/comm_tiling_data.h"

struct ccuAllGatherMxMatmulTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CommTilingData commTile;
    CommTilingData scaleCommTile;
    QuantMatmulTilingData mmTile;
};
