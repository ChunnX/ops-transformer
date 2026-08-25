/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../../tiling/quant_matmul_tiling_data.h"
#include "../../../tiling/comm_tiling_data.h"

struct MatmulAllToAllApaceTilingInfo {
    uint32_t tileM{0};
    uint32_t tileCnt{0};
    uint32_t tailM{0};
    uint32_t tailCnt{0};
    uint32_t rankM{0};
    uint32_t rankN{0};
    uint32_t rankK{0};
    uint32_t rankDim{0};
    uint64_t hcclDataType{0};
};

struct hcommAllToAllMatmulTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MatmulAllToAllApaceTilingInfo matmulAllToAllTilingInfo;
    QuantMatmulTilingData tileQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
};
