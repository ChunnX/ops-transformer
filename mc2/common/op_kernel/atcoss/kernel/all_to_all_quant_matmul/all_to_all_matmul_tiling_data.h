/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_to_all_matmul_tiling_data.h
 * \brief Serialized tiling data passed from the host launcher to the kernel.
 */

#pragma once

#include <cstdint>
#include "../quant_matmul_tiling_data.h"

struct AllToAllCommTilingData {
    uint32_t tileCnt{0};
    uint32_t bufferSize{0};
    uint32_t localMatmul{0};
    uint32_t headTileCnt{0};
    uint32_t headMSize{0};
    uint32_t tailTileCnt{0};
    uint32_t tailMSize{0};
};

struct allToAllMatmulTilingData {
    AllToAllCommTilingData commTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
    QuantMatmulTilingData localQbmmTilingData;
};
