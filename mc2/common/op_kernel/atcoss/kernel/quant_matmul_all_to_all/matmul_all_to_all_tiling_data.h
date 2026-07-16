/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file matmul_all_to_all_tiling_data.h
 * \brief Serialized tiling data passed from the host launcher to the kernel.
 */

#pragma once

#include <cstdint>
#include "tiling/quant_matmul_tiling_data.h"
#include "tiling/comm_tiling_data.h"

struct AllToAllCommTilingData {
    CommTilingData commTilingData;
    uint32_t bufferSize{0}; // 内存复用场景
};

struct MatmulAllToAllTilingData {
    AllToAllCommTilingData alltoallTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
    QuantMatmulTilingData localQbmmTilingData;
};
