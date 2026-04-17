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
 * \file bandwidth_test_tiling.h
 * \brief Bandwidth test tiling data structure
 */
#ifndef BANDWIDTH_TEST_TILING_H
#define BANDWIDTH_TEST_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct BandwidthTestTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    int32_t bs;
    int32_t data_size;
    int32_t world_size;
    int32_t rank_id;
    int32_t max_bs;
    int32_t mode;
    int32_t aiv_num;
    uint64_t total_ub_size;
    uint64_t total_win_size;
    int32_t warmup_iterations;
    int32_t test_iterations;
};

#endif // BANDWIDTH_TEST_TILING_H
