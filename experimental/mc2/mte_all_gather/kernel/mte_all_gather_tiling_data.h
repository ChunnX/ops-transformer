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
 * \file mte_all_gather_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef MTE_ALL_GATHER_TILING_DATA_H
#define MTE_ALL_GATHER_TILING_DATA_H

#include <kernel_tiling/kernel_tiling.h>

struct MteAllGatherTilingInfo {
    uint64_t bs;
    uint64_t hiddenSize;
    uint64_t aivNum;
    uint64_t totalWinSize;
};

struct MteAllGatherTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MteAllGatherTilingInfo mteAllGatherTilingInfo;
};

#endif // MTE_ALL_GATHER_TILING_DATA_H
