/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __GROUPED_MATMUL_ADD_TILING_DEF_H__
#define __GROUPED_MATMUL_ADD_TILING_DEF_H__

#include "kernel_tiling/kernel_tiling.h"
// #include "tiling/tiling_api.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

constexpr uint16_t GMM_MAX_TENSOR_LIST_SIZE = 128;


inline void InitGroupedMatmulAddTilingData(uint8_t* tiling, GroupedMatmulAddTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(GroupedMatmulAddTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    GroupedMatmulAddTilingData tilingData;         \
    InitGroupedMatmulAddTilingData(tilingPointer, &tilingData)
#endif
