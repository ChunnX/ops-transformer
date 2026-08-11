/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_attn_res_update_tiling_data.h
 * \brief BlockAttnResUpdate arch35 tiling data definition.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_TILING_DATA_H
#define BLOCK_ATTN_RES_UPDATE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct BlockAttnResUpdateTilingData {
    uint32_t reserved = 0;
};
#pragma pack(pop)

#endif // BLOCK_ATTN_RES_UPDATE_TILING_DATA_H

