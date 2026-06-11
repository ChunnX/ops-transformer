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
#include "register/tilingdata_base.h"

constexpr uint32_t MAX_CORE_NUM_HOST = 64;
constexpr uint32_t MAX_BATCH_SIZE_HOST = 256;

namespace optiling {

struct StemPrepQCompileInfo {
    int32_t core_num;
    int32_t ub_size;
};

BEGIN_TILING_DATA_DEF(StemPrepQTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch_size)
TILING_DATA_FIELD_DEF(uint32_t, num_q_heads)
TILING_DATA_FIELD_DEF(uint32_t, dim_qk)
TILING_DATA_FIELD_DEF(uint32_t, stem_block_size)
TILING_DATA_FIELD_DEF(uint32_t, stem_stride)
TILING_DATA_FIELD_DEF(uint32_t, R_val)
TILING_DATA_FIELD_DEF(uint32_t, kflat_dim)
TILING_DATA_FIELD_DEF(uint32_t, max_q_padded)
TILING_DATA_FIELD_DEF(uint32_t, max_Qb)
TILING_DATA_FIELD_DEF(uint32_t, total_tokens)
TILING_DATA_FIELD_DEF(uint32_t, total_tasks)
TILING_DATA_FIELD_DEF(uint32_t, used_core_num)
TILING_DATA_FIELD_DEF(uint32_t, split_mode)
TILING_DATA_FIELD_DEF(uint32_t, cores_per_bh_base)
TILING_DATA_FIELD_DEF(uint32_t, remainder_bh)
TILING_DATA_FIELD_DEF(uint32_t, total_blocks)
TILING_DATA_FIELD_DEF(uint32_t, blocks_per_core_base)
TILING_DATA_FIELD_DEF(uint32_t, blocks_remainder)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(StemOamPrepVarlenQ, StemPrepQTilingData)

}  // namespace optiling