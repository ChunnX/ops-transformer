/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEM_OAM_PREP_VARLEN_Q_TILING_DATA_H
#define STEM_OAM_PREP_VARLEN_Q_TILING_DATA_H

#include <cstdint>

struct StemPrepQTilingData {
    uint32_t batch_size;
    uint32_t num_q_heads;
    uint32_t dim_qk;
    uint32_t stem_block_size;
    uint32_t stem_stride;
    uint32_t R_val;
    uint32_t kflat_dim;
    uint32_t max_Qb;
    uint32_t used_core_num;
    uint32_t blocks_per_core_base;
    uint32_t blocks_remainder;
};

#endif
