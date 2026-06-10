/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEM_OAM_PREP_VARLEN_Q_TILING_DATA_H_
#define STEM_OAM_PREP_VARLEN_Q_TILING_DATA_H_

#include <cstdint>

constexpr uint32_t DIM_QK = 128;
constexpr uint32_t MAX_STEM_BLOCK_SIZE = 256;
constexpr uint32_t MAX_STEM_STRIDE = 64;
constexpr uint32_t MAX_QFLAT_DIM = MAX_STEM_STRIDE * DIM_QK;
constexpr uint32_t VARLEN_READ_ALIGN = 8;
constexpr uint32_t MAX_CORE_NUM = 64;
constexpr uint32_t MAX_BATCH_SIZE = 256;

struct StemPrepQTilingData {
    uint32_t batch_size = 0;
    uint32_t num_q_heads = 0;
    uint32_t dim_qk = DIM_QK;
    uint32_t stem_block_size = 128;
    uint32_t stem_stride = 16;
    uint32_t R_val = 8;
    uint32_t kflat_dim = MAX_QFLAT_DIM;
    uint32_t max_q_padded = 0;
    uint32_t max_Qb = 0;
    uint32_t total_tokens = 0;
    uint32_t total_tasks = 0;
    uint32_t used_core_num = 0;
    uint32_t split_mode = 0;
    uint32_t cores_per_bh_base = 1;
    uint32_t remainder_bh = 0;
    uint32_t total_blocks = 0;
    uint32_t blocks_per_core_base = 0;
    uint32_t blocks_remainder = 0;
    uint32_t num_Qb_per_batch[MAX_BATCH_SIZE] = {};
    uint32_t core_batch_idx[MAX_CORE_NUM] = {};
    uint32_t core_head_idx[MAX_CORE_NUM] = {};
    uint32_t core_block_idx[MAX_CORE_NUM] = {};
    uint32_t core_block_num[MAX_CORE_NUM] = {};
};

#endif