/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_tiling_data.h
 * \brief StemOamPrepPagedKV TilingData struct definition (arch35)
 */
#ifndef STEM_OAM_PREP_PAGED_KV_TILING_DATA_H
#define STEM_OAM_PREP_PAGED_KV_TILING_DATA_H

#include "kernel_operator.h"

constexpr uint32_t DIM_QK = 128;
constexpr uint32_t KV_INDICES_BATCH_SIZE = 64;
constexpr uint32_t MAX_STEM_BLOCK_SIZE = 256;
constexpr uint32_t MAX_STEM_STRIDE = 64;
constexpr uint32_t MAX_R_VAL = MAX_STEM_BLOCK_SIZE / 16;
constexpr uint32_t MAX_KFLAT_DIM = MAX_STEM_STRIDE * DIM_QK;
constexpr uint32_t DENSE_ROW_ELEM = DIM_QK;
constexpr uint32_t REDUCE_TMP_SIZE = 9216;

struct StemPrepKVTilingData {
    uint32_t batch_size = 0;
    uint32_t num_kv_heads = 0;
    uint32_t dim_qk = DIM_QK;
    uint32_t dim_v = DIM_QK;
    uint32_t stem_block_size = 128;
    uint32_t stem_stride = 16;
    uint32_t R_val = 8;
    uint32_t kflat_dim = 2048;
    uint32_t kv_block_size = 64;
    uint32_t kv_layout = 0;
    uint32_t total_blocks = 0;
    uint32_t max_k_padded = 0;
    uint32_t max_Kb = 0;
    uint32_t max_kv_blocks = 0;
    uint32_t vbias_stride = 0;
    float lambda_mag = 0.3f;
    float epsilon = 1e-6f;
    uint32_t total_tasks = 0;
    uint32_t used_core_num = 0;
    uint32_t split_mode = 0;
    uint32_t cores_per_bh = 1;
    uint32_t total_stem_blocks = 0;
    uint32_t stem_blocks_per_core = 1;
    uint32_t kv_blocks_per_stem_block = 1;
};

#endif // STEM_OAM_PREP_PAGED_KV_TILING_DATA_H