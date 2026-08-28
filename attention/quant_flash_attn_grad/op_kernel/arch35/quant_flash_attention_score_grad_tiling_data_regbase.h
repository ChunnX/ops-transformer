/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_tiling_data_rebase.h
 * \brief
 */

// test 0818

#ifndef QUANT_FLASH_ATTENTION_SCORE_GRAD_TILING_DATA_REGBASE_H_
#define QUANT_FLASH_ATTENTION_SCORE_GRAD_TILING_DATA_REGBASE_H_

#include <cstdint>

namespace optiling {
namespace QuantFag {

class QuantFlashAttnGradTilingData {
public:
    // ===== common =====
    int64_t b = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t g = 0;
    int64_t d = 0;
    int64_t t1 = 0;
    int64_t n2 = 0;
    int64_t n1 = 0;
    int64_t t2 = 0;
    int64_t s1_outer = 0;
    int64_t s2_outer = 0;
    int64_t s1_tail = 0;
    int64_t s2_tail = 0;
    float softmax_scale = 1.0f;
    bool has_seq_used_q = false;
    bool has_seq_used_k = false;
    // ===== workspace offsets =====
    int64_t dq_work_space_offset = 0;
    int64_t dk_work_space_offset = 0;
    int64_t dv_work_space_offset = 0;
    int64_t sfmg_work_space_offset = 0;
    // ==== pre ====
    int64_t q_pre_block_factor = 0;
    int64_t q_pre_block_total = 0;
    int64_t q_pre_block_tail = 0;
    int64_t k_pre_block_factor = 0;
    int64_t k_pre_block_total = 0;
    int64_t k_pre_block_tail = 0;
    int64_t v_pre_block_factor = 0;
    int64_t v_pre_block_total = 0;
    int64_t v_pre_block_tail = 0;
    // ==== softmax grad ====
    int64_t sfmg_used_core_num = 0;
    int64_t sfmg_dy_buffer_len = 0;
    int64_t sfmg_y_buffer_len = 0;
    int64_t sfmg_output_buffer_len = 0;
    int64_t single_loop_nburst_num = 0;
    int64_t normal_core_loop_times = 0;
    int64_t tail_core_loop_times = 0;
    int64_t normal_core_last_loop_nburst_num = 0;
    int64_t tail_core_last_loop_nburst_num = 0;
    int64_t normal_core_nburst_nums = 0;
    int64_t tail_core_nburst_nums = 0;
    int64_t normal_axis_size = 0;
    // ==== post ====
    int64_t q_post_block_factor = 0;
    int64_t q_post_block_total = 0;
    int64_t q_post_base_num = 0;
    int64_t q_post_tail_num = 0;
    int64_t k_post_block_factor = 0;
    int64_t k_post_block_total = 0;
    int64_t k_post_base_num = 0;
    int64_t k_post_tail_num = 0;
    int64_t v_post_block_factor = 0;
    int64_t v_post_block_total = 0;
    int64_t v_post_base_num = 0;
    int64_t v_post_tail_num = 0;
    int64_t core_num = 0;
};
} // namespace QuantFag
} // namespace optiling

#endif