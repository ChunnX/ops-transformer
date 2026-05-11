/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_tiling_index.h
 * \brief
 */

#ifndef FLASH_ATTN_TILING_INDEX_H
#define FLASH_ATTN_TILING_INDEX_H
#include "register/tilingdata_base.h"

namespace optiling {
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t BLOCK_TABLE_INDEX = 3;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 4;
constexpr uint32_t CU_SEQLENS_KV_INDEX = 5;
constexpr uint32_t SEQUSED_Q_INDEX = 6;
constexpr uint32_t SEQUSED_KV_INDEX = 7;
constexpr uint32_t SINKS_INDEX = 8;
constexpr uint32_t ATTN_MASK_INDEX = 9;
constexpr uint32_t METADATA_INDEX = 10;

// Attributes Index
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 0;  // softmax_mode (scaleValue)
constexpr uint32_t ATTR_MASK_MODE_INDEX = 1;      // mask_mode
constexpr uint32_t ATTR_WIN_LEFT_INDEX = 2;       // win_left (preToken)
constexpr uint32_t ATTR_WIN_RIGHT_INDEX = 3;      // win_right (nextToken)
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 4;   // max_seqlen_q
constexpr uint32_t ATTR_MAX_SEQLEN_KV_INDEX = 5;  // max_seqlen_kv
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 6;       // layout_q
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 7;      // layout_kv
constexpr uint32_t ATTR_LAYOUT_OUT_INDEX = 8;     // layout_out
constexpr uint32_t ATTR_RETURN_LSE_INDEX = 9;     // return_softmax_lse
constexpr uint32_t ATTR_DETERMINISTIC_INDEX = 10; // deterministic

// Legacy aliases for backward compatibility
constexpr uint32_t ATTR_SCALE_INDEX = ATTR_SOFTMAX_SCALE_INDEX;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = ATTR_WIN_LEFT_INDEX;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = ATTR_WIN_RIGHT_INDEX;
constexpr uint32_t SOFTMAX_LSE_FLAG_INDEX = ATTR_RETURN_LSE_INDEX;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = ATTR_LAYOUT_Q_INDEX;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = CU_SEQLENS_Q_INDEX;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = CU_SEQLENS_KV_INDEX;

// Output Index
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;
} // namespace optiling

#endif // FLASH_ATTN_TILING_INDEX_H