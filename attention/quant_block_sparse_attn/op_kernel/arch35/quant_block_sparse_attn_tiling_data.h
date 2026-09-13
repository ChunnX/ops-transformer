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
 * \file quant_block_sparse_attn_tiling_data.h
 * \brief Kernel-side metadata layout for quant block sparse attention.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_
#define QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_

#include <cstdint>

namespace optiling {
// AICPU metadata format: head[8] + qbsa[sectionNum][36][8] + fd[72][8].
// Keep these field indices aligned with quant_block_sparse_attn_metadata's writer.
constexpr uint32_t QBSA_FA_HEAD_METADATA_SIZE = 8U;
constexpr uint32_t QBSA_FA_HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t QBSA_FA_AIC_CORE_NUM = 36U;
constexpr uint32_t QBSA_FA_AIV_CORE_NUM = 72U;
constexpr uint32_t QBSA_FA_METADATA_SIZE = 8U;
constexpr uint32_t QBSA_FA_FD_METADATA_SIZE = 8U;

constexpr uint32_t QBSA_FA_CORE_ENABLE_INDEX = 0U;
constexpr uint32_t QBSA_FA_BN1_START_INDEX = 1U;
constexpr uint32_t QBSA_FA_S1_START_INDEX = 2U;
constexpr uint32_t QBSA_FA_S2_START_INDEX = 3U;
constexpr uint32_t QBSA_FA_BN1_END_INDEX = 4U;
constexpr uint32_t QBSA_FA_S1_END_INDEX = 5U;
constexpr uint32_t QBSA_FA_S2_END_INDEX = 6U;
constexpr uint32_t QBSA_FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7U;
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_DATA_H_
