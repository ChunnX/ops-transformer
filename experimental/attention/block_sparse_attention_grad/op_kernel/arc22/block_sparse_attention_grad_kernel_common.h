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
 * \file block_sparse_attention_grad_kernel_common.h
 * \brief Block Sparse Attention Grad Kernel Common
 */

#ifndef BSAG_KERNEL_COMMON_H
#define BSAG_KERNEL_COMMON_H

// Host scheduling and the device-side wide metadata arrays must use the
// same logical-Q capacity.
constexpr uint32_t BSA_WIDE_Q_TASK_CAPACITY = 48;
constexpr uint32_t BSA_WIDE_PACKET_EDGE_CAPACITY = 8;

constexpr uint32_t CUBE2VEC = 7;
constexpr uint32_t VEC2CUBE = 8;
constexpr uint32_t CUBE2POST = 9;

constexpr uint64_t WORKSPACE_BLOCK_SIZE = 128 * 128 * 2;
constexpr uint64_t WORKSPACE_P16_OFFSET_ELEMENT = 128 * 128 * 2; // dp, P 的在workspace相对起始地址后偏移元素个数
constexpr uint64_t WORKSPACE_P16_OFFSET = 128 * 128 * 4; // dp, P 的在workspace相对起始地址后偏移字节数
constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 2 * 2;
constexpr uint32_t GROUPED_Q_BLOCKS = 4;
constexpr uint32_t GROUPED_ACTIVE_Q_BLOCKS = GROUPED_Q_BLOCKS;
constexpr uint32_t GROUPED_KV_BLOCKS = 4;
constexpr uint32_t GROUPED_PACKED_FRAGMENTS_PER_Q = 4;
constexpr uint32_t GROUPED_TILES = GROUPED_ACTIVE_Q_BLOCKS * GROUPED_KV_BLOCKS;
// A normal-path packet is bounded by active edges, not by a fixed number
// of KV columns.  In the most sparse case every edge can belong to a
// different column, so the unique-KV bound equals the tile capacity.
constexpr uint32_t GROUPED_PACKET_KV_BLOCKS = GROUPED_TILES;
constexpr uint64_t GROUPED_TILE_ELEMENTS = 128 * 128;
constexpr uint64_t GROUPED_WORKSPACE_STAGE_SIZE = GROUPED_TILE_ELEMENTS * GROUPED_TILES;
constexpr uint64_t GROUPED_WORKSPACE_CORE_SIZE = GROUPED_WORKSPACE_STAGE_SIZE * 2;
// One wide work item supports at most 48 internal Q tasks of up to 128 rows.
// The two-stage pipeline has eight edges per packet; dense columns resume
// through the (KV,Q) cursor without requiring a larger workspace.
constexpr uint32_t WIDE_MAX_Q_TASKS = BSA_WIDE_Q_TASK_CAPACITY;
constexpr uint32_t WIDE_MAX_EDGES = BSA_WIDE_PACKET_EDGE_CAPACITY;
// dQ owns eight persistent K slots.  Edge capacity may grow independently,
// but a physical packet must never require more resident KV columns.
constexpr uint32_t WIDE_MAX_KV_COLUMNS = 8;
using WideQMask = uint64_t;
static_assert(WIDE_MAX_Q_TASKS <= sizeof(WideQMask) * 8U,
              "wide Q state mask is too narrow for the configured capacity");
static_assert(WIDE_MAX_Q_TASKS <= (1U << 8), "wide edgeCode reserves only eight bits for qLocal");
static_assert(WIDE_MAX_EDGES < (1U << 8), "wide Q descriptors reserve only eight bits for edge offsets");
static_assert(WIDE_MAX_KV_COLUMNS <= WIDE_MAX_EDGES, "wide KV column capacity cannot exceed packet edge capacity");
static_assert(WIDE_MAX_KV_COLUMNS <= 8, "wide dQ has only eight persistent K cache slots");
static_assert(WIDE_MAX_KV_COLUMNS <= (1U << 16), "wide edgeCode reserves only sixteen bits for kvLocal");
constexpr uint32_t WIDE_CUBE2VEC_STAGE0 = 5;
constexpr uint32_t WIDE_CUBE2VEC_STAGE1 = 6;
constexpr uint32_t WIDE_VEC2CUBE_STAGE0 = 7;
constexpr uint32_t WIDE_VEC2CUBE_STAGE1 = 8;
constexpr uint64_t L1_SIZE_OFFSET = 131072;
constexpr uint32_t PINGPONG_OFFSET_2 = 2;
constexpr uint32_t PINGPONG_OFFSET_4 = 4;

#endif // BSAG_KERNEL_COMMON_H
