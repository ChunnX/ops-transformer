/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_a2_constant.h
 * \brief Define the constant for the A2 Dispatch and Combine operations.
 */
#ifndef MOE_DISTRIBUTE_A2_CONSTANT_H
#define MOE_DISTRIBUTE_A2_CONSTANT_H
namespace Mc2A2Kernel {
// ADump所需常量段
constexpr uint32_t BUFFERID_POS = 0UL;
constexpr uint32_t OPOSITION_POS = 1U;
constexpr uint32_t TILING_EPRANKID_POS = 2U;
constexpr uint32_t MOE_NUM_POS = 3U;
constexpr uint32_t TILING_WORLDSIZE_POS = 4U;
constexpr uint32_t GLOBALBS_POS = 5U;
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint32_t HCCL_DFX_POS = 8U;
constexpr uint32_t HCCL_DFX_NUM = 2U;
constexpr uint32_t HCCL_EPRANKId_POS = 0U;
constexpr uint32_t HCCL_WORLDSIZE_POS = 1U;
constexpr uint32_t ISLAYERED_POS = 11U;
constexpr uint32_t AIVNUM_POS = 12U;
constexpr uint32_t AIV_TASK_FIRST_EPRANKID_POS = 13U;
constexpr uint32_t AIV_TASK_EPRANKNUM_POS = 14U;
constexpr uint32_t AIV_TASK_ARRIVED_EPRANKID_START_POS = 15U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t UB_ALIGN = 0U;

constexpr uint32_t RUNPOS_INIT = 1U;
// A2 Fullmesh
constexpr uint32_t RUNPOS_REORDER_TOKEN = 2U;
constexpr uint32_t RUNPOS_SEND_DATA_TO_SERVER = 3U;
constexpr uint32_t RUNPOS_CREATE_INNER_REDUCE_INFO = 4U;
constexpr uint32_t RUNPOS_CREATE_OUTER_REDUCE_INFO = 5U;
constexpr uint32_t RUNPOS_WIN2IPC = 6U;
constexpr uint32_t RUNPOS_SET_IPC_FLAG = 7U;
constexpr uint32_t RUNPOS_WAIT_IPC_FLAG = 8U;
constexpr uint32_t RUNPOS_IPC2OUT = 9U;
constexpr uint32_t RUNPOS_CLEANUP = 10U;
constexpr uint32_t RUNPOS_COPY_PERFORMANCE_INFO = 11U;

}


#endif
