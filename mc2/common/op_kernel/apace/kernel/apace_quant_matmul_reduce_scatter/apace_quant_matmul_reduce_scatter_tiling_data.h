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
 * \file apace_quant_matmul_reduce_scatter_tiling_data.h
 * \brief ReduceScatter (AllToAll PUT + AtomicAdd) MX 量化 Matmul tiling data。
 *
 * tiling 结构体从 host launcher 传到 kernel。
 * ReduceScatter 语义：每 rank 输出 C[M/rankSize, N]，M 轴切分。
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../tiling/quant_matmul_tiling_data.h"
#include "../../tiling/comm_tiling_data.h"
#include "../../block/aiv_comm/collective_comm_context.h"

// ReduceScatter tiling 结构体（UDMA 模式）
// 与 all_to_all_quant_matmul 的 allToAllMatmulTilingData 结构一致，
// 差异在于 Host 侧填充时 m=M/rankSize、commTilingData 基于 M/rankSize 切分。
struct ApaceQuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;
    CommTilingData scaleCommTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    uint32_t localMatmul{0}; // 0:REMOTE-only; 1:LOCAL+REMOTE+AtomicAdd; 2:DEFERRED_SYNC
};

struct CommContext {
    Apace::AivComm::CommUdmaContext udmaCtx;
    Apace::AivComm::CommUbmemContext ubmemCtx;
};