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
 * \file all_gather_matmul_tiling_data.h
 * \brief Tiling data for AllGather + QuantMatmul fusion kernel
 */

#pragma once

#include <cstdint>
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/block/aiv_comm/collective_comm_context.h"

namespace Apace {
namespace AivComm {
// Convenience wrapper so the kernel receives one pointer instead of two.
struct CommContext {
    CommUdmaContext udmaCtx;
    CommUbmemContext ubmemCtx;
};
} // namespace AivComm
} // namespace Apace

using Apace::AivComm::CommContext;

#pragma pack(push, 8)
struct alignas(8) AllGatherMatmulTilingData {
  // Per-tile matmul tiling (M=tileM, batch=rankSize-1)
  QuantMatmulTilingData mmTile;
  // Tail matmul tiling (M=tailM, batch=rankSize-1), only used when tailCnt > 0
  QuantMatmulTilingData mmTail;
  // Local matmul tiling (M=full m, batch=1)
  QuantMatmulTilingData mmLocal;

  // Problem dimensions
  uint32_t m{0};          // M per rank
  uint32_t k{0};          // K (full)
  uint32_t n{0};          // N (full)
  uint32_t rankNum{0};    // Number of ranks

  // Tiling control
  uint32_t tileCnt{0};    // Number of head tiles
  uint32_t tileM{0};      // M per head tile
  uint32_t tailCnt{0};    // Tail tile count (0 or 1)
  uint32_t tailM{0};      // M for tail tile
  uint32_t commTurn{0};   // Total communication rounds (= tileCnt + tailCnt)
  // paddedTailM: physical rows per slot in the tail band = ceil(tailM, 16).
  uint32_t paddedTailM{0};
};
#pragma pack(pop)
