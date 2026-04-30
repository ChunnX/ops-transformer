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
 * \file matmul_v3_tuning.h
 * \brief Stub version - tuning registry not available in single-op build
 */
#ifndef MATMUL_V3_TUNING_H__
#define MATMUL_V3_TUNING_H__

#include <cstdint>
#include <memory>
#include "graph/types.h"

namespace tuningtiling {

struct MatMulV3InputArgs {
  int64_t m;
  int64_t k;
  int64_t n;
  int64_t batch_a1;
  int64_t batch_a2;
  int64_t batch_a3;
  int64_t batch_a4;
  int64_t batch_b1;
  int64_t batch_b2;
  int64_t batch_b3;
  int64_t batch_b4;
  float l1_fused_num;
  float aub_double_num;
  float bub_double_num;
  float fused_double_operand_num;
  ge::DataType a_dtype;
  ge::DataType b_dtype;
  ge::DataType out_dtype;
  ge::Format a_format;
  ge::Format b_format;
  ge::Format out_format;
  bool trans_a_flag;
  bool trans_b_flag;
  bool bias_flag;
  bool reserved_bool;
  bool m_align_flag;
  bool k_align_flag;
  bool n_align_flag;
  uint64_t reserved_params1;
  uint64_t reserved_params2;
  uint64_t reserved_params3;
  uint64_t reserved_params4;
  uint64_t reserved_params5;
  uint64_t reserved_params6;
};

// Stub tuning tiling def - provides struct layout without registry macros
struct MatMulV3TunnerTiling {
  uint32_t usedCoreNum;
  uint32_t singleCoreM;
  uint32_t singleCoreN;
  uint32_t singleCoreK;
  uint32_t baseM;
  uint32_t baseN;
  uint32_t baseK;
  uint32_t depthA1;
  uint32_t depthB1;
  uint32_t stepM;
  uint32_t stepN;
  uint32_t iterateOrder;
  uint32_t stepKa;
  uint32_t stepKb;
  uint32_t dbL0A;
  uint32_t dbL0B;
  uint32_t dbL0C;
  uint32_t l2MTileCnt;
  uint32_t l2NTileCnt;
  uint32_t l2MTileBlock;
  uint32_t l2NTileBlock;
  uint32_t l2IterateOrder;
  uint32_t tilingEnable;
};

using TuningTilingDefPtr = std::shared_ptr<uint8_t>;

} // namespace tuningtiling

#endif
