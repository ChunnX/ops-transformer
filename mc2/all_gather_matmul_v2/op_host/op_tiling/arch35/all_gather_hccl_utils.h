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
 * \file all_gather_hccl_utils.h
 * \brief HCCL limit adjustment utilities for all_gather_matmul_v2 (arch35)
 */

#ifndef ALL_GATHER_HCCL_UTILS_H
#define ALL_GATHER_HCCL_UTILS_H

#include <cstdint>
#include <string>
#include "op_host/op_tiling/formulaic_tiling_datatype.h"

namespace optiling {

constexpr uint64_t HCCL_MEM_LIMIT = 268435456;  // 256MB
constexpr uint64_t HCCL_MAX_TILE_CNT = 8;        // 最大切分轮次
constexpr uint64_t ALIGN_LEN = 256;              // tileM对齐长度（性能优化）

void AdjustCutResultForHCCL(CutResult& cutRes,
                            uint64_t mValue, uint64_t kValue,
                            uint64_t dtypeSize, uint64_t rankDim,
                            const std::string& opName);

}  // namespace optiling

#endif  // ALL_GATHER_HCCL_UTILS_H