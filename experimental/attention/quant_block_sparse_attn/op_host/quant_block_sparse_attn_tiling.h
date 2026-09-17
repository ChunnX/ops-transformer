/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TILING_H_
#define QUANT_BLOCK_SPARSE_ATTN_TILING_H_

#include <cstddef>

#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include <register/tilingdata_base.h>
#include "../op_kernel/arch35/common/quant_block_sparse_attn_tiling_data.h"
#include "tiling_common/qbsa_op_contract.h"

namespace optiling {
// This is reserved buffer capacity, not compatibility with an older tiling layout.
// Host and Kernel must be rebuilt together from the shared tiling data header.
constexpr std::size_t QBSA_REGISTERED_TILING_DATA_CAPACITY = 344U;
static_assert(sizeof(QuantBlockSparseAttnTilingData) <= QBSA_REGISTERED_TILING_DATA_CAPACITY,
              "QBSA tiling data exceeds its registered runtime capacity");

BEGIN_TILING_DATA_DEF(QuantBlockSparseAttnRegisteredTilingData)
TILING_DATA_FIELD_DEF_ARR(uint8_t, 344, payload);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QuantBlockSparseAttn, QuantBlockSparseAttnRegisteredTilingData)

ge::graphStatus TilingQuantBlockSparseAttn(gert::TilingContext *context);
} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_H_
