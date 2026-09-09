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
 * \file moe_tutel_combine_x_tiling.h
 * \brief Tiling data class and compile info of MoeTutelCombineX.
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_H

#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "log/log.h"

#include "platform/platform_infos_def.h"
#include "util/math_util.h"

namespace optiling {
// ---------------------------------------------------------------------------
// MoeTutelCombineXTilingData — tiling data serialized by the host tiling and
// read back by the device kernel. Layout mirrors the kernel-side
// MoetutelcombinexTilingData struct (op_kernel/arch35/moe_tutel_combine_x_tiling_struct.h).
// ---------------------------------------------------------------------------
BEGIN_TILING_DATA_DEF(MoeTutelCombineXTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize);    // B (no-batch = 1)
TILING_DATA_FIELD_DEF(int64_t, samples);      // S tokens per batch
TILING_DATA_FIELD_DEF(int64_t, hidden);       // H hidden dim
TILING_DATA_FIELD_DEF(int64_t, capacity);     // C = eC / E
TILING_DATA_FIELD_DEF(int64_t, eC);           // E*C rows of y_grad
TILING_DATA_FIELD_DEF(int64_t, tokenTotal);   // B*S
TILING_DATA_FIELD_DEF(int64_t, coreTokenBase); // tokens per base core
TILING_DATA_FIELD_DEF(int32_t, bigCoreCnt);   // cores taking one extra token
TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);  // active cores (0 on empty)
TILING_DATA_FIELD_DEF(int64_t, hChunk);       // UB chunk over H (elements)
TILING_DATA_FIELD_DEF(int64_t, hChunkCnt);    // ceil(H / hChunk)
TILING_DATA_FIELD_DEF(int64_t, hChunkTail);   // tail chunk size
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTutelCombineX, MoeTutelCombineXTilingData)

// Compile-time platform info carried from the parse phase to the tiling phase.
struct MoeTutelCombineXCompileInfo {
    uint64_t ubSize = 0;
    uint64_t numBlocks = 0;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_H
