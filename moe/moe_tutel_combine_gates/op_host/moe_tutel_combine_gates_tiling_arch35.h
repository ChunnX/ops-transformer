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
 * \file moe_tutel_combine_gates_tiling_arch35.h
 * \brief MoeTutelCombineGates arch35 (Ascend 950) tiling compile info.
 */

#ifndef MOE_TUTEL_COMBINE_GATES_TILING_ARCH35_H
#define MOE_TUTEL_COMBINE_GATES_TILING_ARCH35_H

// Include the tiling data struct definition — this is shared between
// host-side tiling and device-side kernel (same struct, different compilation).
#include "../op_kernel/arch35/moe_tutel_combine_gates_tiling_struct.h"
// gert::TilingContext / ge::graphStatus — needed by the tiling callback declarations below.
#include "register/op_impl_registry.h"

namespace optiling {

// ---------------------------------------------------------------------------
// MoeTutelCombineGatesCompileInfo — platform information for tiling compile phase
//
// This struct is populated by TilingPrepareForMoeTutelCombineGates() with platform
// hardware information (number of AIV cores, UB memory size). It is then
// passed to the tiling function so it can make decisions about data
// partitioning and parallelism.
//
// Fields:
//   coreNum — number of available AIV (AI Vector) cores on the NPU
//   ubSize  — size of Unified Buffer (UB) in bytes per core
//             UB is the fast on-chip memory used for kernel computation
// ---------------------------------------------------------------------------
struct MoeTutelCombineGatesCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

// ---------------------------------------------------------------------------
// TilingFuncMoeTutelCombineGates — the runtime tiling callback.
//
// Implemented in moe_tutel_combine_gates_tiling_arch35.cpp and registered with
// the framework through IMPL_OP_OPTILING(MoeTutelCombineGates).
// ---------------------------------------------------------------------------
ge::graphStatus TilingFuncMoeTutelCombineGates(gert::TilingContext *context);

} // namespace optiling

#endif
