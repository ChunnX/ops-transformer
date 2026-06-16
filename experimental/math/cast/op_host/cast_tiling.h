/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_tiling.h
 * \brief Cast tiling for ascend310p, elementwise split (no reg fields, no MicroAPI)
 */
#ifndef EXPERIMENTAL_MATH_CAST_OP_HOST_CAST_TILING_H
#define EXPERIMENTAL_MATH_CAST_OP_HOST_CAST_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

struct CastCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(CastTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockNum);    // used core num
TILING_DATA_FIELD_DEF(int64_t, ubFormer);    // elements per UB loop (half of UB with double buffer)
TILING_DATA_FIELD_DEF(int64_t, blockFormer); // elements of former block
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cast, CastTilingData);

}  // namespace optiling
#endif  // EXPERIMENTAL_MATH_CAST_OP_HOST_CAST_TILING_H
