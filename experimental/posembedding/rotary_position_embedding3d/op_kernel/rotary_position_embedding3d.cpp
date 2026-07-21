/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "rotary_position_embedding3d.h"

template <uint32_t schMode>
__global__ __aicore__ void rotary_position_embedding3d(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(RotaryPositionEmbedding3dTilingData);
    GET_TILING_DATA_WITH_STRUCT(RotaryPositionEmbedding3dTilingData, td, tiling);
    if constexpr (schMode == 0) {
        NsRotaryPositionEmbedding3d::RotaryPositionEmbedding3d<float> op;
        op.Init(x, y, z, workspace, &td); op.Process();
    } else if constexpr (schMode == 1) {
        NsRotaryPositionEmbedding3d::RotaryPositionEmbedding3d<half> op;
        op.Init(x, y, z, workspace, &td); op.Process();
    } else if constexpr (schMode == 2) {
        NsRotaryPositionEmbedding3d::RotaryPositionEmbedding3d<bfloat16_t> op;
        op.Init(x, y, z, workspace, &td); op.Process();
    }
}
