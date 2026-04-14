/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file three_interpolate_backward_tiling.h
 * \brief
 */
#ifndef THREE_INTERPOLATE_BACKWARD_TILING_H_
#define THREE_INTERPOLATE_BACKWARD_TILING_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"

namespace optiling {

enum class ThreeInterpolateBackwardTilingType
{
    TILING_MODE_FP32_INT32,
    TILING_MODE_FP32_INT64,
    TILING_MODE_FP16_INT32,
    TILING_MODE_FP16_INT64
};

struct ThreeInterpolateShapeInfo {
    uint32_t bs;
    uint32_t c1;
    uint32_t ms;
    uint32_t ns;
};

BEGIN_TILING_DATA_DEF(ThreeInterpolateBackwardTilingData)
TILING_DATA_FIELD_DEF(uint32_t, used_core_num);
TILING_DATA_FIELD_DEF(uint32_t, bs);
TILING_DATA_FIELD_DEF(uint32_t, c1);
TILING_DATA_FIELD_DEF(uint32_t, ms);
TILING_DATA_FIELD_DEF(uint32_t, ns);
TILING_DATA_FIELD_DEF(uint32_t, each_core_proc_num);
TILING_DATA_FIELD_DEF(uint32_t, each_core_loop_times);
TILING_DATA_FIELD_DEF(uint32_t, each_core_each_loop_n_cnt);
TILING_DATA_FIELD_DEF(uint32_t, each_core_last_loop_n_cnt);
TILING_DATA_FIELD_DEF(uint32_t, last_core_proc_num);
TILING_DATA_FIELD_DEF(uint32_t, last_core_loop_times);
TILING_DATA_FIELD_DEF(uint32_t, last_core_each_loop_n_cnt);
TILING_DATA_FIELD_DEF(uint32_t, last_core_last_loop_n_cnt);
TILING_DATA_FIELD_DEF(uint32_t, weight_move_block_size);
TILING_DATA_FIELD_DEF(uint32_t, idx_move_block_size);
TILING_DATA_FIELD_DEF(uint32_t, grad_x_move_block_size);
TILING_DATA_FIELD_DEF(uint32_t, grad_y_move_block_size);
TILING_DATA_FIELD_DEF(uint32_t, c_move_num);
TILING_DATA_FIELD_DEF(uint32_t, c_last_loop_move_num);
TILING_DATA_FIELD_DEF(uint32_t, c_move_loop_times);
TILING_DATA_FIELD_DEF(uint32_t, mulit_core_mode);
TILING_DATA_FIELD_DEF(uint32_t, each_core_proc_batch_num);
TILING_DATA_FIELD_DEF(uint32_t, core_proc_batch_padding_idx);
TILING_DATA_FIELD_DEF(uint32_t, rsv);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ThreeInterpolateBackward, ThreeInterpolateBackwardTilingData)

} // namespace optiling

#endif
