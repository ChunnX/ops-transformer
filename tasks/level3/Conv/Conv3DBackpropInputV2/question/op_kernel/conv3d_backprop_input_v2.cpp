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
 * \file conv3d_backprop_input_v2.cpp
 * \brief
 */
#include "./arch32/conv3d_backprop_input_v2_tiling_data.h"
#include "./arch32/conv3d_dx_v2_basic_block.h"
#include "./arch32/conv3d_backprop_input_v2_init_output.h"
#include "./arch32/conv3d_backprop_input_v2.h"
#include "conv3d_backprop_input_v2_tiling_key.h"

extern "C" __global__ __aicore__ void conv3d_backprop_input_v2(
    GM_ADDR input_size, GM_ADDR filter, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling)
{
}
