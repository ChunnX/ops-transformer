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
 * \file rotary_position_embedding_grad.cpp
 * \brief
 */
#include "rotate_half_grad.h"
#include "rope_interleaved_grad_splits.h"

extern "C" __global__ __aicore__ void rotary_position_embedding_grad(
    GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x, GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,
    GM_ADDR workspace, GM_ADDR tiling)
{
}
