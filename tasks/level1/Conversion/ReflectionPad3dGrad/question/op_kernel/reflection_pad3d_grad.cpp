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
 * \file reflection_pad3d_grad.cpp
 * \brief
 */
#include "reflection_pad3d_grad_mid.h"
#include "reflection_pad3d_grad_small.h"
#include "reflection_pad3d_grad_big.h"
#include "reflection_pad3d_grad_flat.h"
#include "reflection_pad3d_grad_f16.h"

extern "C" __global__ __aicore__ void reflection_pad3d_grad(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}