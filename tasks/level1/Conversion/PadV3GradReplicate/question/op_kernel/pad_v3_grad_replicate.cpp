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
 * \file pad_v3_grad_replicate.cpp
 * \brief
 */

#include "pad_v3_grad_replicate_h.h"
#include "pad_v3_grad_replicate_w.h"
#include "pad_v3_grad_replicate_h_w.h"
#include "pad_v3_grad_replicate_h_w_mini.h"
#include "pad_v3_grad_replicate_small_h_large_w.h"
#include "pad_v3_grad_replicate_small_h_large_w_bf16.h"
#include "pad_v3_grad_replicate_large_h_small_w.h"
#include "pad_v3_grad_replicate_large_h_small_w_bf16.h"
#include "pad_v3_grad_replicate_h_w_large.h"

extern "C" __global__ __aicore__ void pad_v3_grad_replicate(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}