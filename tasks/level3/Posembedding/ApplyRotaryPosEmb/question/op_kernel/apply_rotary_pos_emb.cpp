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
 * \file apply_rotary_pos_emb.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_small.h"
#include "apply_rotary_pos_emb_compute_ab.h"
#include "apply_rotary_pos_emb_compute_ab_cast.h"

using namespace ApplyRotaryPosEmb;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin, GM_ADDR q_out, GM_ADDR k_out, GM_ADDR workspace, GM_ADDR tiling)
{
}
