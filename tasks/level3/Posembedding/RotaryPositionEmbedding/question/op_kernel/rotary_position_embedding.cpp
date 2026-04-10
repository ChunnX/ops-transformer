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
 * \file rotary_position_embedding.cpp
 * \brief
 */
#include "rotate_half.h"
#include "rotate_half_bf16.h"
#include "rotate_interleaved_split_s.h"
#include "rotate_interleaved_split_bs.h"
#include "rotate_interleaved_split_bsn.h"
#include "rotate_interleaved_split_s_pad.h"
#include "rotate_interleaved_split_bs_pad.h"
#include "rotate_interleaved_split_bsn_pad.h"
using namespace AscendC;
using namespace RotateHalfN;
using namespace RotateInterleavedN;

extern "C" __global__ __aicore__ void rotary_position_embedding(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
}
