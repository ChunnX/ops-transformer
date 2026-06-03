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
 * \file attention_update.cpp
 * \brief
 */
#include "decode_update.h"
#include "decode_update_large_sp.h"

#define TILING_KEY_GO_FP32 10010
#define TILING_KEY_GO_FP16 10020
#define TILING_KEY_GO_BF16 10030

#define TILING_KEY_GO_FP32_LARGE_SP 10110
#define TILING_KEY_GO_FP16_LARGE_SP 10120
#define TILING_KEY_GO_BF16_LARGE_SP 10130

extern "C" __global__ __aicore__ void attention_update(GM_ADDR les, GM_ADDR in, GM_ADDR out, GM_ADDR lseOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DecodeUpdateTilingData, tilingDataIn, tiling);
    const DecodeUpdateTilingData* __restrict__ tilingData = &tilingDataIn;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILING_KEY_GO_FP32)) {
        AttentionUpdate::DecodeUpdate<float, float> op;
        op.Init(les, in, out, lseOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GO_FP16)) {
        AttentionUpdate::DecodeUpdate<float, half> op;
        op.Init(les, in, out, lseOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GO_BF16)) {
        AttentionUpdate::DecodeUpdate<float, bfloat16_t> op;
        op.Init(les, in, out, lseOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GO_FP32_LARGE_SP)) {
        AttentionUpdate::DecodeUpdateLargeSp<float, float> op;
        op.Init(les, in, out, lseOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GO_FP16_LARGE_SP)) {
        AttentionUpdate::DecodeUpdateLargeSp<float, half> op;
        op.Init(les, in, out, lseOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GO_BF16_LARGE_SP)) {
        AttentionUpdate::DecodeUpdateLargeSp<float, bfloat16_t> op;
        op.Init(les, in, out, lseOut, tilingData);
        op.Process();
    }
}
