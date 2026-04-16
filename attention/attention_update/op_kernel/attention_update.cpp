/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
 /*!
 * \file attention_update.cpp
 * \brief
 */
#include "decode_update.h"

extern "C" __global__ __aicore__ void attention_update(GM_ADDR les, GM_ADDR in, GM_ADDR out, GM_ADDR lseOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    AttentionUpdate::DecodeUpdate op;
    op.Init(les, in, out, lseOut, &tilingData);
    op.Process();
}
