/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_attn_res_update.h
 * \brief BlockAttnResUpdate arch35 kernel framework.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_H
#define BLOCK_ATTN_RES_UPDATE_H

#include "kernel_operator.h"
#include "block_attn_res_update_tiling_data.h"

namespace BlockAttnResUpdate {
using namespace AscendC;

template <int8_t RESERVED_MODE>
class BlockAttnResUpdateKernel {
public:
    __aicore__ inline BlockAttnResUpdateKernel() = default;

    __aicore__ inline void Init(GM_ADDR partialBlock, GM_ADDR partialDelta, GM_ADDR effectiveQuery, GM_ADDR interMax,
                                GM_ADDR interExpSum, GM_ADDR interNumerator, GM_ADDR h, GM_ADDR partialBlocks,
                                GM_ADDR workspace, const BlockAttnResUpdateTilingData *tilingData, TPipe *pipe)
    {
        (void)partialBlock;
        (void)partialDelta;
        (void)effectiveQuery;
        (void)interMax;
        (void)interExpSum;
        (void)interNumerator;
        (void)h;
        (void)partialBlocks;
        (void)workspace;
        (void)tilingData;
        (void)pipe;
    }

    __aicore__ inline void Process()
    {
        (void)RESERVED_MODE;
    }
};

} // namespace BlockAttnResUpdate

#endif // BLOCK_ATTN_RES_UPDATE_H

