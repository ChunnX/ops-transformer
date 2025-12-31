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
 * \file allto_all_all_gather_batch_mat_mul.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "allto_all_all_gather_batch_mat_mul.h"
#include "allto_all_all_gather_batch_mat_mul_shard_h.h"
#include "allto_all_all_gather_batch_mat_mul_tiling_key.h"

using namespace AscendC;

#ifdef __CCE_KT_TEST__
namespace{
    REGISTER_TILING_DEFAULT(AlltoAllAllGatherBatchMatMulTilingData);
}
#endif

#if ORIG_DTYPE_X == DT_FLOAT16
using DT_X = half;
#else
using DT_X = bfloat16_t;
#endif

#if ORIG_DTYPE_X == DT_FLOAT16
using DT_BIAS = half;
#else
using DT_BIAS = float;
#endif

#define AlltoAllAllGatherBatchMatMul_IMPL_CLASS(...)                                     \
    do {                                                                                 \
        TPipe pipe;                                                                      \                                                         
        AlltoAllAllGatherBatchMatMul<DT_X, DT_BIAS, __VA_ARGS__> op;                     \
        op.Init(xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspaceGM, &pipe, &tilingData);  \
        op.Process();                                                                    \
    } while (0)

#define AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(...)                                     \
    do {                                                                                 \
        TPipe pipe;                                                                      \                                                         
        AlltoAllAllGatherBatchMatMulShardH<DT_X, DT_BIAS, __VA_ARGS__> op;                     \
        op.Init(xGM, weightGM, biasGM, y1GM, y2GM, y3GM, workspaceGM, &pipe, &tilingData);  \
        op.Process();                                                                    \
    } while (0)

template<int XShard, bool WeightTransPose, bool IsBias, bool Y2Need, bool Y3Need>
__global__ __aicore__ void allto_all_all_gather_batch_mat_mul(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM,
                                                                         GM_ADDR y2GM, GM_ADDR y3GM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tilingGM);

    if constexpr (XShard == 1){
        AlltoAllAllGatherBatchMatMul_IMPL_CLASS(XShard, WeightTransPose, IsBias, Y2Need, Y3Need);
        return;
    }
    else if constexpr (XShard == 0) {
        AlltoAllAllGatherBatchMatMul_SHARD_H_IMPL_CLASS(XShard, WeightTransPose, IsBias, Y2Need, Y3Need);
        return;
    }
}