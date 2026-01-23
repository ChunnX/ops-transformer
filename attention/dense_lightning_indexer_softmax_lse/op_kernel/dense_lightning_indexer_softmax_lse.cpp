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
 * \file dense_lightning_indexer_softmax_lse.cpp
 * \brief
 */

#include "dense_lightning_indexer_softmax_lse.h"
#include "dense_lightning_indexer_softmax_lse_common.h"

using namespace DenseLISoftmaxLseKernel;
using namespace DenseLISoftmaxLseCommon;
using namespace AscendC;

#define DENSE_LI_SOFTMAX_OP_IMPL(templateClass, ...)                                          \
    do {                                                                                      \
        templateClass<DenseLISoftmaxLseCommon::DenseLISoftmaxLseType<__VA_ARGS__>> op;              \
        GET_TILING_DATA_WITH_STRUCT(DenseLISoftmaxLseTilingData, tiling_data_in, tiling);        \
        const DenseLISoftmaxLseTilingData *__restrict tiling_data = &tiling_data_in;             \
        op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, softmaxMax,         \
                softmaxSum, user, tiling_data, &tPipe);                                       \
        op.Process();                                                                         \
    } while (0)

extern "C" __global__ __aicore__ void dense_lightning_indexer_softmax_lse(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                           __gm__ uint8_t *weights, __gm__ uint8_t *actualSeqLengthsQ,
                                                           __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *softmaxMax,
                                                           __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
                                                           __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)

#else
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(0)) {
        DENSE_LI_SOFTMAX_OP_IMPL(DenseLISoftmaxLse, half, half, float, LAYOUT::BSND);
    } else if (TILING_KEY_IS(1)) {
        DENSE_LI_SOFTMAX_OP_IMPL(DenseLISoftmaxLse, bfloat16_t, bfloat16_t, float, LAYOUT::BSND);
    } else  if (TILING_KEY_IS(10)) {
        DENSE_LI_SOFTMAX_OP_IMPL(DenseLISoftmaxLse, half, half, float, LAYOUT::TND);
    } else if (TILING_KEY_IS(11)) {
        DENSE_LI_SOFTMAX_OP_IMPL(DenseLISoftmaxLse, bfloat16_t, bfloat16_t, float, LAYOUT::TND);
    }
#endif
}