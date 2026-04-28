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
 * \brief Kernel entry with architecture branching
 */

#include "kernel_operator.h"
using namespace AscendC;

#if __CCE_AICORE__ != 310
#include "arch32/dense_li_softmax_lse.h"
#include "arch32/dense_lightning_indexer_softmax_lse_tiling_key.h"
#include "arch32/dense_lightning_indexer_softmax_lse_common.h"

using namespace DenseLISoftmaxLseKernel;
using namespace DenseLISoftmaxLseCommon;

#define DENSE_LI_SOFTMAX_LSE_OP_IMPL(templateClass, ...)                                         \
    do {                                                                                         \
        templateClass<DenseLISoftmaxLseType<__VA_ARGS__>> op;                                    \
        GET_TILING_DATA_WITH_STRUCT(DenseLISoftmaxLseTilingData, tiling_data_in, tiling);        \
        const DenseLISoftmaxLseTilingData *__restrict tiling_data = &tiling_data_in;             \
        op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, softmaxMax,            \
                softmaxSum, user, tiling_data, &tPipe);                                          \
        op.Process();                                                                            \
    } while (0)

#else
#include "arch35/dense_lightning_indexer_softmax_lse_regbase_common.h"
#include "arch35/dense_lightning_indexer_softmax_lse_tiling_regbase.h"
#include "arch35/dense_lightning_indexer_softmax_lse_template_tiling_key_regbase.h"
#include "arch35/dense_lightning_indexer_softmax_lse_vector_block.h"
#include "arch35/dense_lightning_indexer_softmax_lse_cube_block.h"
#include "arch35/dense_li_softmax_lse_kernel_base.h"

using namespace DenseLISoftmaxLseRegbase;

#define DENSE_LI_SOFTMAX_LSE_OP_IMPL_REGBASE(templateClass, ...)    \
    do {                                                            \
        using CubeBlockType = typename std::conditional<            \
            g_coreType == AscendC::AIC,                             \
            DenseLISoftmaxLseCubeBlock<__VA_ARGS__>,                \
            DenseLISoftmaxLseCubeBlockDummy<__VA_ARGS__>>::type;    \
        using VecBlockType = typename std::conditional<             \
            g_coreType == AscendC::AIV,                             \
            DenseLISoftmaxLseVecBlock<__VA_ARGS__>,                 \
            DenseLISoftmaxLseVecBlockDummy<__VA_ARGS__>>::type;     \
        templateClass<CubeBlockType, VecBlockType> op;              \
        GET_TILING_DATA_WITH_STRUCT(optiling::DenseLISoftmaxLseRegBaseTilingData, tiling_data_in, tiling); \
        const optiling::DenseLISoftmaxLseRegBaseTilingData *__restrict tiling_data = &tiling_data_in; \
        op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, \
                softmaxMax, softmaxSum, user, tiling_data, &tPipe); \
        op.Process();                                               \
    } while (0)

#endif

template <int LAYOUT_T>
__global__ __aicore__ void dense_lightning_indexer_softmax_lse(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                           __gm__ uint8_t *weights, __gm__ uint8_t *actualSeqLengthsQ,
                                                           __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *softmaxMax,
                                                           __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
                                                           __gm__ uint8_t *tiling)
{
#if __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::DenseLISoftmaxLseRegBaseTilingData);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    DENSE_LI_SOFTMAX_LSE_OP_IMPL_REGBASE(DenseLISoftmaxLseKernelBase,
        DTYPE_QUERY_INDEX, DTYPE_KEY_INDEX);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    DENSE_LI_SOFTMAX_LSE_OP_IMPL(DenseLISoftmaxLse, DTYPE_QUERY_INDEX, DTYPE_KEY_INDEX, DTYPE_WEIGHT,
        DTYPE_SOFTMAX_MAX, LAYOUT(LAYOUT_T));
#endif
}