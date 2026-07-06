/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_v2.cpp
 * \brief Kernel entry point for DenseLightningIndexerSoftmaxLseV2 (empty implementation)
 */

#include "kernel_operator.h"
#include "../op_host/dense_lightning_indexer_softmax_lse_v2_tiling.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void dense_lightning_indexer_softmax_lse_v2(
    GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weight,
    GM_ADDR cu_seq_lens_q, GM_ADDR cu_seq_lens_k,
    GM_ADDR seq_used_q, GM_ADDR seq_used_k,
    GM_ADDR cmp_residual_k, GM_ADDR metadata,
    GM_ADDR softmax_lse,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::DenseLISoftmaxLseV2TilingData);
}
