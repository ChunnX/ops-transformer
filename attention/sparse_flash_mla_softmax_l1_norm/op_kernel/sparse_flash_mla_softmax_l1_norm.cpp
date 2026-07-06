/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_mla_softmax_l1_norm.cpp
 * \brief kernel 入口（骨架阶段空实现）
 */
#include "kernel_operator.h"
#include "sparse_flash_mla_softmax_l1_norm_tiling_data.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sparse_flash_mla_softmax_l1_norm(
    GM_ADDR query, GM_ADDR key, GM_ADDR lse,
    GM_ADDR sparse_indices_optional, GM_ADDR cu_seqlens_q_optional, GM_ADDR cu_seqlens_k_optional,
    GM_ADDR seqused_q_optional, GM_ADDR seqused_k_optional, GM_ADDR cmp_residual_k_optional,
    GM_ADDR topk_length_optional, GM_ADDR metadata_optional,
    GM_ADDR softmax_l1_norm_optional, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::SmlaSoftmaxL1NormTilingData);
}
