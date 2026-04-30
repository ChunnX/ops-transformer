/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
using namespace at_npu::native;

// 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_recurrence(
    at::Tensor &initial_state, const at::Tensor &kgexp, const at::Tensor &value, const at::Tensor &k_cumdecay,
    const at::Tensor &qgexp, const at::Tensor &gexp, const at::Tensor &actual_seqlens)
{
    at::Tensor attn_inter_out = at::empty_like(value);
    at::Tensor v_new_out = at::empty_like(value);

    EXEC_NPU_CMD_V1(aclnnAiInfraChunkGatedDeltaRuleRecurrence, initial_state, kgexp, value, k_cumdecay, qgexp, gexp,
        actual_seqlens, attn_inter_out, v_new_out);

    return std::tuple<at::Tensor, at::Tensor>(attn_inter_out, v_new_out);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_recurrence_meta(
    at::Tensor &initial_state, const at::Tensor &kgexp, const at::Tensor &value, const at::Tensor &k_cumdecay,
    const at::Tensor &qgexp, const at::Tensor &gexp, const at::Tensor &actual_seqlens)
{
    at::Tensor attn_inter_out = at::empty_like(value);
    at::Tensor v_new_out = at::empty_like(value);

    return std::tuple<at::Tensor, at::Tensor>(attn_inter_out, v_new_out);
}
}

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_chunk_gated_delta_rule_recurrence",
        &custom::npu_chunk_gated_delta_rule_recurrence);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_chunk_gated_delta_rule_recurrence",
        &custom::npu_chunk_gated_delta_rule_recurrence_meta);
}
