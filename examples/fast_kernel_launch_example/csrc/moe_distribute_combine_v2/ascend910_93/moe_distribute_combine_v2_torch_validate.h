/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_torch_validate.h
 * \brief
 */

#pragma once

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>

namespace ascend_ops {
namespace MoeDistributeCombineV2 {

struct MoeDistributeCombineV2ValidateParams {
    int64_t ep_world_size;
    int64_t ep_rank_id;
    int64_t moe_expert_num;
    int64_t expert_shard_type;
    int64_t shared_expert_num;
    int64_t shared_expert_rank_num;
    int64_t global_bs;
    int64_t comm_quant_mode;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;
};

bool ValidateMoeDistributeCombineV2Input(const at::Tensor& expand_x, const at::Tensor& expert_ids,
                                         const at::Tensor& assist_info_for_combine,
                                         const at::Tensor& ep_send_counts,
                                         const at::Tensor& expert_scales,
                                         const c10::optional<at::Tensor>& x_active_mask,
                                         const c10::optional<at::Tensor>& shared_expert_x,
                                         const c10::optional<at::Tensor>& ori_x,
                                         const c10::optional<at::Tensor>& const_expert_alpha_1,
                                         const c10::optional<at::Tensor>& const_expert_alpha_2,
                                         const c10::optional<at::Tensor>& const_expert_v,
                                         const c10::optional<at::Tensor>& performance_info,
                                         const MoeDistributeCombineV2ValidateParams& params);

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops