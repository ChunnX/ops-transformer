/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_torch_validate.cpp
 * \brief
 */

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>
#include <string>
#include <cstring>
#include <cstdint>

#include "moe_distribute_combine_v2_torch_validate.h"

namespace ascend_ops {
namespace MoeDistributeCombineV2 {

constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 768;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
constexpr int64_t K_MAX = 16;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t CEIL_ALIGN32 = 8;
constexpr int64_t INT8_COMM_QUANT = 2;

enum class CommQuantMode : int32_t {
    NON_QUANT = 0,
    INT12_QUANT = 1,
    INT8_QUANT = 2,
    MXFP8_E5M2_QUANT = 3,
    MXFP8_E4M3_QUANT = 4
};

static bool ValidateInputParams(const MoeDistributeCombineV2ValidateParams& params)
{
    int64_t maxEpWorldSize = MAX_EP_WORLD_SIZE;
    int64_t maxMoeExpertNum = MOE_EXPERT_MAX_NUM;
    int64_t maxH = H_MAX;
    int64_t maxBs = BS_UPPER_BOUND;
    
    if (params.ep_world_size < MIN_EP_WORLD_SIZE || params.ep_world_size > maxEpWorldSize) {
        return false;
    }
    
    if (params.ep_rank_id < 0 || params.ep_rank_id >= params.ep_world_size) {
        return false;
    }
    
    if (params.expert_shard_type != 0) {
        return false;
    }
    
    bool isSharedDefault = (params.shared_expert_num == 1) && (params.shared_expert_rank_num == 0);
    bool isNoShared = (params.shared_expert_num == 0) && (params.shared_expert_rank_num == 0);
    bool isValidShared = (params.shared_expert_num > 0) &&
                        ((params.shared_expert_rank_num / params.shared_expert_num) > 0) &&
                        ((params.shared_expert_rank_num % params.shared_expert_num) == 0);
    
    if (!isSharedDefault && !isNoShared && !isValidShared) {
        return false;
    }
    
    if (params.shared_expert_rank_num < 0 || params.shared_expert_rank_num >= params.ep_world_size) {
        return false;
    }
    
    if (params.moe_expert_num <= 0 || params.moe_expert_num > maxMoeExpertNum) {
        return false;
    }
    
    if (params.comm_quant_mode < static_cast<int64_t>(CommQuantMode::NON_QUANT) || 
        params.comm_quant_mode > static_cast<int64_t>(CommQuantMode::MXFP8_E4M3_QUANT)) {
        return false;
    }
    
    if (params.global_bs < 0) {
        return false;
    }
    
    if (params.zero_expert_num < 0) {
        return false;
    }
    
    if (params.copy_expert_num < 0) {
        return false;
    }
    
    if (params.const_expert_num < 0) {
        return false;
    }
    
    int64_t zeroComputeExpertNum = params.zero_expert_num + params.copy_expert_num + params.const_expert_num;
    if (zeroComputeExpertNum + params.moe_expert_num > INT32_MAX) {
        return false;
    }
    
    return true;
}

static bool ValidateTensorShapes(const at::Tensor& expand_x, const at::Tensor& expert_ids,
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
                               const MoeDistributeCombineV2ValidateParams& params)
{
    if (expand_x.dim() != 2) {
        return false;
    }
    
    if (expert_ids.dim() != 2) {
        return false;
    }
    
    if (assist_info_for_combine.dim() != 1) {
        return false;
    }
    
    if (ep_send_counts.dim() != 1) {
        return false;
    }
    
    if (expert_scales.dim() != 2) {
        return false;
    }
    
    int64_t bs = expert_ids.size(0);
    int64_t h = expand_x.size(1);
    int64_t k = expert_ids.size(1);
    
    if (h < H_MIN || h > H_MAX) {
        return false;
    }
    
    if (bs <= 0 || bs > BS_UPPER_BOUND) {
        return false;
    }
    
    if (k <= 0 || k > K_MAX) {
        return false;
    }
    
    int64_t zeroComputeExpertNum = params.zero_expert_num + params.copy_expert_num + params.const_expert_num;
    if (k > params.moe_expert_num + zeroComputeExpertNum) {
        return false;
    }
    
    if (x_active_mask.has_value()) {
        const auto& x_active_mask_tensor = x_active_mask.value();
        if (x_active_mask_tensor.dim() != 1 && x_active_mask_tensor.dim() != 2) {
            return false;
        }
        if (x_active_mask_tensor.size(0) != bs) {
            return false;
        }
        if (x_active_mask_tensor.dim() == 2 && x_active_mask_tensor.size(1) != k) {
            return false;
        }
    }
    
    if (shared_expert_x.has_value()) {
        const auto& shared_expert_x_tensor = shared_expert_x.value();
        if (shared_expert_x_tensor.dim() != 2 && shared_expert_x_tensor.dim() != 3) {
            return false;
        }
    }
    
    if (ori_x.has_value()) {
        const auto& ori_x_tensor = ori_x.value();
        if (ori_x_tensor.dim() != 2) {
            return false;
        }
        if (ori_x_tensor.size(0) != bs) {
            return false;
        }
        if (ori_x_tensor.size(1) != h) {
            return false;
        }
    }
    
    if (const_expert_alpha_1.has_value()) {
        const auto& const_expert_alpha_1_tensor = const_expert_alpha_1.value();
        if (const_expert_alpha_1_tensor.dim() != 2) {
            return false;
        }
    }
    
    if (const_expert_alpha_2.has_value()) {
        const auto& const_expert_alpha_2_tensor = const_expert_alpha_2.value();
        if (const_expert_alpha_2_tensor.dim() != 2) {
            return false;
        }
    }
    
    if (const_expert_v.has_value()) {
        const auto& const_expert_v_tensor = const_expert_v.value();
        if (const_expert_v_tensor.dim() != 2) {
            return false;
        }
    }
    
    if (performance_info.has_value()) {
        const auto& performance_info_tensor = performance_info.value();
        if (performance_info_tensor.dim() != 1) {
            return false;
        }
        if (performance_info_tensor.size(0) != params.ep_world_size) {
            return false;
        }
    }
    
    return true;
}

static bool ValidateTensorDataTypes(const at::Tensor& expand_x, const at::Tensor& expert_ids,
                                    const at::Tensor& assist_info_for_combine,
                                    const at::Tensor& ep_send_counts,
                                    const at::Tensor& expert_scales,
                                    const c10::optional<at::Tensor>& x_active_mask,
                                    const c10::optional<at::Tensor>& shared_expert_x,
                                    const c10::optional<at::Tensor>& ori_x,
                                    const c10::optional<at::Tensor>& const_expert_alpha_1,
                                    const c10::optional<at::Tensor>& const_expert_alpha_2,
                                    const c10::optional<at::Tensor>& const_expert_v,
                                    const c10::optional<at::Tensor>& performance_info)
{
    if (expand_x.scalar_type() != at::kBFloat16 && expand_x.scalar_type() != at::kHalf) {
        return false;
    }
    
    if (expert_ids.scalar_type() != at::kInt) {
        return false;
    }
    
    if (assist_info_for_combine.scalar_type() != at::kInt) {
        return false;
    }
    
    if (ep_send_counts.scalar_type() != at::kInt) {
        return false;
    }
    
    if (expert_scales.scalar_type() != at::kFloat) {
        return false;
    }
    
    if (x_active_mask.has_value()) {
        const auto& x_active_mask_tensor = x_active_mask.value();
        if (x_active_mask_tensor.scalar_type() != at::kBool) {
            return false;
        }
    }
    
    if (shared_expert_x.has_value()) {
        const auto& shared_expert_x_tensor = shared_expert_x.value();
        if (shared_expert_x_tensor.scalar_type() != expand_x.scalar_type()) {
            return false;
        }
    }
    
    if (ori_x.has_value()) {
        const auto& ori_x_tensor = ori_x.value();
        if (ori_x_tensor.scalar_type() != expand_x.scalar_type()) {
            return false;
        }
    }
    
    if (const_expert_alpha_1.has_value()) {
        const auto& const_expert_alpha_1_tensor = const_expert_alpha_1.value();
        if (const_expert_alpha_1_tensor.scalar_type() != expand_x.scalar_type()) {
            return false;
        }
    }
    
    if (const_expert_alpha_2.has_value()) {
        const auto& const_expert_alpha_2_tensor = const_expert_alpha_2.value();
        if (const_expert_alpha_2_tensor.scalar_type() != expand_x.scalar_type()) {
            return false;
        }
    }
    
    if (const_expert_v.has_value()) {
        const auto& const_expert_v_tensor = const_expert_v.value();
        if (const_expert_v_tensor.scalar_type() != expand_x.scalar_type()) {
            return false;
        }
    }
    
    if (performance_info.has_value()) {
        const auto& performance_info_tensor = performance_info.value();
        if (performance_info_tensor.scalar_type() != at::kLong) {
            return false;
        }
    }
    
    return true;
}

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
                                         const MoeDistributeCombineV2ValidateParams& params)
{
    if (!ValidateInputParams(params)) {
        return false;
    }
    
    if (!ValidateTensorShapes(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales,
                              x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1,
                              const_expert_alpha_2, const_expert_v, performance_info, params)) {
        return false;
    }
    
    if (!ValidateTensorDataTypes(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales,
                                 x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1,
                                 const_expert_alpha_2, const_expert_v, performance_info)) {
        return false;
    }
    
    return true;
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops