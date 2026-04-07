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
 * \file moe_distribute_dispatch_v2_torch_validate.cpp
 * \brief
 */

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>
#include <string>
#include <cstring>
#include <cstdint>

#include "moe_distribute_dispatch_v2_torch_validate.h"

namespace ascend_ops {
namespace MoeDistributeDispatchV2 {

constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE = 768;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
constexpr int64_t K_MAX = 16;
constexpr int64_t FULLMESH_K_MAX = 12;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t FULLMESH_BS_UPPER_BOUND = 256;
constexpr int64_t CEIL_ALIGN32 = 8;

enum class QuantModeA5 : uint32_t {
    NON_QUANT = 0,
    STATIC_QUANT = 1,
    PERTOKEN_DYNAMIC_QUANT = 2,
    PERGROUP_QUANT = 3,
    MX_QUANT = 4
};

static bool CheckQuantModeAndScales(const bool isScales, const uint32_t quantMode, const at::ScalarType xType)
{
    if (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) {
        if (!isScales) {
            return false;
        }
    }
    if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        if (isScales) {
            return false;
        }
    }
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        if (!isScales && (xType == at::kFloat8_e4m3fn || xType == at::kFloat8_e5m2)) {
            return false;
        }
        if (isScales && (xType == at::kBFloat16 || xType == at::kHalf)) {
            return false;
        }
    }
    return true;
}

static bool CheckXDataType(const at::ScalarType xType, const uint32_t quantMode)
{
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        return (xType == at::kBFloat16 || xType == at::kHalf);
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        return (xType == at::kBFloat16 || xType == at::kHalf);
    } else {
        return (xType == at::kBFloat16 || xType == at::kHalf);
    }
}

static bool CheckScalesDataType(const at::ScalarType scalesType, const at::ScalarType xType)
{
    // if (xType == at::kFloat8_e4m3fn || xType == at::kFloat8_e5m2) {
    //     return (scalesType == at::kFloat || scalesType == at::kFloat8_e8m0);
    // }
    return (scalesType == at::kFloat);
}

static bool ValidateInputParams(const MoeDistributeDispatchV2ValidateParams& params)
{
    int64_t maxEpWorldSize = MAX_EP_WORLD_SIZE;
    int64_t maxMoeExpertNum = MOE_EXPERT_MAX_NUM;
    int64_t maxH = H_MAX;
    int64_t maxBs = BS_UPPER_BOUND;
    
    if (params.comm_alg == "fullmesh_v2") {
        maxBs = FULLMESH_BS_UPPER_BOUND;
    }
    
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
    
    if (params.quant_mode < static_cast<int64_t>(QuantModeA5::NON_QUANT) || 
        params.quant_mode > static_cast<int64_t>(QuantModeA5::MX_QUANT)) {
        return false;
    }
    
    if (params.expert_token_nums_type != 0 && params.expert_token_nums_type != 1) {
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

static bool ValidateTensorShapes(const at::Tensor& x, const at::Tensor& expert_ids,
                                  const c10::optional<at::Tensor>& scales,
                                  const c10::optional<at::Tensor>& x_active_mask,
                                  const c10::optional<at::Tensor>& expert_scales,
                                  const c10::optional<at::Tensor>& elastic_info,
                                  const c10::optional<at::Tensor>& performance_info,
                                  const MoeDistributeDispatchV2ValidateParams& params)
{
    if (x.dim() != 2) {
        return false;
    }
    
    if (expert_ids.dim() != 2) {
        return false;
    }
    
    int64_t bs = x.size(0);
    int64_t h = x.size(1);
    int64_t k = expert_ids.size(1);
    
    int64_t maxH = H_MAX;
    int64_t maxBs = BS_UPPER_BOUND;
    
    if (params.comm_alg == "fullmesh_v2") {
        maxBs = FULLMESH_BS_UPPER_BOUND;
    }
    
    if (h < H_MIN || h > maxH) {
        return false;
    }
    
    if (bs <= 0 || bs > maxBs) {
        return false;
    }
    
    int64_t maxK = (params.comm_alg == "fullmesh_v2") ? FULLMESH_K_MAX : K_MAX;
    if (k <= 0 || k > maxK) {
        return false;
    }
    
    int64_t zeroComputeExpertNum = params.zero_expert_num + params.copy_expert_num + params.const_expert_num;
    if (k > params.moe_expert_num + zeroComputeExpertNum) {
        return false;
    }
    
    bool isScales = scales.has_value();
    if (isScales) {
        const auto& scales_tensor = scales.value();
        if (params.quant_mode == static_cast<int64_t>(QuantModeA5::STATIC_QUANT)) {
            if (scales_tensor.dim() != 1 && scales_tensor.dim() != 2) {
                return false;
            }
        } else {
            if (scales_tensor.dim() != 2) {
                return false;
            }
        }
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
    
    if (elastic_info.has_value()) {
        const auto& elastic_info_tensor = elastic_info.value();
        if (elastic_info_tensor.dim() != 1) {
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

static bool ValidateTensorDataTypes(const at::Tensor& x, const at::Tensor& expert_ids,
                                     const c10::optional<at::Tensor>& scales,
                                     const c10::optional<at::Tensor>& x_active_mask,
                                     const c10::optional<at::Tensor>& expert_scales,
                                     const c10::optional<at::Tensor>& elastic_info,
                                     const c10::optional<at::Tensor>& performance_info,
                                     const MoeDistributeDispatchV2ValidateParams& params)
{
    if (!CheckXDataType(x.scalar_type(), params.quant_mode)) {
        return false;
    }
    
    if (expert_ids.scalar_type() != at::kInt) {
        return false;
    }
    
    bool isScales = scales.has_value();
    if (!CheckQuantModeAndScales(isScales, params.quant_mode, x.scalar_type())) {
        return false;
    }
    
    if (isScales) {
        const auto& scales_tensor = scales.value();
        if (!CheckScalesDataType(scales_tensor.scalar_type(), x.scalar_type())) {
            return false;
        }
    }
    
    if (x_active_mask.has_value()) {
        const auto& x_active_mask_tensor = x_active_mask.value();
        if (x_active_mask_tensor.scalar_type() != at::kBool) {
            return false;
        }
    }
    
    if (expert_scales.has_value()) {
        const auto& expert_scales_tensor = expert_scales.value();
        if (expert_scales_tensor.scalar_type() != at::kFloat) {
            return false;
        }
    }
    
    if (elastic_info.has_value()) {
        const auto& elastic_info_tensor = elastic_info.value();
        if (elastic_info_tensor.scalar_type() != at::kInt) {
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

bool ValidateMoeDistributeDispatchV2Input(const at::Tensor& x, const at::Tensor& expert_ids,
                                           const c10::optional<at::Tensor>& scales,
                                           const c10::optional<at::Tensor>& x_active_mask,
                                           const c10::optional<at::Tensor>& expert_scales,
                                           const c10::optional<at::Tensor>& elastic_info,
                                           const c10::optional<at::Tensor>& performance_info,
                                           const MoeDistributeDispatchV2ValidateParams& params)
{
    if (!ValidateInputParams(params)) {
        return false;
    }
    
    if (!ValidateTensorShapes(x, expert_ids, scales, x_active_mask, expert_scales, 
                               elastic_info, performance_info, params)) {
        return false;
    }
    
    if (!ValidateTensorDataTypes(x, expert_ids, scales, x_active_mask, expert_scales,
                                  elastic_info, performance_info, params)) {
        return false;
    }
    
    return true;
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops