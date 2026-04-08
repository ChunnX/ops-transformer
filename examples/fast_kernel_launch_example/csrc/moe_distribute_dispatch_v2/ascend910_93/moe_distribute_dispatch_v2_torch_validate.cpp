/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_torch_validate.cpp
 * \brief
 */

#include "moe_distribute_dispatch_v2_torch_validate.h"

namespace ascend_ops {
namespace MoeDistributeDispatchV2 {

static void InitValidateContext(MoeDistributeDispatchV2ValidateParams& params)
{
    params.maxEpWorldSize = MAX_EP_WORLD_SIZE;
    params.maxMoeExpertNum = MOE_EXPERT_MAX_NUM;
    params.maxH = H_MAX;
    params.maxBs = BS_UPPER_BOUND;
    params.isFullmeshV2 = (params.comm_alg == "fullmesh_v2");
    if (params.isFullmeshV2) {
        params.maxBs = FULLMESH_BS_UPPER_BOUND;
    }
    params.zeroComputeExpertNum = params.zero_expert_num + params.copy_expert_num + params.const_expert_num;
    params.moeExpertNumDivisor = params.ep_world_size - params.shared_expert_rank_num;
    params.localMoeExpertNum = params.moe_expert_num / params.moeExpertNumDivisor;
    params.isSharedDefault = (params.shared_expert_num == 1) && (params.shared_expert_rank_num == 0);
    params.isNoShared = (params.shared_expert_num == 0) && (params.shared_expert_rank_num == 0);
    params.isValidShared = (params.shared_expert_num > 0) &&
                        ((params.shared_expert_rank_num / params.shared_expert_num) > 0) &&
                        ((params.shared_expert_rank_num % params.shared_expert_num) == 0);
    params.maxK = params.isFullmeshV2 ? FULLMESH_K_MAX : K_MAX;
    params.isScales = false;
    params.bs = 0;
    params.h = 0;
    params.k = 0;
}

static void ValidateEpAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.ep_world_size >= MIN_EP_WORLD_SIZE) && (params.ep_world_size <= MAX_EP_WORLD_SIZE),
                "ep_world_size is invalid, only support [", MIN_EP_WORLD_SIZE, ", ", MAX_EP_WORLD_SIZE, "], but got ",
                "ep_world_size: ", params.ep_world_size);
    
    TORCH_CHECK((params.ep_rank_id >= 0) && (params.ep_rank_id < params.ep_world_size),
                "ep_rank_id is invalid, only support [0, ep_world_size), but got",
                " ep_world_size: ", params.ep_world_size,
                ", ep_rank_id: ", params.ep_rank_id);
}

static void ValidateSharedExpertAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK(params.shared_expert_num == 1 || params.shared_expert_rank_num == 0,
                "shared_expert_num and shared_expert_rank_num have obvious value situations: ",
                "1. shared_expert_num is 1, shared_expert_rank_num is 0; 2. shared_expert num is 0, ",
                "shared_expert_rank_num is 0; 3. shared_expert_num in (0, shared_expert_rank_num] and ",
                "shared_expert_rank_num % shared_expert_num = 0. but current input value is ",
                " shared_expert_num: ", params.shared_expert_num,
                ", shared_expert_rank_num: ", params.shared_expert_rank_num);
    
    TORCH_CHECK((params.shared_expert_rank_num >= 0) && (params.shared_expert_rank_num < params.ep_world_size),
                "shared_expert_rank_num is invalid, only support [0, ep_world_size), but got",
                " ep_world_size: ", params.ep_world_size,
                ", shared_expert_rank_num: ", params.shared_expert_rank_num);
}

static void ValidateMoeExpertAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.moe_expert_num > 0) && (params.moe_expert_num <= params.maxMoeExpertNum),
                "moe_expert_num is invalid, only support (0, ", params.maxMoeExpertNum, "], but got ",
                params.moe_expert_num);
    
    TORCH_CHECK(params.moeExpertNumDivisor > 0,
                "moeExpertNum divisor (ep_world_size - shared_expert_rank_num) must be greater than 0, ",
                "but ep_world_size is ", params.ep_world_size,
                ", shared_expert_rank_num is ", params.shared_expert_rank_num);
    
    TORCH_CHECK(params.moe_expert_num % params.moeExpertNumDivisor == 0,
                "moe_expert_num should be divisible by (ep_world_size - shared_expert_rank_num), "
                "but moe_expert_num = ", params.moe_expert_num,
                ", ep_world_size = ", params.ep_world_size,
                ", shared_expert_rank_num = ", params.shared_expert_rank_num,
                ", divisor = ", params.moeExpertNumDivisor);
    
    TORCH_CHECK(params.localMoeExpertNum > 0,
                "localMoeExpertNum must be greater than 0, but got ", params.localMoeExpertNum,
                ", moe_expert_num = ", params.moe_expert_num,
                ", ep_world_size = ", params.ep_world_size,
                ", shared_expert_rank_num = ", params.shared_expert_rank_num);
    
    TORCH_CHECK(params.localMoeExpertNum * params.ep_world_size <= LOCAL_EXPERT_MAX_SIZE,
                "localMoeExpertNum * ep_world_size must be less than or equal to ", LOCAL_EXPERT_MAX_SIZE,
                ", but localMoeExpertNum * ep_world_size = ", params.localMoeExpertNum * params.ep_world_size,
                ", localMoeExpertNum = ", params.localMoeExpertNum,
                ", ep_world_size = ", params.ep_world_size);
}

static void ValidateQuantModeAndScales(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.quant_mode == static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
                (params.quant_mode == static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)),
                "quant_mode is invalid, only support 0 (NON_QUANT) or 2 (PERTOKEN_DYNAMIC_QUANT), but got ", params.quant_mode);
}

static void ValidateCommonAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.expert_token_nums_type == 0) || (params.expert_token_nums_type == 1),
                "The expert_token_nums_type should be 0 or 1.");
    
    TORCH_CHECK(params.global_bs >= 0,
                "global_bs is invalid, only support >= 0, but got ", params.global_bs);
    
    TORCH_CHECK(params.zero_expert_num >= 0,
                "zero_expert_num is invalid, only support >= 0, but got ", params.zero_expert_num);
    
    TORCH_CHECK(params.copy_expert_num >= 0,
                "copy_expert_num is invalid, only support >= 0, but got ", params.copy_expert_num);
    
    TORCH_CHECK(params.const_expert_num >= 0,
                "const_expert_num is invalid, only support >= 0, but got ", params.const_expert_num);
    
    TORCH_CHECK((params.zeroComputeExpertNum + params.moe_expert_num) <= INT32_MAX,
                "zero_expert_num + copy_expert_num + const_expert_num + moe_expert_num exceeds INT32_MAX. ",
                "zero_expert_num: ", params.zero_expert_num, ", copy_expert_num: ", params.copy_expert_num,
                ", const_expert_num: ", params.const_expert_num, ", moe_expert_num: ", params.moe_expert_num);
}

static void ValidateInputTensorDim(const at::Tensor& x, const at::Tensor& expert_ids,
                                    MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK(x.dim() == 2,
                "x must be 2-dimension, but got ", x.dim(), " dimension");
    
    TORCH_CHECK(expert_ids.dim() == 2,
                "expert_ids must be 2-dimension, but got ", expert_ids.dim(), " dimension");
    
    params.bs = x.size(0);
    params.h = x.size(1);
    params.k = expert_ids.size(1);
    
    TORCH_CHECK((params.h >= H_MIN) && (params.h <= params.maxH),
                "h (hidden size) is invalid, only support [", H_MIN, ", ", params.maxH, "], but got ", params.h);
    
    TORCH_CHECK((params.bs > 0) && (params.bs <= params.maxBs),
                "bs (batch size) is invalid, only support (0, ", params.maxBs, "], but got ", params.bs);
    
    TORCH_CHECK((params.k > 0) && (params.k <= params.maxK),
                "k (top-k) is invalid, only support (0, ", params.maxK, "], but got ", params.k);
    
    TORCH_CHECK(params.k <= params.moe_expert_num + params.zeroComputeExpertNum,
                "k (top-k) is invalid, should not exceed moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num. ",
                "k: ", params.k, ", moe_expert_num: ", params.moe_expert_num, 
                ", zero_expert_num: ", params.zero_expert_num, ", copy_expert_num: ", params.copy_expert_num,
                ", const_expert_num: ", params.const_expert_num);
}

static void ValidateOptionalTensorDim(const c10::optional<at::Tensor>& scales,
                                       const c10::optional<at::Tensor>& x_active_mask,
                                       const c10::optional<at::Tensor>& expert_scales,
                                       const c10::optional<at::Tensor>& elastic_info,
                                       const c10::optional<at::Tensor>& performance_info,
                                       MoeDistributeDispatchV2ValidateParams& params)
{
    params.isScales = scales.has_value();
    
    if (x_active_mask.has_value()) {
        const auto& x_active_mask_tensor = x_active_mask.value();
        TORCH_CHECK((x_active_mask_tensor.dim() == 1) || (x_active_mask_tensor.dim() == 2),
                    "x_active_mask must be 1-dimension or 2-dimension, but got ",
                    x_active_mask_tensor.dim(), " dimension");
        TORCH_CHECK(x_active_mask_tensor.size(0) == params.bs,
                    "x_active_mask's dim0 not equal to bs, x_active_mask: ", x_active_mask_tensor.size(0),
                    ", bs: ", params.bs);
        if (x_active_mask_tensor.dim() == 2) {
            TORCH_CHECK(x_active_mask_tensor.size(1) == params.k,
                        "x_active_mask's dim1 not equal to k, x_active_mask: ", x_active_mask_tensor.size(1),
                        ", k: ", params.k);
        }
    }
    
    if (expert_scales.has_value()) {
        const auto& expert_scales_tensor = expert_scales.value();
        TORCH_CHECK(expert_scales_tensor.dim() == 2,
                    "expert_scales must be 2-dimension, but got ", expert_scales_tensor.dim(), " dimension");
    }
    
    if (elastic_info.has_value()) {
        const auto& elastic_info_tensor = elastic_info.value();
        TORCH_CHECK(elastic_info_tensor.dim() == 1,
                    "elastic_info must be 1-dimension, but got ", elastic_info_tensor.dim(), " dimension");
    }
    
    if (performance_info.has_value()) {
        const auto& performance_info_tensor = performance_info.value();
        TORCH_CHECK(performance_info_tensor.dim() == 1,
                    "performance_info must be 1-dimension, but got ", performance_info_tensor.dim(), " dimension");
        TORCH_CHECK(performance_info_tensor.size(0) == params.ep_world_size,
                    "performance_info's dim0 not equal to ep_world_size, performance_info: ", performance_info_tensor.size(0),
                    ", ep_world_size: ", params.ep_world_size);
    }
}

static void ValidateInputDataType(const at::Tensor& x, const at::Tensor& expert_ids,
                                   const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf),
                "dtype of x should be BFloat16 or Float16, but got ", c10::toString(x.scalar_type()));
    
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got ", c10::toString(expert_ids.scalar_type()));
}

static void ValidateOptionalTensorDataType(const c10::optional<at::Tensor>& scales,
                                            const c10::optional<at::Tensor>& x_active_mask,
                                            const c10::optional<at::Tensor>& expert_scales,
                                            const c10::optional<at::Tensor>& elastic_info,
                                            const c10::optional<at::Tensor>& performance_info)
{
    if (scales.has_value()) {
        const auto& scales_tensor = scales.value();
        TORCH_CHECK(scales_tensor.scalar_type() == at::kFloat,
                    "scales datatype is invalid, datatype should be Float, but is ",
                    c10::toString(scales_tensor.scalar_type()));
    }
    
    if (x_active_mask.has_value()) {
        const auto& x_active_mask_tensor = x_active_mask.value();
        TORCH_CHECK(x_active_mask_tensor.scalar_type() == at::kBool,
                    "x_active_mask datatype is invalid, datatype should be Bool, but is ",
                    c10::toString(x_active_mask_tensor.scalar_type()));
    }
    
    if (expert_scales.has_value()) {
        const auto& expert_scales_tensor = expert_scales.value();
        TORCH_CHECK(expert_scales_tensor.scalar_type() == at::kFloat,
                    "expert_scales datatype is invalid, datatype should be Float, but is ",
                    c10::toString(expert_scales_tensor.scalar_type()));
    }
    
    if (elastic_info.has_value()) {
        const auto& elastic_info_tensor = elastic_info.value();
        TORCH_CHECK(elastic_info_tensor.scalar_type() == at::kInt,
                    "elastic_info datatype is invalid, datatype should be Int, but is ",
                    c10::toString(elastic_info_tensor.scalar_type()));
    }
    
    if (performance_info.has_value()) {
        const auto& performance_info_tensor = performance_info.value();
        TORCH_CHECK(performance_info_tensor.scalar_type() == at::kLong,
                    "performance_info datatype is invalid, datatype should be Long, but is ",
                    c10::toString(performance_info_tensor.scalar_type()));
    }
}

bool ValidateMoeDistributeDispatchV2Input(const at::Tensor& x, const at::Tensor& expert_ids,
                                           const c10::optional<at::Tensor>& scales,
                                           const c10::optional<at::Tensor>& x_active_mask,
                                           const c10::optional<at::Tensor>& expert_scales,
                                           const c10::optional<at::Tensor>& elastic_info,
                                           const c10::optional<at::Tensor>& performance_info,
                                           MoeDistributeDispatchV2ValidateParams& params)
{
    InitValidateContext(params);
    
    ValidateEpAttrs(params);
    
    ValidateSharedExpertAttrs(params);
    
    ValidateMoeExpertAttrs(params);
    
    ValidateQuantModeAndScales(params);
    
    ValidateCommonAttrs(params);
    
    ValidateInputTensorDim(x, expert_ids, params);
    
    ValidateOptionalTensorDim(scales, x_active_mask, expert_scales, elastic_info, performance_info, params);
    
    ValidateInputDataType(x, expert_ids, params);
    
    ValidateOptionalTensorDataType(scales, x_active_mask, expert_scales, elastic_info, performance_info);
    
    return true;
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops
