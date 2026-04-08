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

bool ValidateMoeDistributeDispatchV2Input(const at::Tensor& x, const at::Tensor& expert_ids,
                                           const c10::optional<at::Tensor>& scales,
                                           const c10::optional<at::Tensor>& x_active_mask,
                                           const c10::optional<at::Tensor>& expert_scales,
                                           const c10::optional<at::Tensor>& elastic_info,
                                           const c10::optional<at::Tensor>& performance_info,
                                           const MoeDistributeDispatchV2ValidateParams& params)
{
    int64_t maxEpWorldSize = MAX_EP_WORLD_SIZE;
    int64_t maxMoeExpertNum = MOE_EXPERT_MAX_NUM;
    int64_t maxH = H_MAX;
    int64_t maxBs = BS_UPPER_BOUND;
    
    if (params.comm_alg == "fullmesh_v2") {
        maxBs = FULLMESH_BS_UPPER_BOUND;
    }
    
    TORCH_CHECK((params.ep_world_size >= MIN_EP_WORLD_SIZE) && (params.ep_world_size <= maxEpWorldSize),
                "ep_world_size is invalid, only support [", MIN_EP_WORLD_SIZE, ", ", maxEpWorldSize, "], but got ",
                "ep_world_size: ", params.ep_world_size);
    
    TORCH_CHECK((params.ep_rank_id >= 0) && (params.ep_rank_id < params.ep_world_size),
                "ep_rank_id is invalid, only support [0, ep_world_size), but got",
                " ep_world_size: ", params.ep_world_size,
                ", ep_rank_id: ", params.ep_rank_id);
    
    TORCH_CHECK(params.expert_shard_type == 0,
                "expert_shard_type is invalid, only support 0, but got ", params.expert_shard_type);
    
    bool isSharedDefault = (params.shared_expert_num == 1) && (params.shared_expert_rank_num == 0);
    bool isNoShared = (params.shared_expert_num == 0) && (params.shared_expert_rank_num == 0);
    bool isValidShared = (params.shared_expert_num > 0) &&
                        ((params.shared_expert_rank_num / params.shared_expert_num) > 0) &&
                        ((params.shared_expert_rank_num % params.shared_expert_num) == 0);
    
    TORCH_CHECK(isSharedDefault || isNoShared || isValidShared,
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
    
    TORCH_CHECK((params.moe_expert_num > 0) && (params.moe_expert_num <= maxMoeExpertNum),
                "moe_expert_num is invalid, only support (0, ", maxMoeExpertNum, "], but got ",
                params.moe_expert_num);
    
    TORCH_CHECK((params.quant_mode >= static_cast<int64_t>(QuantModeA5::NON_QUANT)) && 
                (params.quant_mode <= static_cast<int64_t>(QuantModeA5::MX_QUANT)),
                "quant_mode is invalid, only support [", static_cast<int64_t>(QuantModeA5::NON_QUANT), 
                ", ", static_cast<int64_t>(QuantModeA5::MX_QUANT), "], but got ", params.quant_mode);
    
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
    
    int64_t zeroComputeExpertNum = params.zero_expert_num + params.copy_expert_num + params.const_expert_num;
    TORCH_CHECK((zeroComputeExpertNum + params.moe_expert_num) <= INT32_MAX,
                "zero_expert_num + copy_expert_num + const_expert_num + moe_expert_num exceeds INT32_MAX. ",
                "zero_expert_num: ", params.zero_expert_num, ", copy_expert_num: ", params.copy_expert_num,
                ", const_expert_num: ", params.const_expert_num, ", moe_expert_num: ", params.moe_expert_num);
    
    TORCH_CHECK(x.dim() == 2,
                "x must be 2-dimension, but got ", x.dim(), " dimension");
    
    TORCH_CHECK(expert_ids.dim() == 2,
                "expert_ids must be 2-dimension, but got ", expert_ids.dim(), " dimension");
    
    int64_t bs = x.size(0);
    int64_t h = x.size(1);
    int64_t k = expert_ids.size(1);
    
    TORCH_CHECK((h >= H_MIN) && (h <= maxH),
                "h (hidden size) is invalid, only support [", H_MIN, ", ", maxH, "], but got ", h);
    
    TORCH_CHECK((bs > 0) && (bs <= maxBs),
                "bs (batch size) is invalid, only support (0, ", maxBs, "], but got ", bs);
    
    int64_t maxK = (params.comm_alg == "fullmesh_v2") ? FULLMESH_K_MAX : K_MAX;
    TORCH_CHECK((k > 0) && (k <= maxK),
                "k (top-k) is invalid, only support (0, ", maxK, "], but got ", k);
    
    TORCH_CHECK(k <= params.moe_expert_num + zeroComputeExpertNum,
                "k (top-k) is invalid, should not exceed moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num. ",
                "k: ", k, ", moe_expert_num: ", params.moe_expert_num, 
                ", zero_expert_num: ", params.zero_expert_num, ", copy_expert_num: ", params.copy_expert_num,
                ", const_expert_num: ", params.const_expert_num);
    
    bool isScales = scales.has_value();
    if (isScales) {
        const auto& scales_tensor = scales.value();
        if (params.quant_mode == static_cast<int64_t>(QuantModeA5::STATIC_QUANT)) {
            TORCH_CHECK((scales_tensor.dim() == 1) || (scales_tensor.dim() == 2),
                        "scales must be 1-dimension or 2-dimension when quantMode is 1, but got ",
                        scales_tensor.dim(), " dimension");
        } else {
            TORCH_CHECK(scales_tensor.dim() == 2,
                        "scales must be 2-dimension, but got ", scales_tensor.dim(), " dimension");
        }
    }
    
    if (x_active_mask.has_value()) {
        const auto& x_active_mask_tensor = x_active_mask.value();
        TORCH_CHECK((x_active_mask_tensor.dim() == 1) || (x_active_mask_tensor.dim() == 2),
                    "x_active_mask must be 1-dimension or 2-dimension, but got ",
                    x_active_mask_tensor.dim(), " dimension");
        TORCH_CHECK(x_active_mask_tensor.size(0) == bs,
                    "x_active_mask's dim0 not equal to bs, x_active_mask: ", x_active_mask_tensor.size(0),
                    ", bs: ", bs);
        if (x_active_mask_tensor.dim() == 2) {
            TORCH_CHECK(x_active_mask_tensor.size(1) == k,
                        "x_active_mask's dim1 not equal to k, x_active_mask: ", x_active_mask_tensor.size(1),
                        ", k: ", k);
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
    
    TORCH_CHECK((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf),
                "dtype of x should be BFloat16 or Half, but got ", c10::toString(x.scalar_type()));
    
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got ", c10::toString(expert_ids.scalar_type()));

    TORCH_CHECK(isScales && params.quant_mode == static_cast<int64_t>(QuantModeA5::STATIC_QUANT),
                "The scales should not be nullptr when quantMode is ", params.quant_mode);

    TORCH_CHECK(isScales && params.quant_mode == static_cast<int64_t>(QuantModeA5::MX_QUANT),
                "The scales should be nullptr when quantMode is ", params.quant_mode);

    if (params.quant_mode == static_cast<int64_t>(QuantModeA5::NON_QUANT)) {
        TORCH_CHECK(isScales && ((x.scalar_type() == at::kFloat8_e4m3fn) || (x.scalar_type() == at::kFloat8_e5m2)),
                    "The scales should not be nullptr when quantMode is ", params.quant_mode,
                    " and X datatype is ", c10::toString(x.scalar_type()));
    }

    if (isScales) {
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
    
    return true;
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops