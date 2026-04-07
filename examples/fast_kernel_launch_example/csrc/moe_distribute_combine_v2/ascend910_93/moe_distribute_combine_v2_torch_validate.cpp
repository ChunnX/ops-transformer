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
 * \file moeoe_distribute_combine_v2_torch_validate.cpp
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
constexpr int64ceil_align32 = 8;
constexpr int64_t INT8_COMM_QUANT = 2;

enum class CommQuantMode : int32_t {
    NON_QUANT = 0,
    INT12_QUANT = 1,
    INT8_QUANT = 2,
    MXFP8_E5M2_QUANT = 3,
    MXFP8_E4M3_QUANTES = 4
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
                                         const MoeDistributeCombineV2ValidateParams& params)
{
    int64_t maxEpWorldSize = MAX_EP_WORLD_SIZE;
    int64_t maxMoeExpertNum = MOE_EXPERT_MAX_NUM;
    int64_t maxH = H_MAX;
    int64_t maxBs = BS_UPPER_BOUND;
    
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
    bool isNoShared = (params.shared_expert_num == 0) && (params.shared_expert_rank_num == == 0);
    bool isValidShared = (params.shared_expert_num > 0) &&
                        ((params.shared_expert_rank_num / params.shared_expert_num) > 0) &&
                        ((params.shared_expert_rank_num % params.shared_expert_num) == 0);
    
    TORCH_CHECK(isSharedDefault || isNoShared || isValidShared,
                "shared_expert_num and shared_expert_rank_num have obvious value situations: ",
                "1. shared_expert_num is 1, shared_expert_rank_num is 0; 2. shared_expert_num is 0, shared_expert_rank_num is 0; ",
                "3. shared_expert_num in (0, shared_expert_rank_num] and shared_expert_rank_num % shared_expert_num = 0. ",
                "but the current input value is shared_expert_num: ", params.shared_expert_num,
                ", shared_expert_rank_num: ", params.shared_expert_rank_num);
    
    TORCH_CHECK((params.shared_expert_rank_num >= 0) && (params.shared_expert_rank_num < params.ep_world_size),
                "shared_expert_rank_num is invalid, only support [0, ep_world_size), but got",
                " ep_world_size: ", params.ep_world_size,
                ", shared_expert_rank_num: ", params.shared_expert_rank_num);
    
    TORCH_CHECK((params.moe_expert_num > 0) && (params.moe_expert_num <= maxMoeExpertNum),
                "moe_expert_num is invalid, only support (0, ", maxMoeExpertNum, "], but got ",
                params.moe_expert_num);
    
    TORCH_CHECK((params.comm_quant_mode >= static_cast<int64_t>(CommQuantMode::NON_QUANT)) && 
                (params.comm_quant_mode <= static_cast<int64_t>(CommQuantMode::MXFP8_E4M3_QUANTES)),
                "comm_quant_mode is invalid, only support [", static_cast<int64_t>(CommQuantMode::NON_QUANT), 
                ", ", static_cast<int64_t>(CommQuantMode::MXFP8_E4M3_QUANTES), "], but got ",
                params.comm_quant_mode);
    
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
    
    TORCH_CHECK(expand_x.dim() == 2,
                "expand_x must be 2-dimension, but got ", expand_x.dim(), " dimension");
    
    TORCH_CHECK(expert_ids.dim() == 2,
                "expert_ids must be 2-dimension, but got ", expert_ids.dim(), " dimension");
    
    TORCH_CHECK(assist_info_for_combine.dim() == 1,
                "assist_info_for_combine must be 1-dimension, but got ", assist_info_for_combine.dim(), " dimension");
    
    TORCH_CHECK(ep_send_counts.dim() == 1,
                "ep_send_counts must be 1-dimension, but got ", ep_send_counts.dim(), " dimension");
    
    TORCH_CHECK(expert_scales.dim() == 2,
                "expert_scales must be 2-dimension, but got ", expert_scales.dim(), " dimension");
    
    int64_t bs = expert_ids.size(0);
    int64_t h = expand_x.size(1);
    int64_t k = expert_ids.size(1);
    
    TORCH_CHECK((h >= H_MIN) && (h <= H_MAX),
                "h (hidden size) is invalid, only support [", H_MIN, ", ", H_MAX, "], but got ", h);
    
    TORCH_CHECK((bs > 0) && (bs <= BS_UPPER_BOUND),
                "bs (batch size) is invalid, only support (0, ", BS_UPPER_BOUND, "], but got ", bs);
    
    TORCH_CHECK((k > 0) && (k <= K_MAX),
                "k (top-k) is invalid, only support (0, ", K_MAX, "], but got ", k);
    
    TORCH_CHECK(k <= params.moe_expert_num + zeroComputeExpertNum,
                "k (top-k) is invalid, should not exceed moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num. ",
                "k: ", k, ", moe_expert_num: ", params.moe_expert_num, ", zero_expert_num: ", params.zero_expert_num,
                ", copy_expert_num: ", params.copy_expert_num, ", const_expert_num: ", params.const_expert_num);
    
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
    
    if (shared_expert_x.has_value()) {
        const auto& shared_expert_x_tensor = shared_expert_x.value();
        TORCH_CHECK((shared_expert_x_tensor.dim() == 2) || (shared_expert_x_tensor.dim() == 3),
                    "shared_expert_x must be 2-dimension or 3-dimension, but got ",
                    shared_expert_x_tensor.dim(), " dimension");
    }
    
    if (ori_x.has_value()) {
        const auto& ori_x_tensor = ori_x.value();
        TORCH_CHECK(ori_x_tensor.dim() == 2,
                    "ori_x must be 2-dimension, but got ", ori_x_tensor.dim(), " dimension");
        TORCH_CHECK(ori_x_tensor.size(0) == bs,
                    "ori_x's dim0 not equal to bs, ori_x: ", ori_x_tensor.size(0),
                    ", bs: ", bs);
        TORCH_CHECK(ori_x_tensor.size(1) == h,
                    "ori_x's dim1 not equal to h, ori_x: ", ori_x_tensor.size(1),
                    ", h: ", h);
    }
    
    if (const_expert_alpha_1.has_value()) {
        const auto& const_expert_alpha_1_tensor = const_expert_alpha_1.value();
        TORCH_CHECK(const_expert_alpha_1_tensor.dim() == 2,
                    "const_expert_alpha_1 must be 2-dimension, but got ",
                    const_expert_alpha_1_tensor.dim(), " dimension");
    }
    
    if (const_expert_alpha_2.has_value()) {
        const auto& const_expert_alpha_2_tensor = const_expert_alpha_2.value();
        TORCH_CHECK(const_expert_alpha_2_tensor.dim() == 2,
                    "const_expert_alpha_2 must be 2-dimension, but got ",
                    const_expert_alpha_2_tensor.dim(), " dimension");
    }
    
    if (const_expert_v.has_value()) {
        const auto& const_expert_v_tensor = const_expert_v.value();
        TORCH_CHECK(const_expert_v_tensor.dim() == 2,
                    "const_expert_v must be 2-dimension, but got ",
                    const_expert_v_tensor.dim(), " dimension");
    }
    
    if (performance_info.has_value()) {
        const auto& performance_info_tensor = performance_info.value();
        TORCH_CHECK(performance_info_tensor.dim() == 1,
                    "performance_info must be 1-dimension, but got ",
                    performance_info_tensor.dim(), " dimension");
        TORCH_CHECK(performance_info_tensor.size(0) == params.ep_world_size,
                    "performance_info's dim0 not equal to ep_world_size, performance_info: ", performance_info_tensor.size(0),
                    ", ep_world_size: ", params.ep_world_size);
    }
    
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf),
                "expand_x datatype is invalid, datatype should be BFloat16 or Half, but is ",
                c10::toString(expand_x.scalar_type()));
    
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "expert_ids datatype is invalid, datatype should be Int, but is ",
                c10::toString(expert_ids.scalar_type()));
    
    TORCH_CHECK(assist_info_for_combine.scalar_type() == at::kInt,
                "assist_info_for_combine datatype is invalid, datatype should be Int, but is ",
                c10::toString(assist_info_for_combine.scalar_type()));
    
    TORCH_CHECK(ep_send_counts.scalar_type() == at::kInt,
                "ep_send_counts datatype is invalid, datatype should be Int, but is ",
                c10::toString(ep_send_counts.scalar_type()));
    
    TORCH_CHECK(expert_scales.scalar_type() == at::kFloat,
                "expert_scales datatype is invalid, datatype should be Float, but is ",
                c10::toString(expert_scales.scalar_type()));
    
    if (x_active_mask.has_value()) {
        const auto& x_active_mask_tensor = x_active_mask.value();
        TORCH_CHECK(x_active_mask_tensor.scalar_type() == at::kBool,
                    "x_active_mask datatype is invalid, datatype should be Bool, but is ",
                    c10::toString(x_active_mask_tensor.scalar_type()));
    }
    
    if (shared_expert_x.has_value()) {
        const auto& shared_expert_x_tensor = shared_expert_x.value();
        TORCH_CHECK(shared_expert_x_tensor.scalar_type() == expand_x.scalar_type(),
                    "shared_expert_x datatype is invalid, datatype should be same as expand_x datatype ",
                    c10::toString(expand_x.scalar_type()), ", but is ",
                    c10::toString(shared_expert_x_tensor.scalar_type()));
    }
    
    if (ori_x.has_value()) {
        const auto& ori_x_tensor = ori_x.value();
        TORCH_CHECK(ori_x_tensor.scalar_type() == expand_x.scalar_type(),
                    "ori_x datatype is invalid, datatype should be same as expand_x datatype ",
                    c10::toString(expand_x.scalar_type()), ", but is ",
                    c10::toString(ori_x_tensor.scalar_type()));
    }
    
    if (const_expert_alpha_1.has_value()) {
        const auto& const_expert_alpha_1_tensor = const_expert_alpha_1.value();
        TORCH_CHECK(const_expert_alpha_1_tensor.scalar_type() == expand_x.scalar_type(),
                    "const_expert_alpha_1 datatype is invalid, datatype should be same as expand_x datatype ",
                    c10::toString(expand_x.scalar_type()), ", but is ",
                    c10::toString(const_expert_alpha_1_tensor.scalar_type()));
    }
    
    if (const_expert_alpha_2.has_value()) {
        const auto& const_expert_alpha_2_tensor = const_expert_alpha_2.value();
        TORCH_CHECK(const_expert_alpha_2_tensor.scalar_type() == expand_x.scalar_type(),
                    "const_expert_alpha_2 datatype is invalid, datatype should be same as expand_x datatype ",
                    c10::toString(expand_x.scalar_type()), ", but is ",
                    c10::toString(const_expert_alpha_2_tensor.scalar_type()));
    }
    
    if (const_expert_v.has_value()) {
        const auto& const_expert_v_tensor = const_expert_v.value();
        TORCH_CHECK(const_expert_v_tensor.scalar_type() == expand_x.scalar_type(),
                    "const_expert_v datatype is invalid, datatype should be same as expand_x datatype ",
                    c10::toString(expand_x.scalar_type()), ", but is ",
                    c10::toString(const_expert_v_tensor.scalar_type()));
    }
    
    if (performance_info.has_value()) {
        const auto& performance_info_tensor = performance_info.value();
        TORCH_CHECK(performance_info_tensor.scalar_type() == at::kLong,
                    "performance_info datatype is invalid, datatype should be Long, but is ",
                    c10::toString(performance_info_tensor.scalar_type()));
    }
    
    return true;
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops