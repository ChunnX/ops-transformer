/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_torch_validate.h
 * \brief
 */

#pragma once

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>
#include <string>
#include <cstdint>

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

struct MoeDistributeDispatchV2ValidateParams {
    int64_t ep_world_size;
    int64_t ep_rank_id;
    int64_t moe_expert_num;
    int64_t expert_shard_type;
    int64_t shared_expert_num;
    int64_t shared_expert_rank_num;
    int64_t quant_mode;
    int64_t global_bs;
    int64_t expert_token_nums_type;
    std::string comm_alg;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;

    int64_t maxEpWorldSize;
    int64_t maxMoeExpertNum;
    int64_t maxH;
    int64_t maxBs;
    int64_t zeroComputeExpertNum;
    int64_t moeExpertNumDivisor;
    int64_t localMoeExpertNum;
    int64_t bs;
    int64_t h;
    int64_t k;
    int64_t maxK;
    bool isScales;
    bool isFullmeshV2;
    bool isSharedDefault;
    bool isNoShared;
    bool isValidShared;
};

bool ValidateMoeDistributeDispatchV2Input(const at::Tensor& x, const at::Tensor& expert_ids,
                                           const c10::optional<at::Tensor>& scales,
                                           const c10::optional<at::Tensor>& x_active_mask,
                                           const c10::optional<at::Tensor>& expert_scales,
                                           const c10::optional<at::Tensor>& elastic_info,
                                           const c10::optional<at::Tensor>& performance_info,
                                           MoeDistributeDispatchV2ValidateParams& params);

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops
