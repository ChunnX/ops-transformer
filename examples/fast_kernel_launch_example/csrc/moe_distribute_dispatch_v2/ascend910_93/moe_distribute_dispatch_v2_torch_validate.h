/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * Refer to LICENSE for details. This software is provided "AS IS" without warranties.
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_TORCH_VALIDATE_H
#define MOE_DISTRIBUTE_DISPATCH_V2_TORCH_VALIDATE_H

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <string>

namespace ascend_ops {
namespace MoeDistributeDispatchV2 {

/**
 * @brief 量化模式枚举
 */
enum class QuantModeA5 : uint32_t {
    NON_QUANT = 0,
    STATIC_QUANT = 1,
    PERTOKEN_DYNAMIC_QUANT = 2,
    PERGROUP_QUANT = 3,
    MX_QUANT = 4
};

/**
 * @brief 验证参数常量结构体
 * 
 * 包含所有用于输入验证的常量值，便于统一管理和调整
 */
struct ValidationConstants {
    static constexpr int64_t MIN_EP_WORLD_SIZE = 2;
    static constexpr int64_t MAX_EP_WORLD_SIZE = 768;
    static constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
    static constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
    static constexpr int64_t K_MAX = 16;
    static constexpr int64_t FULLMESH_K_MAX = 12;
    static constexpr int64_t H_MIN = 1024;
    static constexpr int64_t H_MAX = 8192;
    static constexpr int64_t BS_UPPER_BOUND = 512;
    static constexpr int64_t FULLMESH_BS_UPPER_BOUND = 256;
    static constexpr int64_t CEIL_ALIGN32 = 8;
};

/**
 * @brief MOE分布式调度V2验证参数结构体
 * 
 * 用于验证MOE分布式调度V2操作的输入参数合法性
 */
struct MoeDistributeDispatchV2ValidateParams {
    // 必需属性参数
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
    
    // 计算出的中间变量
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
    
    // 状态标志
    bool isScales;
    bool isFullmeshV2;
    bool isNoShared;
    bool isValidShared;
};

/**
 * @brief 验证MOE分布式调度V2输入参数
 * 
 * @param x 输入张量 [bs, h]
 * @param expert_ids 专家ID张量 [bs, k]
 * @param scales 可选: 缩放张量
 * @param x_active_mask 可选: 激活掩码张量
 * @param expert_scales 可选: 专家缩放张量
 * @param performance_info 可选: 性能信息张量
 * @param params 验证参结构体
 * @return 验证通过返回true，失败则抛出TorchCheckError异常
 */
bool ValidateMoeDistributeDispatchV2Input(
    const at::Tensor& x,
    const at::Tensor& expert_ids,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& expert_scales,
    const c10::optional<at::Tensor>& performance_info,
    MoeDistributeDispatchV2ValidateParams& params);

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops

#endif // MOE_DISTRIBUTE_DISPATCH_V2_TORCH_VALIDATE_H
