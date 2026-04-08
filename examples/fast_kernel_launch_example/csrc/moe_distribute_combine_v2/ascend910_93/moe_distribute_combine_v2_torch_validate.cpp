/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * Refer to LICENSE for details. This software is provided "AS IS" without warranties.
 */

#include "moe_distribute_combine_v2_torch_validate.h"
#include <torch/torch.h>

namespace ascend_ops {
namespace MoeDistributeCombineV2 {

// ============================================================================
// 公共辅助函数
// ============================================================================

/**
 * @brief 初始化验证上下文
 * @param params 验证参数结构体引用
 * @note 计算派生参数和共享专家状态标志
 */
static void InitValidateContext(MoeDistributeCombineV2ValidateParams& params) {
    params.maxEpWorldSize = ValidationConstants::MAX_EP_WORLD_SIZE;
    params.maxMoeExpertNum = ValidationConstants::MOE_EXPERT_MAX_NUM;
    
    // 计算零计算专家数量
    params.zeroComputeExpertNum = params.zero_expert_num + params.copy_expert_num + params.const_expert_num;
    
    // 计算MOE专家除数
    params.moeExpertNumDivisor = params.ep_world_size - params.shared_expert_rank_num;
    params.localMoeExpertNum = params.moe_expert_num / params.moeExpertNumDivisor;
    
    // 计算共享专家状态标志
    params.isNoShared = (params.shared_expert_num == 0) && (params.shared_expert_rank_num == 0);
    params.isValidShared = (params.shared_expert_num > 0) &&
                       ((params.shared_expert_rank_num / params.shared_expert_num) > 0) &&
                       ((params.shared_expert_rank_num % params.shared_expert_num) == 0);
}

// ============================================================================
// EP相关属性验证
// ============================================================================

/**
 * @brief 验证端点属性
 * @param params 验证参数结构体常量引用
 * 
 * 验证ep_world_size和ep_rank_id的合法性
 */
static void ValidateEpAttrs(const MoeDistributeCombineV2ValidateParams& params) {
    TORCH_CHECK((params.ep_world_size >= ValidationConstants::MIN_EP_WORLD_SIZE) && 
                (params.ep_world_size <= params.maxEpWorldSize),
                "ep_world_size invalid: support [", ValidationConstants::MIN_EP_WORLD_SIZE, 
                ", ", params.maxEpWorldSize, "], got: ", params.ep_world_size);
    
    TORCH_CHECK((params.ep_rank_id >= 0) && (params.ep_rank_id < params.ep_world_size),
                "ep_rank_id invalid: support [0, ep_world_size), ep_world_size=", params.ep_world_size,
                ", ep_rank_id=", params.ep_rank_id);
}

/**
 * @brief 验证共享专家属性
 * @param params 验证参数结构体常量引用
 * 
 * 验证shared_expert_num和shared_expert_rank_num的组合合法性
 */
static void ValidateSharedExpertAttrs(const MoeDistributeCombineV2ValidateParams& params) {
    // 验证三种有效的共享专家配置场景
    TORCH_CHECK(params.isNoShared || params.isValidShared,
                "shared_expert_num/rank_num invalid configs. Valid: (0,0), or (n,m) where m%n==0. ",
                "Got: shared_expert_num=", params.shared_expert_num,
                ", shared_expert_rank_num=", params.shared_expert_rank_num);
    
    // 验证shared_expert_rank_num范围
    TORCH_CHECK((params.shared_expert_rank_num >= 0) && 
                (params.shared_expert_rank_num < params.ep_world_size),
                "shared_expert_rank_num invalid: [0, ep_world_size), ep_world_size=", params.ep_world_size,
                ", rank_num=", params.shared_expert_rank_num);
}

/**
 * @brief 验证MOE专家属性
 * @param params 验证参数结构体常量引用
 * 
 * 验证moe_expert_num及其与ep_world_size的关系
 */
static void ValidateMoeExpertAttrs(const MoeDistributeCombineV2ValidateParams& params) {
    // 验证moe_expert_num范围
    TORCH_CHECK((params.moe_expert_num > 0) && (params.moe_expert_num <= params.maxMoeExpertNum),
                "moe_expert_num invalid: (0, ", params.maxMoeExpertNum, "], got: ", params.moe_expert_num);
    
    // 验证除数有效性
    TORCH_CHECK(params.moeExpertNumDivisor > 0,
                "moeExpertNumDivisor invalid: ep_world_size=", params.ep_world_size,
                " - shared_expert_rank_num=", params.shared_expert_rank_num, " must be > 0");
    
    // 验证整除关系
    TORCH_CHECK(params.moe_expert_num % params.moeExpertNumDivisor == 0,
                "moe_expert_num divisible check failed: num=", params.moe_expert_num,
                ", divisor=", params.moeExpertNumDivisor);
    
    // 验证localMoeExpertNum有效性
    TORCH_CHECK(params.localMoeExpertNum > 0,
                "localMoeExpertNum invalid: ", params.localMoeExpertNum);
    
    // 验证localMoeExpertNum * ep_world_size不超过限制
    TORCH_CHECK(params.localMoeExpertNum * params.ep_world_size <= ValidationConstants::LOCAL_EXPERT_MAX_SIZE,
                "localMoeExpertNum * ep_world_size exceeds ", ValidationConstants::LOCAL_EXPERT_MAX_SIZE,
                ": got ", params.localMoeExpertNum * params.ep_world_size);
}

/**
 * @brief 验证量化模式
 * @param params 验证参数结构体常量引用
 * 
 * 验证comm_quant_mode在支持范围内
 */
static void ValidateQuantMode(const MoeDistributeCombineV2ValidateParams& params) {
    TORCH_CHECK((params.comm_quant_mode >= static_cast<int64_t>(CommQuantMode::NON_QUANT)) && 
                (params.comm_quant_mode <= static_cast<int64_t>(CommQuantMode::MXFP8_E4M3_QUANT)),
                "comm_quant_mode invalid: [", static_cast<int64_t>(CommQuantMode::NON_QUANT), 
                ", ", static_cast<int64_t>(CommQuantMode::MXFP8_E4M3_QUANT), 
                "], got: ", params.comm_quant_mode);
}

/**
 * @brief 验证公共属性
 * @param params 验证参数结构体常量引用
 * 
 * 验证expert_shard_type、global_bs、zero/copy/const_expert_num等
 */
static void ValidateCommonAttrs(const MoeDistributeCombineV2ValidateParams& params) {
    TORCH_CHECK(params.expert_shard_type == 0,
                "expert_shard_type invalid: only 0 supported, got: ", params.expert_shard_type);
    
    TORCH_CHECK(params.global_bs >= 0,
                "global_bs invalid: >= 0 required, got: ", params.global_bs);
    
    TORCH_CHECK(params.zero_expert_num >= 0,
                "zero_expert_num invalid: >= 0 required, got: ", params.zero_expert_num);
    
    TORCH_CHECK(params.copy_expert_num >= 0,
                "copy_expert_num invalid: >= 0 required, got: ", params.copy_expert_num);
    
    TORCH_CHECK(params.const_expert_num >= 0,
                "const_expert_num invalid: >= 0 required, got: ", params.const_expert_num);
    
    // 验证总专家数量不超过INT32_MAX
    TORCH_CHECK((params.zeroComputeExpertNum + params.moe_expert_num) <= INT32_MAX,
                "expert count exceeds INT32_MAX: zero=", params.zero_expert_num,
                ", copy=", params.copy_expert_num,
                ", const=", params.const_expert_num,
                ", moe=", params.moe_expert_num);
}

/**
 * @brief 验证全局批次大小
 * @param params 验证参数结构体常量引用
 * @param bs 批次大小
 * @param hasActiveMask 是否存在激活掩码
 * 
 * 验证global_bs与bs和ep_world_size的关系
 */
static void ValidateGlobalBs(const MoeDistributeCombineV2ValidateParams& params,
                            int64_t bs,
                            bool hasActiveMask) {
    TORCH_CHECK(!((params.global_bs != 0) && 
                ((params.global_bs < params.ep_world_size * bs) || 
                 (params.global_bs % params.ep_world_size != 0))),
                "global_bs invalid: only support 0 or max_bs * ep_world_size, "
                "but got global_bs=", params.global_bs,
                ", bs=", bs,
                ", ep_world_size=", params.ep_world_size);
    
    if (hasActiveMask) {
        TORCH_CHECK(params.global_bs <= params.ep_world_size * bs,
                    "Variable bs on different rank cannot work when isActiveMask=true, "
                    "global_bs=", params.global_bs,
                    ", bs=", bs,
                    ", ep_world_size=", params.ep_world_size);
    }
}

// ============================================================================
// 张量维度验证函数
// ============================================================================

/**
 * @brief 验证必需张量维度
 * @param expand_x 扩展输入张量
 * @param expert_ids 专家ID张量
 * @param assist_info 辅助信息张量
 * @param ep_send_counts EP发送计数张量
 * @param expert_scales 专家缩放张量
 * 
 * 验证所有必需输入张量的维度合法性
 */
static void ValidateRequiredTensorDims(const at::Tensor& expand_x,
                                     const at::Tensor& expert_ids,
                                     const at::Tensor& assist_info_for_combine,
                                     const at::Tensor& ep_send_counts,
                                     const at::Tensor& expert_scales) {
    TORCH_CHECK(expand_x.dim() == 2,
                "expand_x dim invalid: 2D required, got ", expand_x.dim());
    
    TORCH_CHECK(expert_ids.dim() == 2,
                "expert_ids dim invalid: 2D required, got ", expert_ids.dim());
    
    TORCH_CHECK(assist_info_for_combine.dim() == 1,
                "assist_info dim invalid: 1D required, got ", assist_info_for_combine.dim());
    
    TORCH_CHECK(ep_send_counts.dim() == 1,
                "ep_send_counts dim invalid: 1D required, got ", ep_send_counts.dim());
    
    TORCH_CHECK(expert_scales.dim() == 2,
                "expert_scales dim invalid: 2D required, got ", expert_scales.dim());
}

/**
 * @brief 验证张量维度范围值
 * @param expand_x 扩展输入张量
 * @param expert_ids 专家ID张量
 * @param params 验证参数结构体常量引用
 * @param bs 输出: 批量大小
 * @param h 输出: 隐藏层大小
 * @param k 输出: top-k值
 * 
 * 验证h、bs、k的具体取值范围
 */
static void ValidateDimValues(const at::Tensor& expand_x,
                            const at::Tensor& expert_ids,
                            const MoeDistributeCombineV2ValidateParams& params,
                            int64_t& bs, int64_t& h, int64_t& k) {
    bs = expert_ids.size(0);
    h = expand_x.size(1);
    k = expert_ids.size(1);
    
    TORCH_CHECK((h >= ValidationConstants::H_MIN) && (h <= ValidationConstants::H_MAX),
                "h invalid: [", ValidationConstants::H_MIN, ", ", ValidationConstants::H_MAX, 
                "], got: ", h);
    
    TORCH_CHECK((bs > 0) && (bs <= ValidationConstants::BS_UPPER_BOUND),
                "bs invalid: (0, ", ValidationConstants::BS_UPPER_BOUND, "], got: ", bs);
    
    TORCH_CHECK((k > 0) && (k <= ValidationConstants::K_MAX),
                "k invalid: (0, ", ValidationConstants::K_MAX, "], got: ", k);
    
    TORCH_CHECK(k <= params.moe_expert_num + params.zeroComputeExpertNum,
                "k exceeds expert_num limit: k=", k,
                ", moe=", params.moe_expert_num,
                ", zero_compute=", params.zeroComputeExpertNum);
}

/**
 * @brief 验证可选张量维度
 * @param x_active_mask 激活掩码张量
 * @param shared_expert_x 共享专家输入张量
 * @param ori_x 原始输入张量
 * @param const_alpha1 常数专家alpha1张量
 * @param const_alpha2 常数专家alpha2张量
 * @param const_v 常数专家V张量
 * @param performance_info 性能信息张量
 * @param bs 批量大小
 * @param h 隐藏层大小
 * @param k top-k值
 * @param params 验证参数结构体常量引用
 */
static void ValidateOptionalDims(const c10::optional<at::Tensor>& x_active_mask,
                               const c10::optional<at::Tensor>& shared_expert_x,
                               const c10::optional<at::Tensor>& ori_x,
                               const c10::optional<at::Tensor>& const_alpha1,
                               const c10::optional<at::Tensor>& const_alpha2,
                               const c10::optional<at::Tensor>& const_v,
                               const c10::optional<at::Tensor>& performance_info,
                               int64_t bs, int64_t h, int64_t k,
                               const MoeDistributeCombineV2ValidateParams& params) {
    // x_active_mask验证
    if (x_active_mask.has_value()) {
        const auto& mask = x_active_mask.value();
        TORCH_CHECK((mask.dim() == 1) || (mask.dim() == 2),
                    "x_active_mask dim invalid: 1D or 2D, got ", mask.dim());
        TORCH_CHECK(mask.size(0) == bs,
                    "x_active_mask dim0 mismatch: expected ", bs, ", got ", mask.size(0));
        if (mask.dim() == 2) {
            TORCH_CHECK(mask.size(1) == k,
                        "x_active_mask dim1 mismatch: expected ", k, ", got ", mask.size(1));
        }
    }
    
    // shared_expert_x验证
    if (shared_expert_x.has_value()) {
        const auto& sxp = shared_expert_x.value();
        TORCH_CHECK((sxp.dim() == 2) || (sxp.dim() == 3),
                    "shared_expert_x dim invalid: 2D or 3D, got ", sxp.dim());
    }
    
    // ori_x验证
    if (ori_x.has_value()) {
        const auto& ox = ori_x.value();
        TORCH_CHECK(ox.dim() == 2, "ori_x dim invalid: 2D, got ", ox.dim());
        TORCH_CHECK(ox.size(0) == bs, "ori_x dim0 mismatch: ", bs, ", got ", ox.size(0));
        TORCH_CHECK(ox.size(1) == h, "ori_x dim1 mismatch: ", h, ", got ", ox.size(1));
    }
    
    // const_expert_alpha_1/2/v验证
    if (const_alpha1.has_value()) {
        TORCH_CHECK(const_alpha1.value().dim() == 2,
                    "const_expert_alpha_1 dim invalid: 2D, got ", const_alpha1.value().dim());
    }
    if (const_alpha2.has_value()) {
        TORCH_CHECK(const_alpha2.value().dim() == 2,
                    "const_expert_alpha_2 dim invalid: 2D, got ", const_alpha2.value().dim());
    }
    if (const_v.has_value()) {
        TORCH_CHECK(const_v.value().dim() == 2,
                    "const_expert_v dim invalid: 2D, got ", const_v.value().dim());
    }
    
    // performance_info验证
    if (performance_info.has_value()) {
        const auto& perf = performance_info.value();
        TORCH_CHECK(perf.dim() == 1, "performance_info dim invalid: 1D, got ", perf.dim());
        TORCH_CHECK(perf.size(0) == params.ep_world_size,
                    "performance_info size mismatch: ", params.ep_world_size, ", got ", perf.size(0));
    }
}

// ============================================================================
// 张量数据类型验证函数
// ============================================================================

/**
 * @brief 验证必需张量数据类型
 * @param expand_x 扩展输入张量
 * @param expert_ids 专家ID张量
 * @param assist_info 辅助信息张量
 * @param ep_send_counts EP发送计数张量
 * @param expert_scales 专家缩放张量
 * 
 * 验证必需张量的数据类型合法性
 */
static void ValidateRequiredDataTypes(const at::Tensor& expand_x,
                                     const at::Tensor& expert_ids,
                                     const at::Tensor& assist_info_for_combine,
                                     const at::Tensor& ep_send_counts,
                                     const at::Tensor& expert_scales) {
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || 
                (expand_x.scalar_type() == at::kHalf),
                "expand_x dtype invalid: BF16 or FP16 required, got ", c10::toString(expand_x.scalar_type()));
    
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "expert_ids dtype invalid: Int32 required, got ", c10::toString(expert_ids.scalar_type()));
    
    TORCH_CHECK(assist_info_for_combine.scalar_type() == at::kInt,
                "assist_info dtype invalid: Int32 required, got ", 
                c10::toString(assist_info_for_combine.scalar_type()));
    
    TORCH_CHECK(ep_send_counts.scalar_type() == at::kInt,
                "ep_send_counts dtype invalid: Int32 required, got ", 
                c10::toString(ep_send_counts.scalar_type()));
    
    TORCH_CHECK(expert_scales.scalar_type() == at::kFloat,
                "expert_scales dtype invalid: Float required, got ", 
                c10::toString(expert_scales.scalar_type()));
}

/**
 * @brief 验证可选张量数据类型
 * @param x_active_mask 激活掩码张量
 * @param shared_expert_x 共享专家输入张量
 * @param ori_x 原始输入张量
 * @param const_alpha1/2/v 常数专家张量
 * @param performance_info 性能信息张量
 * @param expand_x 扩展输入张量(作为dtype参考)
 */
static void ValidateOptionalDataTypes(const c10::optional<at::Tensor>& x_active_mask,
                                     const c10::optional<at::Tensor>& shared_expert_x,
                                     const c10::optional<at::Tensor>& ori_x,
                                     const c10::optional<at::Tensor>& const_alpha1,
                                     const c10::optional<at::Tensor>& const_alpha2,
                                     const c10::optional<at::Tensor>& const_v,
                                     const c10::optional<at::Tensor>& performance_info,
                                     const at::Tensor& expand_x) {
    if (x_active_mask.has_value()) {
        const auto& mask = x_active_mask.value();
        TORCH_CHECK(mask.scalar_type() == at::kBool,
                    "x_active_mask dtype invalid: Bool required, got ", c10::toString(mask.scalar_type()));
    }
    
    if (shared_expert_x.has_value()) {
        const auto& sxp = shared_expert_x.value();
        TORCH_CHECK(sxp.scalar_type() == expand_x.scalar_type(),
                    "shared_expert_x dtype mismatch: expected ", 
                    c10::toString(expand_x.scalar_type()), 
                    ", got ", c10::toString(sxp.scalar_type()));
    }
    
    if (ori_x.has_value()) {
        const auto& ox = ori_x.value();
        TORCH_CHECK(ox.scalar_type() == expand_x.scalar_type(),
                    "ori_x dtype mismatch: ", c10::toString(expand_x.scalar_type()),
                    ", got ", c10::toString(ox.scalar_type()));
    }
    
    if (const_alpha1.has_value()) {
        const auto& ca1 = const_alpha1.value();
        TORCH_CHECK(ca1.scalar_type() == expand_x.scalar_type(),
                    "const_expert_alpha_1 dtype mismatch: ", c10::toString(expand_x.scalar_type()));
    }
    
    if (const_alpha2.has_value()) {
        const auto& ca2 = const_alpha2.value();
        TORCH_CHECK(ca2.scalar_type() == expand_x.scalar_type(),
                    "const_expert_alpha_2 dtype mismatch: ", c10::toString(expand_x.scalar_type()));
    }
    
    if (const_v.has_value()) {
        const auto& cv = const_v.value();
        TORCH_CHECK(cv.scalar_type() == expand_x.scalar_type(),
                    "const_expert_v dtype mismatch: ", c10::toString(expand_x.scalar_type()));
    }
    
    if (performance_info.has_value()) {
        const auto& perf = performance_info.value();
        TORCH_CHECK(perf.scalar_type() == at::kLong,
                    "performance_info dtype invalid: Int64 required, got ", c10::toString(perf.scalar_type()));
    }
}

// ============================================================================
// 主验证函数
// ============================================================================

bool ValidateMoeDistributeCombineV2Input(
    const at::Tensor& expand_x,
    const at::Tensor& expert_ids,
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
    MoeDistributeCombineV2ValidateParams& params) {
    
    // 第一阶段: 初始化和属性验证
    InitValidateContext(params);
    ValidateEpAttrs(params);
    ValidateSharedExpertAttrs(params);
    ValidateMoeExpertAttrs(params);
    ValidateQuantMode(params);
    ValidateCommonAttrs(params);
    
    // 第二阶段: 张量维度验证
    int64_t bs, h, k;
    ValidateRequiredTensorDims(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales);
    ValidateDimValues(expand_x, expert_ids, params, bs, h, k);
    ValidateOptionalDims(x_active_mask, shared_expert_x, ori_x, 
                       const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
                       performance_info, bs, h, k, params);
    
    // 验证global_bs与bs的关系
    ValidateGlobalBs(params, bs, x_active_mask.has_value());
    
    // 第三阶段: 数据类型验证
    ValidateRequiredDataTypes(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales);
    ValidateOptionalDataTypes(x_active_mask, shared_expert_x, ori_x,
                            const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
                            performance_info, expand_x);
    
    return true;
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops
