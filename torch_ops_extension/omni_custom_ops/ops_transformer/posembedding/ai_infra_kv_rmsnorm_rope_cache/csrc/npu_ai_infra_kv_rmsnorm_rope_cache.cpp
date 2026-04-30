/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {

using namespace at_npu::native;

// v1 接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> construct_output_tensor_v1(
    const at::Tensor &kv,
    const at::Tensor &gamma,
    const at::Tensor &k_cache,
    const at::Tensor &ckv_cache,
    bool is_output_kv)
{
    // 根据kv相关维度构造k_rope
    int64_t dim_0 = kv.size(0);     // Bkv
    int64_t dim_1 = kv.size(1);     // N
    int64_t dim_2 = kv.size(2);     // Skv
    int64_t dim_3 = kv.size(3);     // Dnope + Drope
    int64_t dim_3_nope = gamma.size(0); // rms_size
    int64_t dim_3_rope = dim_3 - dim_3_nope; // rope_size

    // k_cache_out、ckv_cache_out 根据k_cache、ckv_cache的shape和dtype创建
    at::Tensor k_cache_out;
    at::Tensor ckv_cache_out = at::empty_like(ckv_cache);

    if (!k_cache.defined()) {
        k_cache_out = at::empty({dim_0, dim_1, dim_2, dim_3_rope}, ckv_cache.options().dtype(ckv_cache.scalar_type()));
    }
    else {
        k_cache_out = at::empty_like(k_cache);
    }

    at::Tensor k_rope = at::empty({dim_0, 1, dim_2, dim_3_rope}, kv.options().dtype(kv.scalar_type()));
    // 根据 gamma 构造 c_kv
    at::Tensor c_kv = at::empty({dim_0, 1, dim_2, dim_3_nope}, kv.options().dtype(kv.scalar_type()));

    return std::make_tuple(k_cache_out, ckv_cache_out, k_rope, c_kv);
}

// v2 接口，原地操作
std::tuple<at::Tensor, at::Tensor> construct_output_tensor_v2(
    const at::Tensor &kv,
    const at::Tensor &v,
    const at::Tensor &gamma)
{
    // Construct output tensors for both aclnnV1 and aclnnV2
    const int64_t vDim = v.dim();

    // Parse kv shape (shared sizes) 
    int64_t dim_0 = kv.size(0);     // Bkv
    int64_t dim_1 = kv.size(1);     // N
    int64_t dim_2 = kv.size(2);     // Skv
    int64_t dim_3 = kv.size(3);     // Dk

    int64_t hdim_k = 0;
    int64_t hdim_v = 0;
    if (vDim==4){
        // aclnnv2
        // Dk = D_rope + D_k_flat
        hdim_k = dim_3;
        // Parse v shape
        hdim_v = v.size(3); // Dv

    }else {
        // aclnnV1
        hdim_v = gamma.size(0);         // rms_size(D_nope)
        hdim_k = dim_3 - hdim_v;        // Dkv = D_rope + D_nope
    }

    //int64_t dim_3_rope = dim_3 - dim_3_nope;    // rope_size

    // 原地更新k_cache、ckv_cache，只需创建新的输出
    // k_cache
    at::Tensor k_rope = at::empty({dim_0, dim_1, dim_2, hdim_k}, kv.options().dtype(kv.scalar_type()));
    // 根据 gamma 构造 c_kv
    at::Tensor c_kv = at::empty({dim_0, dim_1, dim_2, hdim_v}, kv.options().dtype(kv.scalar_type()));

    return std::make_tuple(k_rope, c_kv);
}

// 为NPU设备实现前向接口 - v1 (pta的v1)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_ai_infra_kv_rmsnorm_rope_cache_npu(
    const at::Tensor &kv,
    const at::Tensor &gamma,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &index,
    const at::Tensor &k_cache,
    const at::Tensor &ckv_cache,
    const c10::optional<at::Tensor> &k_rope_scale,
    const c10::optional<at::Tensor> &c_kv_scale,
    const c10::optional<at::Tensor> &k_rope_offset,
    const c10::optional<at::Tensor> &c_kv_offset,
    const c10::optional<at::Tensor> &v,
    double epsilon,
    c10::string_view cache_mode,
    c10::string_view rotary_mode,
    c10::string_view quant_mode,
    bool is_output_kv)
{
    // 1. 构建输出张量
    auto outputs = custom::construct_output_tensor_v1(kv, gamma, k_cache, ckv_cache, is_output_kv);
    at::Tensor k_cache_out = std::get<0>(outputs);
    at::Tensor ckv_cache_out = std::get<1>(outputs);
    at::Tensor k_rope_out = std::get<2>(outputs);
    at::Tensor c_kv_out = std::get<3>(outputs);

    // 2. 转换字符串参数
    std::string cache_mode_str = std::string(cache_mode);
    char *cache_mode_ptr = const_cast<char *>(cache_mode_str.c_str());
    std::string rotary_mode_str = std::string(rotary_mode);
    char *rotary_mode_ptr = const_cast<char *>(rotary_mode_str.c_str());
    std::string quant_mode_str = std::string(quant_mode);
    char *quant_mode_ptr = const_cast<char *>(quant_mode_str.c_str());

    // 3. 调用 ACLNN API
    EXEC_NPU_CMD_V1(aclnnAiInfraKvRmsNormRopeCache, 
        kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v,
        epsilon, cache_mode_ptr, rotary_mode_ptr, quant_mode_ptr, is_output_kv,
        k_rope_out, c_kv_out);

    return std::make_tuple(k_cache, ckv_cache, k_rope_out, c_kv_out);
}

// 为Meta设备实现前向接口 - v1
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_ai_infra_kv_rmsnorm_rope_cache_meta(
    const at::Tensor &kv,
    const at::Tensor &gamma,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &index,
    const at::Tensor &k_cache,
    const at::Tensor &ckv_cache,
    const c10::optional<at::Tensor> &k_rope_scale,
    const c10::optional<at::Tensor> &c_kv_scale,
    const c10::optional<at::Tensor> &k_rope_offset,
    const c10::optional<at::Tensor> &c_kv_offset,
    const c10::optional<at::Tensor> &v,
    double epsilon,
    c10::string_view cache_mode,
    c10::string_view rotary_mode,
    c10::string_view quant_mode,
    bool is_output_kv)
{
    auto outputs = custom::construct_output_tensor_v1(kv, gamma, k_cache, ckv_cache, is_output_kv);
    return outputs;
}

// 为NPU设备实现前向接口 - v2
std::tuple<at::Tensor, at::Tensor> npu_ai_infra_kv_rmsnorm_rope_cache_v2_npu(
    const at::Tensor &kv,
    const at::Tensor &gamma,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &index,
    at::Tensor &k_cache,
    at::Tensor &ckv_cache,
    const c10::optional<at::Tensor> &k_rope_scale,
    const c10::optional<at::Tensor> &c_kv_scale,
    const c10::optional<at::Tensor> &k_rope_offset,
    const c10::optional<at::Tensor> &c_kv_offset,
    const c10::optional<at::Tensor> &v,
    double epsilon,
    c10::string_view cache_mode,
    c10::string_view rotary_mode,
    c10::string_view quant_mode,
    bool is_output_kv)
{
    // extract optional tensors
    const at::Tensor &v_tsr_opt = v.value_or(at::Tensor());

    // 1. 构建输出张量
    auto outputs = custom::construct_output_tensor_v2(kv, v_tsr_opt, gamma);
    at::Tensor k_rope_out = std::get<0>(outputs);
    at::Tensor c_kv_out = std::get<1>(outputs);

    // 2. 转换字符串参数
    std::string cache_mode_str = std::string(cache_mode);
    char *cache_mode_ptr = const_cast<char *>(cache_mode_str.c_str());
    std::string rotary_mode_str = std::string(rotary_mode);
    char *rotary_mode_ptr = const_cast<char *>(rotary_mode_str.c_str());
    std::string quant_mode_str = std::string(quant_mode);
    char *quant_mode_ptr = const_cast<char *>(quant_mode_str.c_str());

    // 3. 调用 ACLNN API
    EXEC_NPU_CMD_V1(aclnnAiInfraKvRmsNormRopeCache, 
        kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v,
        epsilon, cache_mode_ptr, rotary_mode_ptr, quant_mode_ptr, is_output_kv,
        k_rope_out, c_kv_out);

    return std::make_tuple(k_rope_out, c_kv_out);
}

// 为Meta设备实现前向接口 - v2
std::tuple<at::Tensor, at::Tensor> npu_ai_infra_kv_rmsnorm_rope_cache_v2_meta(
    const at::Tensor &kv,
    const at::Tensor &gamma,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &index,
    at::Tensor &k_cache,
    at::Tensor &ckv_cache,
    const c10::optional<at::Tensor> &k_rope_scale,
    const c10::optional<at::Tensor> &c_kv_scale,
    const c10::optional<at::Tensor> &k_rope_offset,
    const c10::optional<at::Tensor> &c_kv_offset,
    const c10::optional<at::Tensor> &v,
    double epsilon,
    c10::string_view cache_mode,
    c10::string_view rotary_mode,
    c10::string_view quant_mode,
    bool is_output_kv)
{
    // extract optional tensors
    const at::Tensor &v_tsr_opt = v.value_or(at::Tensor());

    auto outputs = custom::construct_output_tensor_v2(kv, v_tsr_opt, gamma);
    return outputs;
}

} // namespace custom

// NPU设备注册
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_ai_infra_kv_rmsnorm_rope_cache", &custom::npu_ai_infra_kv_rmsnorm_rope_cache_npu);
    m.impl("npu_ai_infra_kv_rmsnorm_rope_cache_v2", &custom::npu_ai_infra_kv_rmsnorm_rope_cache_v2_npu);
}

// Meta设备注册
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_ai_infra_kv_rmsnorm_rope_cache", &custom::npu_ai_infra_kv_rmsnorm_rope_cache_meta);
    m.impl("npu_ai_infra_kv_rmsnorm_rope_cache_v2", &custom::npu_ai_infra_kv_rmsnorm_rope_cache_v2_meta);
}