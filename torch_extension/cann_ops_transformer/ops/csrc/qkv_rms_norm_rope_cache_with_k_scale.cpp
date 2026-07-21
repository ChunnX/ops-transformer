/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_with_k_scale.cpp
 * \brief PyTorch extension wrapper for aclnnQkvRmsNormRopeCacheWithKScale.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t HEAD_NUMS_SIZE = 3;
constexpr const char *QKV_LAYOUT_TND = "TND";
constexpr const char *QKV_LAYOUT_NTD = "NTD";
constexpr const char *DEFAULT_QKV_LAYOUT = QKV_LAYOUT_TND;
constexpr const char *DEFAULT_Q_OUT_LAYOUT = QKV_LAYOUT_NTD;

struct QkvKScaleParams {
    int64_t n_q;
    int64_t token_num;
    int64_t head_size;
    bool q_out_is_tnd;
};

std::string ResolveLayout(const c10::optional<c10::string_view> &layout, const char *defaultLayout)
{
    if (!layout.has_value() || layout.value().empty()) {
        return std::string(defaultLayout);
    }
    return std::string(layout.value());
}

QkvKScaleParams ValidateInputs(const at::Tensor &qkv, const at::Tensor &query_start_loc, const at::Tensor &seq_lens,
                               at::IntArrayRef head_nums, const std::string &layout_qkv,
                               const std::string &layout_q_out, const c10::optional<at::Tensor> &rotation,
                               const c10::optional<at::Tensor> &v_scale)
{
    TORCH_CHECK(head_nums.size() == HEAD_NUMS_SIZE,
                "qkv_rms_norm_rope_cache_with_k_scale: head_nums must have exactly 3 elements.");
    const int64_t n_q = head_nums[0];
    const int64_t n_k = head_nums[1];
    const int64_t n_v = head_nums[2];
    TORCH_CHECK(n_q > 0 && n_k > 0 && n_v > 0,
                "qkv_rms_norm_rope_cache_with_k_scale: all head_nums values must be positive.");
    TORCH_CHECK(n_v == n_k, "qkv_rms_norm_rope_cache_with_k_scale: Nv must equal Nk, but got Nv=", n_v, ", Nk=", n_k,
                ".");

    TORCH_CHECK(qkv.dim() == DIM_THREE, "qkv_rms_norm_rope_cache_with_k_scale: qkv must be 3D, but got ", qkv.dim(),
                "D.");
    const bool is_tnd = layout_qkv == QKV_LAYOUT_TND;
    const bool is_ntd = layout_qkv == QKV_LAYOUT_NTD;
    TORCH_CHECK(is_tnd || is_ntd, "qkv_rms_norm_rope_cache_with_k_scale: layout_qkv must be TND or NTD, but got ",
                layout_qkv, ".");
    const bool q_out_is_tnd = layout_q_out == QKV_LAYOUT_TND;
    const bool q_out_is_ntd = layout_q_out == QKV_LAYOUT_NTD;
    TORCH_CHECK(q_out_is_tnd || q_out_is_ntd,
                "qkv_rms_norm_rope_cache_with_k_scale: layout_q_out must be TND or NTD, but got ", layout_q_out, ".");

    const int64_t head_dim = is_tnd ? DIM_1 : DIM_0;
    TORCH_CHECK(qkv.size(head_dim) == n_q + n_k + n_v,
                "qkv_rms_norm_rope_cache_with_k_scale: qkv head dimension must equal Nq+Nk+Nv, but got ",
                qkv.size(head_dim), " vs expected ", n_q + n_k + n_v, ".");
    TORCH_CHECK(query_start_loc.dim() == DIM_1 && query_start_loc.size(DIM_0) >= 2,
                "qkv_rms_norm_rope_cache_with_k_scale: query_start_loc must be 1D with length >= 2.");
    TORCH_CHECK(seq_lens.dim() == DIM_1 && seq_lens.size(DIM_0) == query_start_loc.size(DIM_0) - 1,
                "qkv_rms_norm_rope_cache_with_k_scale: seq_lens must be 1D with length query_start_loc.size(0)-1.");

    TORCH_CHECK(rotation.has_value() && rotation.value().defined(),
                "qkv_rms_norm_rope_cache_with_k_scale: rotation is required.");
    TORCH_CHECK(v_scale.has_value() && v_scale.value().defined(),
                "qkv_rms_norm_rope_cache_with_k_scale: v_scale is required.");

    const int64_t token_num = is_tnd ? qkv.size(DIM_0) : qkv.size(DIM_1);
    const int64_t head_size = qkv.size(DIM_2);
    return {n_q, token_num, head_size, q_out_is_tnd};
}

std::tuple<at::Tensor, at::Tensor> MakeOutputs(const at::Tensor &qkv, const QkvKScaleParams &params)
{
    c10::SmallVector<int64_t, DIM_THREE> q_out_shape =
        params.q_out_is_tnd ? c10::SmallVector<int64_t, DIM_THREE>{params.token_num, params.n_q, params.head_size} :
                              c10::SmallVector<int64_t, DIM_THREE>{params.n_q, params.token_num, params.head_size};
    c10::SmallVector<int64_t, DIM_2> q_scale_shape =
        params.q_out_is_tnd ? c10::SmallVector<int64_t, DIM_2>{params.token_num, params.n_q} :
                              c10::SmallVector<int64_t, DIM_2>{params.n_q, params.token_num};

    at::Tensor q_out;
    at::Tensor q_scale;
    // q_out dtype is fixed by the ACLNN contract: Q is dynamically quantized to FP8 E4M3FN.
    q_out = at::empty(q_out_shape, qkv.options().dtype(at::kFloat8_e4m3fn));
    q_scale = at::empty(q_scale_shape, qkv.options().dtype(at::kFloat));
    return {q_out, q_scale};
}

void RunQkvKScaleAclnn(const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma,
                       const at::Tensor &cos_sin, const at::Tensor &slot_mapping, const at::Tensor &k_cache,
                       const at::Tensor &v_cache, const at::Tensor &k_scale_cache, const at::Tensor &query_start_loc,
                       const at::Tensor &seq_lens, at::IntArrayRef head_nums, const std::string &layout_qkv,
                       const std::string &layout_q_out, const at::Tensor &rotation, const at::Tensor &v_scale,
                       double epsilon, at::Tensor &q_out, at::Tensor &q_scale)
{
    const char *layout_qkv_ptr = layout_qkv.c_str();
    const char *layout_q_out_ptr = layout_q_out.c_str();
    float epsilon_value = static_cast<float>(epsilon);

    ACLNN_CMD(aclnnQkvRmsNormRopeCacheWithKScale, qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
              k_scale_cache, query_start_loc, seq_lens, rotation, v_scale, head_nums, layout_qkv_ptr, layout_q_out_ptr,
              epsilon_value, q_out, q_scale);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> qkv_rms_norm_rope_cache_with_k_scale_(
    const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos_sin,
    const at::Tensor &slot_mapping, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &k_scale_cache, const at::Tensor &query_start_loc, const at::Tensor &seq_lens,
    at::IntArrayRef head_nums, const c10::optional<c10::string_view> &layout_qkv,
    const c10::optional<c10::string_view> &layout_q_out, const c10::optional<at::Tensor> &rotation,
    const c10::optional<at::Tensor> &v_scale, double epsilon)
{
    const std::string layout_qkv_str = ResolveLayout(layout_qkv, DEFAULT_QKV_LAYOUT);
    const std::string layout_q_out_str = ResolveLayout(layout_q_out, DEFAULT_Q_OUT_LAYOUT);
    const QkvKScaleParams params =
        ValidateInputs(qkv, query_start_loc, seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation, v_scale);
    auto outputs = MakeOutputs(qkv, params);
    at::Tensor q_out = std::get<0>(outputs);
    at::Tensor q_scale = std::get<1>(outputs);

    RunQkvKScaleAclnn(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache, k_scale_cache, query_start_loc,
                      seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation.value(), v_scale.value(), epsilon,
                      q_out, q_scale);
    return {q_out, q_scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> qkv_rms_norm_rope_cache_with_k_scale(
    const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos_sin,
    const at::Tensor &slot_mapping, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &k_scale_cache, const at::Tensor &query_start_loc, const at::Tensor &seq_lens,
    at::IntArrayRef head_nums, const c10::optional<c10::string_view> &layout_qkv,
    const c10::optional<c10::string_view> &layout_q_out, const c10::optional<at::Tensor> &rotation,
    const c10::optional<at::Tensor> &v_scale, double epsilon)
{
    const std::string layout_qkv_str = ResolveLayout(layout_qkv, DEFAULT_QKV_LAYOUT);
    const std::string layout_q_out_str = ResolveLayout(layout_q_out, DEFAULT_Q_OUT_LAYOUT);
    const QkvKScaleParams params =
        ValidateInputs(qkv, query_start_loc, seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation, v_scale);

    at::Tensor k_cache_clone = k_cache.clone();
    at::Tensor v_cache_clone = v_cache.clone();
    at::Tensor k_scale_cache_clone = k_scale_cache.clone();
    auto outputs = MakeOutputs(qkv, params);
    at::Tensor q_out = std::get<0>(outputs);
    at::Tensor q_scale = std::get<1>(outputs);

    RunQkvKScaleAclnn(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache_clone, v_cache_clone, k_scale_cache_clone,
                      query_start_loc, seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation.value(),
                      v_scale.value(), epsilon, q_out, q_scale);
    return {q_out, q_scale, k_cache_clone, v_cache_clone, k_scale_cache_clone};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("qkv_rms_norm_rope_cache_with_k_scale_", &qkv_rms_norm_rope_cache_with_k_scale_,
          "qkv_rms_norm_rope_cache_with_k_scale_");
    m.def("qkv_rms_norm_rope_cache_with_k_scale", &qkv_rms_norm_rope_cache_with_k_scale,
          "qkv_rms_norm_rope_cache_with_k_scale");
}
} // namespace op_api
