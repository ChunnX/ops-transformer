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
constexpr const char *Q_QUANT_PER_TOKEN_PER_HEAD = "PerTokenPerHead";
constexpr const char *Q_QUANT_NO_QUANT = "NoQuant";
constexpr int64_t GE_DTYPE_BF16 = 27;
constexpr int64_t GE_DTYPE_FLOAT8_E4M3FN = 36;
constexpr int64_t MROPE_HEAD_SIZE = 128;

struct QkvKScaleParams {
    int64_t n_q;
    int64_t token_num;
    int64_t head_size;
    bool q_out_is_tnd;
    bool is_mrope;
    bool q_scale_required;
};

std::string ResolveLayout(const c10::optional<c10::string_view> &layout, const char *defaultLayout)
{
    if (!layout.has_value() || layout.value().empty()) {
        return std::string(defaultLayout);
    }
    return std::string(layout.value());
}

std::string ResolveQQuantMode(const std::string &q_quant_mode)
{
    if (q_quant_mode.empty()) {
        return std::string(Q_QUANT_PER_TOKEN_PER_HEAD);
    }
    TORCH_CHECK(q_quant_mode == Q_QUANT_PER_TOKEN_PER_HEAD || q_quant_mode == Q_QUANT_NO_QUANT,
                "qkv_rms_norm_rope_cache_with_k_scale: q_quant_mode must be PerTokenPerHead or NoQuant, but got ",
                q_quant_mode, ".");
    return q_quant_mode;
}

void CheckOptionalTensorDefined(const c10::optional<at::Tensor> &tensor, const char *name)
{
    if (tensor.has_value()) {
        TORCH_CHECK(tensor.value().defined(), "qkv_rms_norm_rope_cache_with_k_scale: ", name,
                    " is present but undefined.");
    }
}

QkvKScaleParams ValidateInputs(const at::Tensor &qkv, const c10::optional<at::Tensor> &query_start_loc,
                               const c10::optional<at::Tensor> &seq_lens, at::IntArrayRef head_nums,
                               const std::string &layout_qkv, const std::string &layout_q_out,
                               const c10::optional<at::Tensor> &rotation, const c10::optional<at::Tensor> &v_scale,
                               const c10::optional<at::Tensor> &mrope_position,
                               const c10::optional<std::vector<int64_t>> &mrope_section,
                               const std::string &q_quant_mode, int64_t q_out_dtype, const at::Tensor &k_cache,
                               const at::Tensor &v_cache, const at::Tensor &k_scale_cache)
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
    const int64_t token_num = is_tnd ? qkv.size(DIM_0) : qkv.size(DIM_1);
    const int64_t head_size = qkv.size(DIM_2);
    CheckOptionalTensorDefined(query_start_loc, "query_start_loc");
    CheckOptionalTensorDefined(seq_lens, "seq_lens");
    CheckOptionalTensorDefined(rotation, "rotation");
    CheckOptionalTensorDefined(v_scale, "v_scale");
    CheckOptionalTensorDefined(mrope_position, "mrope_position");

    const bool query_source_complete = query_start_loc.has_value() && seq_lens.has_value();
    const bool mrope_source_complete = mrope_position.has_value() && mrope_section.has_value();
    TORCH_CHECK(query_start_loc.has_value() == seq_lens.has_value(),
                "qkv_rms_norm_rope_cache_with_k_scale: query_start_loc and seq_lens must be provided together.");
    TORCH_CHECK(mrope_position.has_value() == mrope_section.has_value(),
                "qkv_rms_norm_rope_cache_with_k_scale: mrope_position and mrope_section must be provided together.");
    TORCH_CHECK(query_source_complete != mrope_source_complete,
                "qkv_rms_norm_rope_cache_with_k_scale: exactly one position source must be provided.");

    const bool is_mrope = mrope_source_complete;
    const bool q_scale_required = !is_mrope;
    TORCH_CHECK((is_mrope && q_quant_mode == Q_QUANT_NO_QUANT && q_out_dtype == GE_DTYPE_BF16) ||
                    (!is_mrope && q_quant_mode == Q_QUANT_PER_TOKEN_PER_HEAD &&
                     q_out_dtype == GE_DTYPE_FLOAT8_E4M3FN),
                "qkv_rms_norm_rope_cache_with_k_scale: q_quant_mode and q_out_dtype do not match the selected ",
                is_mrope ? "M-RoPE" : "legacy", " scene.");

    if (is_mrope) {
        TORCH_CHECK(layout_qkv == QKV_LAYOUT_TND && layout_q_out == QKV_LAYOUT_TND,
                    "qkv_rms_norm_rope_cache_with_k_scale: M-RoPE mode only supports TND to TND.");
        TORCH_CHECK(head_size == MROPE_HEAD_SIZE,
                    "qkv_rms_norm_rope_cache_with_k_scale: M-RoPE mode requires head dimension 128, but got ",
                    head_size, ".");
        const at::Tensor &position = mrope_position.value();
        TORCH_CHECK(position.scalar_type() == at::kInt && position.dim() == DIM_2 &&
                        position.size(DIM_0) == DIM_THREE && position.size(DIM_1) == token_num,
                    "qkv_rms_norm_rope_cache_with_k_scale: mrope_position must be int32 with shape [3, T].");
        const auto &section = mrope_section.value();
        TORCH_CHECK(section.size() == HEAD_NUMS_SIZE,
                    "qkv_rms_norm_rope_cache_with_k_scale: mrope_section must have exactly 3 elements.");
        const int64_t half_dim = head_size / 2;
        const int64_t section_caps[HEAD_NUMS_SIZE] = {(half_dim + 2) / 3, (half_dim + 1) / 3, half_dim / 3};
        for (size_t index = 0; index < section.size(); ++index) {
            TORCH_CHECK(section[index] >= 0 && section[index] <= section_caps[index],
                        "qkv_rms_norm_rope_cache_with_k_scale: mrope_section[", index,
                        "] must be in [0, ", section_caps[index], "].");
        }
        TORCH_CHECK(k_cache.scalar_type() == at::kChar,
                    "qkv_rms_norm_rope_cache_with_k_scale: M-RoPE k_cache must be int8.");
    } else {
        TORCH_CHECK(!(layout_qkv == QKV_LAYOUT_NTD && layout_q_out == QKV_LAYOUT_TND),
                    "qkv_rms_norm_rope_cache_with_k_scale: legacy NTD input does not support TND output.");
        const at::Tensor &query_loc = query_start_loc.value();
        const at::Tensor &lengths = seq_lens.value();
        TORCH_CHECK(query_loc.scalar_type() == at::kInt && lengths.scalar_type() == at::kInt &&
                        query_loc.dim() == DIM_1 && query_loc.size(DIM_0) >= 2 && lengths.dim() == DIM_1 &&
                        lengths.size(DIM_0) == query_loc.size(DIM_0) - 1,
                    "qkv_rms_norm_rope_cache_with_k_scale: legacy position tensors must be int32 with matching 1D "
                    "shapes.");
        TORCH_CHECK(k_cache.scalar_type() == at::kFloat8_e4m3fn,
                    "qkv_rms_norm_rope_cache_with_k_scale: legacy k_cache must be FP8 E4M3FN.");
    }

    TORCH_CHECK(v_cache.scalar_type() == at::kFloat8_e4m3fn,
                "qkv_rms_norm_rope_cache_with_k_scale: v_cache must be FP8 E4M3FN.");
    TORCH_CHECK(k_scale_cache.scalar_type() == at::kFloat,
                "qkv_rms_norm_rope_cache_with_k_scale: k_scale_cache must be FP32.");
    TORCH_CHECK(rotation.has_value() && rotation.value().defined(),
                "qkv_rms_norm_rope_cache_with_k_scale: rotation is required.");
    TORCH_CHECK(v_scale.has_value() && v_scale.value().defined(),
                "qkv_rms_norm_rope_cache_with_k_scale: v_scale is required.");

    return {n_q, token_num, head_size, q_out_is_tnd, is_mrope, q_scale_required};
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> MakeOutputs(const at::Tensor &qkv, const QkvKScaleParams &params)
{
    c10::SmallVector<int64_t, DIM_THREE> q_out_shape =
        params.q_out_is_tnd ? c10::SmallVector<int64_t, DIM_THREE>{params.token_num, params.n_q, params.head_size} :
                              c10::SmallVector<int64_t, DIM_THREE>{params.n_q, params.token_num, params.head_size};
    c10::SmallVector<int64_t, DIM_2> q_scale_shape =
        params.q_out_is_tnd ? c10::SmallVector<int64_t, DIM_2>{params.token_num, params.n_q} :
                              c10::SmallVector<int64_t, DIM_2>{params.n_q, params.token_num};

    at::Tensor q_out;
    c10::optional<at::Tensor> q_scale = c10::nullopt;
    q_out = at::empty(q_out_shape, qkv.options().dtype(params.is_mrope ? at::kBFloat16 : at::kFloat8_e4m3fn));
    if (params.q_scale_required) {
        q_scale = at::empty(q_scale_shape, qkv.options().dtype(at::kFloat));
    }
    return {q_out, q_scale};
}

void RunQkvKScaleAclnn(const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma,
                       const at::Tensor &cos_sin, const at::Tensor &slot_mapping, const at::Tensor &k_cache,
                       const at::Tensor &v_cache, const at::Tensor &k_scale_cache,
                       const c10::optional<at::Tensor> &query_start_loc, const c10::optional<at::Tensor> &seq_lens,
                       at::IntArrayRef head_nums, const std::string &layout_qkv, const std::string &layout_q_out,
                       const c10::optional<at::Tensor> &rotation, const c10::optional<at::Tensor> &v_scale,
                       const c10::optional<at::Tensor> &mrope_position,
                       const c10::optional<std::vector<int64_t>> &mrope_section, const std::string &q_quant_mode,
                       int64_t q_out_dtype, double epsilon, at::Tensor &q_out,
                       const c10::optional<at::Tensor> &q_scale)
{
    const char *layout_qkv_ptr = layout_qkv.c_str();
    const char *layout_q_out_ptr = layout_q_out.c_str();
    const char *q_quant_mode_ptr = q_quant_mode.c_str();
    float epsilon_value = static_cast<float>(epsilon);
    c10::optional<at::IntArrayRef> mrope_section_ref = c10::nullopt;
    if (mrope_section.has_value()) {
        mrope_section_ref = at::IntArrayRef(mrope_section.value());
    }

    ACLNN_CMD(aclnnQkvRmsNormRopeCacheWithKScale, qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache,
              k_scale_cache, query_start_loc, seq_lens, rotation, v_scale, mrope_position, head_nums, layout_qkv_ptr,
              layout_q_out_ptr, epsilon_value, mrope_section_ref, q_quant_mode_ptr, q_out_dtype, q_out, q_scale);
}
} // namespace

std::tuple<at::Tensor, c10::optional<at::Tensor>> qkv_rms_norm_rope_cache_with_k_scale_(
    const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos_sin,
    const at::Tensor &slot_mapping, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &k_scale_cache, const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &seq_lens,
    at::IntArrayRef head_nums, const c10::optional<c10::string_view> &layout_qkv,
    const c10::optional<c10::string_view> &layout_q_out, const c10::optional<at::Tensor> &rotation,
    const c10::optional<at::Tensor> &v_scale, double epsilon,
    const c10::optional<at::Tensor> &mrope_position,
    const c10::optional<std::vector<int64_t>> &mrope_section, const std::string &q_quant_mode,
    int64_t q_out_dtype)
{
    const c10::OptionalDeviceGuard device_guard(qkv.device());
    const std::string layout_qkv_str = ResolveLayout(layout_qkv, DEFAULT_QKV_LAYOUT);
    const std::string layout_q_out_str = ResolveLayout(layout_q_out, DEFAULT_Q_OUT_LAYOUT);
    const std::string q_quant_mode_str = ResolveQQuantMode(q_quant_mode);
    const QkvKScaleParams params = ValidateInputs(qkv, query_start_loc, seq_lens, head_nums, layout_qkv_str,
                                                  layout_q_out_str, rotation, v_scale, mrope_position, mrope_section,
                                                  q_quant_mode_str, q_out_dtype, k_cache, v_cache, k_scale_cache);
    auto outputs = MakeOutputs(qkv, params);
    at::Tensor q_out = std::get<0>(outputs);
    c10::optional<at::Tensor> q_scale = std::get<1>(outputs);

    RunQkvKScaleAclnn(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache, v_cache, k_scale_cache, query_start_loc,
                      seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation, v_scale, mrope_position,
                      mrope_section, q_quant_mode_str, q_out_dtype, epsilon, q_out, q_scale);
    return {q_out, q_scale};
}

std::tuple<at::Tensor, c10::optional<at::Tensor>, at::Tensor, at::Tensor, at::Tensor>
qkv_rms_norm_rope_cache_with_k_scale(
    const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos_sin,
    const at::Tensor &slot_mapping, const at::Tensor &k_cache, const at::Tensor &v_cache,
    const at::Tensor &k_scale_cache, const c10::optional<at::Tensor> &query_start_loc,
    const c10::optional<at::Tensor> &seq_lens,
    at::IntArrayRef head_nums, const c10::optional<c10::string_view> &layout_qkv,
    const c10::optional<c10::string_view> &layout_q_out, const c10::optional<at::Tensor> &rotation,
    const c10::optional<at::Tensor> &v_scale, double epsilon,
    const c10::optional<at::Tensor> &mrope_position,
    const c10::optional<std::vector<int64_t>> &mrope_section, const std::string &q_quant_mode,
    int64_t q_out_dtype)
{
    const c10::OptionalDeviceGuard device_guard(qkv.device());
    const std::string layout_qkv_str = ResolveLayout(layout_qkv, DEFAULT_QKV_LAYOUT);
    const std::string layout_q_out_str = ResolveLayout(layout_q_out, DEFAULT_Q_OUT_LAYOUT);
    const std::string q_quant_mode_str = ResolveQQuantMode(q_quant_mode);
    const QkvKScaleParams params = ValidateInputs(qkv, query_start_loc, seq_lens, head_nums, layout_qkv_str,
                                                  layout_q_out_str, rotation, v_scale, mrope_position, mrope_section,
                                                  q_quant_mode_str, q_out_dtype, k_cache, v_cache, k_scale_cache);

    at::Tensor k_cache_clone = k_cache.clone();
    at::Tensor v_cache_clone = v_cache.clone();
    at::Tensor k_scale_cache_clone = k_scale_cache.clone();
    auto outputs = MakeOutputs(qkv, params);
    at::Tensor q_out = std::get<0>(outputs);
    c10::optional<at::Tensor> q_scale = std::get<1>(outputs);

    RunQkvKScaleAclnn(qkv, q_gamma, k_gamma, cos_sin, slot_mapping, k_cache_clone, v_cache_clone, k_scale_cache_clone,
                      query_start_loc, seq_lens, head_nums, layout_qkv_str, layout_q_out_str, rotation, v_scale,
                      mrope_position, mrope_section, q_quant_mode_str, q_out_dtype, epsilon, q_out, q_scale);
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
