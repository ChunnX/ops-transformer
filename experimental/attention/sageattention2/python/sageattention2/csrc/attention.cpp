/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include <torch/extension.h>
#include "op_plugin/utils/op_api_common_base.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace op_api {
namespace {
constexpr int64_t RANK = 4;
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr int64_t ND_FORMAT = 2;
constexpr int64_t NCHW_FORMAT = 0;
constexpr int64_t MAX_DIM = (int64_t{1} << 32) - 64;

inline void check_storage_format(const at::Tensor &tensor)
{
    auto format = at_npu::native::custom_ops::get_npu_format(tensor);
    if (C10_LIKELY(format == ND_FORMAT || format == NCHW_FORMAT)) {
        return;
    }
    TORCH_CHECK(false,
                "SageAttention2 requires base ND/NCHW storage on Ascend 950; private formats are unsupported");
}
} // namespace

std::tuple<at::Tensor, c10::optional<at::Tensor>> sageattention2(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const std::string &tensor_layout, bool is_causal,
    const std::string &qk_quant_gran,
    c10::optional<double> sm_scale,
    const std::string &pv_accum_dtype,
    bool smooth_k, bool return_lse)
{
    const char *layout = nullptr;
    bool is_hnd = false;
    if (C10_LIKELY(tensor_layout == "HND")) {
        layout = "HND";
        is_hnd = true;
    } else if (tensor_layout == "NHD") {
        layout = "NHD";
        is_hnd = false;
    } else {
        TORCH_CHECK(false, "tensor_layout must be HND or NHD");
    }

    const char *gran = nullptr;
    if (C10_LIKELY(qk_quant_gran == "per_thread")) {
        gran = "per_thread";
    } else if (qk_quant_gran == "per_warp") {
        gran = "per_warp";
    } else if (qk_quant_gran == "per_token") {
        gran = "per_token";
    } else {
        TORCH_CHECK(false, "qk_quant_gran must be per_thread, per_warp or per_token");
    }

    const char *accum = nullptr;
    if (C10_LIKELY(pv_accum_dtype == "fp32+fp16")) {
        accum = "fp32+fp16";
    } else if (pv_accum_dtype == "fp32+fp32") {
        accum = "fp32+fp32";
    } else if (pv_accum_dtype == "fp32") {
        accum = "fp32";
    } else {
        TORCH_CHECK(false, "pv_accum_dtype must be fp32+fp16, fp32+fp32 or fp32");
    }

    const auto q_sizes = q.sizes();
    const auto k_sizes = k.sizes();
    const auto v_sizes = v.sizes();
    TORCH_CHECK(q_sizes.size() == RANK && k_sizes.size() == RANK && v_sizes.size() == RANK,
                "q/k/v must have rank 4");

    const auto &q_device = q.device();
    TORCH_CHECK(q_device.type() == c10::DeviceType::PrivateUse1 &&
                k.device() == q_device && v.device() == q_device,
                "q/k/v must share an NPU device");

    const auto q_dtype = q.scalar_type();
    TORCH_CHECK((q_dtype == at::kHalf || q_dtype == at::kBFloat16) &&
                k.scalar_type() == q_dtype && v.scalar_type() == q_dtype,
                "q/k/v must share FP16 or BF16 dtype");

    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
                "last dimension must be contiguous");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
                "forward-only operator does not support autograd");

    TORCH_CHECK(k_sizes == v_sizes, "incompatible q/k/v shapes");

    const int64_t head_axis = is_hnd ? 1 : 2;
    const int64_t seq_axis = is_hnd ? 2 : 1;

    const int64_t b = q_sizes[0];
    const int64_t hq = q_sizes[head_axis];
    const int64_t sq = q_sizes[seq_axis];
    const int64_t d = q_sizes[3];

    const int64_t kb = k_sizes[0];
    const int64_t hk = k_sizes[head_axis];
    const int64_t sk = k_sizes[seq_axis];
    const int64_t kd = k_sizes[3];

    TORCH_CHECK(b > 0 && b < MAX_DIM && hq > 0 && hq < MAX_DIM && sq > 0 && sq < MAX_DIM &&
                d > 0 && d < MAX_DIM && hk > 0 && hk < MAX_DIM && sk > 0 && sk < MAX_DIM,
                "invalid tensor dimension");

    TORCH_CHECK(d <= MAX_HEAD_DIM, "Only head_dim <= 128 is supported");
    TORCH_CHECK(kb == b && kd == d, "incompatible q/k/v shapes");
    TORCH_CHECK(hq % hk == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(b * hk < (int64_t{1} << 32), "batch times KV heads exceeds uint32 range");
    TORCH_CHECK(!is_causal || sq == sk, "causal requires Sq == Sk");

    double scale = sm_scale.has_value() ? sm_scale.value() : (1.0 / std::sqrt(static_cast<double>(d)));
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
                "invalid FP32 sm_scale");

    const c10::OptionalDeviceGuard device_guard(q_device);
    check_storage_format(q);
    check_storage_format(k);
    check_storage_format(v);

    auto out = at::empty(q_sizes, q.options());
    c10::optional<at::Tensor> lse;
    if (return_lse) {
        lse = at::empty({b, hq, sq}, q.options().dtype(at::kFloat));
    }
    bool smooth_v = false;
    EXEC_NPU_CMD_EXT(aclnnSageAttention2, q, k, v, layout, is_causal,
                     gran, scale, accum, smooth_k, smooth_v, return_lse, out, lse);
    return {out, lse};
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &op_api::sageattention2, "SageAttention2 INT8 QK / MXFP8 PV forward");
}
