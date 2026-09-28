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
    TORCH_CHECK(format == ND_FORMAT || format == NCHW_FORMAT,
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
    TORCH_CHECK(tensor_layout == "HND" || tensor_layout == "NHD", "tensor_layout must be HND or NHD");
    TORCH_CHECK(qk_quant_gran == "per_thread" || qk_quant_gran == "per_warp" || qk_quant_gran == "per_token",
                "qk_quant_gran must be per_thread, per_warp or per_token");
    TORCH_CHECK(pv_accum_dtype == "fp32+fp16" || pv_accum_dtype == "fp32+fp32" || pv_accum_dtype == "fp32",
                "pv_accum_dtype must be fp32+fp16, fp32+fp32 or fp32");

    TORCH_CHECK(q.dim() == RANK && k.dim() == RANK && v.dim() == RANK, "q/k/v must have rank 4");
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1 &&
                k.device() == q.device() && v.device() == q.device(),
                "q/k/v must share an NPU device");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
                (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16),
                "q/k/v must share FP16 or BF16 dtype");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
                "last dimension must be contiguous");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
                "forward-only operator does not support autograd");

    for (int64_t i = 0; i < RANK; ++i) {
        auto qs = q.size(i);
        auto ks = k.size(i);
        auto vs = v.size(i);
        TORCH_CHECK(qs > 0 && qs < MAX_DIM && ks > 0 && ks < MAX_DIM && vs > 0 && vs < MAX_DIM,
                    "invalid tensor dimension");
    }

    const bool is_hnd = (tensor_layout == "HND");
    const int64_t head_axis = is_hnd ? 1 : 2;
    const int64_t seq_axis = is_hnd ? 2 : 1;
    TORCH_CHECK(q.size(-1) <= MAX_HEAD_DIM, "head_dim must be in [1,128]");
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == q.size(0) && k.size(-1) == q.size(-1),
                "incompatible q/k/v shapes");
    TORCH_CHECK(q.size(head_axis) % k.size(head_axis) == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(q.size(0) * k.size(head_axis) < (int64_t{1} << 32), "batch times KV heads exceeds uint32 range");
    TORCH_CHECK(!is_causal || q.size(seq_axis) == k.size(seq_axis), "causal requires Sq == Sk");

    double scale = sm_scale.has_value() ? sm_scale.value() : (1.0 / std::sqrt(static_cast<double>(q.size(-1))));
    TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
                "invalid FP32 sm_scale");

    const c10::OptionalDeviceGuard device_guard(q.device());
    check_storage_format(q);
    check_storage_format(k);
    check_storage_format(v);

    auto out = at::empty(q.sizes(), q.options());
    c10::optional<at::Tensor> lse;
    if (return_lse) {
        lse = at::empty({q.size(0), q.size(head_axis), q.size(seq_axis)}, q.options().dtype(at::kFloat));
    }
    const char *layout = is_hnd ? "HND" : "NHD";
    const char *gran = qk_quant_gran.c_str();
    const char *accum = pv_accum_dtype.c_str();
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
