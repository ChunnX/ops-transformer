/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <cmath>
#include "aclnn_common.h"

namespace op_api {
/** Allocate the four public outputs on h_out.device and dispatch the ACLNN forward operation. */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mega_mhc(
    const at::Tensor &h_out, const at::Tensor &residual, const at::Tensor &h_post,
    const at::Tensor &h_res, const at::Tensor &phi, const at::Tensor &alpha,
    const at::Tensor &bias, const at::Tensor &gamma_1, const at::Tensor &gamma_0,
    const c10::optional<at::Tensor> &gamma_2, double norm_eps, double hc_eps, int64_t compute_mode)
{
    constexpr int64_t HEADS = 4;
    constexpr int64_t PROJECTION = 24;
    TORCH_CHECK(h_out.device().type() == c10::DeviceType::PrivateUse1, "h_out must be an NPU tensor");
    TORCH_CHECK(h_out.dim() == 2 || h_out.dim() == 3, "h_out must be [T,D] or [B,S,D]");
    TORCH_CHECK(h_out.scalar_type() == at::kHalf || h_out.scalar_type() == at::kBFloat16,
                "h_out must have FP16/BF16 dtype");
    const int64_t d = h_out.size(-1);
    TORCH_CHECK(d == 2560 || d == 5120, "D must be 2560 or 5120; got ", d);
    TORCH_CHECK(compute_mode == 0 || compute_mode == 1, "compute_mode must be FP32(0) or HF32(1)");
    TORCH_CHECK(std::isfinite(static_cast<float>(norm_eps)) && static_cast<float>(norm_eps) > 0,
                "norm_eps must be positive and finite in FP32");
    TORCH_CHECK(std::isfinite(static_cast<float>(hc_eps)) && hc_eps >= 0, "hc_eps must be nonnegative and finite");
    auto leading = h_out.sizes().vec();
    leading.pop_back();
    uint64_t rows = 1;
    for (const auto size : leading) {
        TORCH_CHECK(size > 0 && rows <= UINT32_MAX / static_cast<uint64_t>(size),
                    "flattened token count must be positive and fit uint32");
        rows *= size;
    }
    auto post_shape = leading;
    post_shape.push_back(HEADS);
    auto res_shape = post_shape;
    res_shape.push_back(HEADS);
    auto x2_shape = post_shape;
    x2_shape.push_back(d);
    // ACLNN repeats the contract for callers bypassing this Torch wrapper.
    const auto check = [&](const at::Tensor &value, const char *name,
                           const std::vector<int64_t> &shape, at::ScalarType dtype) {
        TORCH_CHECK(value.defined(), name, " is required");
        TORCH_CHECK(value.device() == h_out.device(), name, " must be on h_out.device");
        TORCH_CHECK(value.scalar_type() == dtype, name, " has incorrect dtype");
        TORCH_CHECK(value.sizes().vec() == shape, name, " has incorrect shape: ", value.sizes());
    };
    check(residual, "residual", x2_shape, h_out.scalar_type());
    check(h_post, "h_post", post_shape, at::kFloat);
    check(h_res, "h_res", res_shape, at::kFloat);
    check(phi, "phi", {PROJECTION, HEADS * d}, at::kFloat);
    check(alpha, "alpha", {3}, at::kFloat);
    check(bias, "bias", {PROJECTION}, at::kFloat);
    check(gamma_1, "gamma_1", {d}, at::kFloat);
    check(gamma_0, "gamma_0", {d}, at::kFloat);
    if (gamma_2.has_value()) {
        check(*gamma_2, "gamma_2", {HEADS * d}, at::kFloat);
    }
    at::Tensor h_in_prime, h_post_new, h_res_new, x_2_out;
    {
        const c10::OptionalDeviceGuard guard(h_out.device());
        h_in_prime = at::empty(h_out.sizes(), h_out.options());
        h_post_new = at::empty(post_shape, h_out.options().dtype(at::kFloat));
        h_res_new = at::empty(res_shape, h_out.options().dtype(at::kFloat));
        x_2_out = at::empty(x2_shape, h_out.options());
        ACLNN_CMD(aclnnAiInfraMegaMhc, h_out, residual, h_post, h_res, phi, alpha, bias,
                  gamma_1, gamma_0, gamma_2, norm_eps, hc_eps, compute_mode,
                  h_in_prime, h_post_new, h_res_new, x_2_out);
    }
    return {h_in_prime, h_post_new, h_res_new, x_2_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mega_mhc", &mega_mhc, "Ascend 950 mega_mhc forward");
}
} // namespace op_api
