#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
using namespace at_npu::native;

// 为NPU设备实现反向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_ai_infra_mhc_post_grad_npu(
    const at::Tensor & grad_output,
    const at::Tensor & x,
    const at::Tensor & h_res,
    const at::Tensor & h_out,
    const at::Tensor & h_post)
{
    // grad_output shape: (B, S, n, D) or (T, n, D)
    // x shape: (B, S, n, D) or (T, n, D)
    // h_res shape: (B, S, n, n) or (T, n, n)
    // h_out shape: (B, S, D) or (T, D)
    // h_post shape: (B, S, n) or (T, n)

    // 输出shapes
    at::Tensor grad_x = at::empty_like(x);                    // (B, S, n, D) or (T, n, D)
    at::Tensor grad_h_res = at::empty_like(h_res);            // (B, S, n, n) or (T, n, n)
    at::Tensor grad_h_out = at::empty_like(h_out);            // (B, S, D) or (T, D)
    at::Tensor grad_h_post = at::empty_like(h_post);          // (B, S, n) or (T, n)

    EXEC_NPU_CMD_V1(aclnnAiInfraMhcPostGrad,
                    grad_output, x, h_res, h_out, h_post,
                    grad_x, grad_h_res, grad_h_out, grad_h_post);

    return std::make_tuple(grad_x, grad_h_res, grad_h_out, grad_h_post);
}

// 为META设备实现反向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_ai_infra_mhc_post_grad_meta(
    const at::Tensor & grad_output,
    const at::Tensor & x,
    const at::Tensor & h_res,
    const at::Tensor & h_out,
    const at::Tensor & h_post)
{
    at::Tensor grad_x = at::empty_like(x);
    at::Tensor grad_h_res = at::empty_like(h_res);
    at::Tensor grad_h_out = at::empty_like(h_out);
    at::Tensor grad_h_post = at::empty_like(h_post);

    return std::make_tuple(grad_x, grad_h_res, grad_h_out, grad_h_post);
}

// 为NPU设备注册反向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_ai_infra_mhc_post_grad", &custom::npu_ai_infra_mhc_post_grad_npu);
}

// 为META设备注册反向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_ai_infra_mhc_post_grad", &custom::npu_ai_infra_mhc_post_grad_meta);
}

}
