#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <vector>

#include "pytorch_npu_helper.hpp"

// grad_output（正向输出尺寸）-> grad_input（与正向 self 同形）
static std::vector<int64_t> circular_pad_grad_in_sizes(const at::Tensor &grad_out, const std::vector<int64_t> &padding)
{
    TORCH_CHECK(padding.size() == 4 || padding.size() == 6, "padding length must be 4 (2d) or 6 (3d)");
    std::vector<int64_t> in_sizes(grad_out.sizes().begin(), grad_out.sizes().end());
    const int64_t nd = static_cast<int64_t>(in_sizes.size());
    if (padding.size() == 4) {
        TORCH_CHECK(nd >= 3, "circular_pad_grad 2d expects grad_out dim >= 3");
        TORCH_CHECK(
            in_sizes[nd - 2] >= padding[2] + padding[3] && in_sizes[nd - 1] >= padding[0] + padding[1],
            "grad_out spatial dims too small for padding");
        in_sizes[nd - 2] -= padding[2] + padding[3];
        in_sizes[nd - 1] -= padding[0] + padding[1];
    } else {
        TORCH_CHECK(nd >= 4, "circular_pad_grad 3d expects grad_out dim >= 4");
        TORCH_CHECK(in_sizes[nd - 3] >= padding[4] + padding[5] && in_sizes[nd - 2] >= padding[2] + padding[3] &&
                         in_sizes[nd - 1] >= padding[0] + padding[1],
                     "grad_out spatial dims too small for padding");
        in_sizes[nd - 3] -= padding[4] + padding[5];
        in_sizes[nd - 2] -= padding[2] + padding[3];
        in_sizes[nd - 1] -= padding[0] + padding[1];
    }
    return in_sizes;
}

static std::vector<int64_t> padding_tensor_to_int64_vec(const at::Tensor &padding)
{
    at::Tensor flat = padding.reshape(-1).to(at::kCPU, at::kLong).contiguous();
    TORCH_CHECK(flat.dim() == 1, "padding must be 1-D (after reshape)");
    TORCH_CHECK(flat.numel() == 4 || flat.numel() == 6, "padding must have 4 or 6 elements");
    auto acc = flat.accessor<int64_t, 1>();
    std::vector<int64_t> out(static_cast<size_t>(flat.numel()));
    for (int64_t i = 0; i < flat.numel(); ++i) {
        out[static_cast<size_t>(i)] = acc[i];
    }
    return out;
}

at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &padding)
{
    TORCH_CHECK(x.defined(), "grad_output x is not defined");
    TORCH_CHECK(padding.defined(), "padding is not defined");

    std::vector<int64_t> padding_vec = padding_tensor_to_int64_vec(padding);
    at::Tensor paddings_tensor =
        at::tensor(padding_vec, at::TensorOptions().dtype(at::kLong).device(at::kCPU)).contiguous();

    std::vector<int64_t> in_sizes = circular_pad_grad_in_sizes(x, padding_vec);
    at::Tensor result = at::empty(in_sizes, x.options());

    TORCH_CHECK(result.defined(), "result must be defined");

    EXEC_NPU_CMD(aclnnCustomOp, x, paddings_tensor, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
