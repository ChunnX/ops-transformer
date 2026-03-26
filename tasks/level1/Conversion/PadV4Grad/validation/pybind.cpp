#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <vector>

#include "pytorch_npu_helper.hpp"

static std::vector<int64_t> reflect_pad_grad_in_sizes(
    const at::Tensor &grad_out, int64_t pad_top, int64_t pad_bottom, int64_t pad_left, int64_t pad_right)
{
    TORCH_CHECK(grad_out.dim() == 4, "PadV4Grad expects 4D grad_output (NCHW)");
    std::vector<int64_t> in_sizes(grad_out.sizes().begin(), grad_out.sizes().end());
    TORCH_CHECK(
        in_sizes[2] >= pad_top + pad_bottom && in_sizes[3] >= pad_left + pad_right,
        "grad_out H/W too small for padding");
    in_sizes[2] -= pad_top + pad_bottom;
    in_sizes[3] -= pad_left + pad_right;
    return in_sizes;
}

static void parse_ge_padding_eight(const at::Tensor &padding, int64_t &pt, int64_t &pb, int64_t &pl, int64_t &pr)
{
    at::Tensor flat = padding.reshape(-1).to(at::kCPU, at::kLong).contiguous();
    TORCH_CHECK(flat.dim() == 1, "padding must be 1-D (after reshape)");
    TORCH_CHECK(flat.numel() == 8, "padding must have 8 elements for 4D PadV4Grad");
    auto acc = flat.accessor<int64_t, 1>();
    pt = acc[4];
    pb = acc[5];
    pl = acc[6];
    pr = acc[7];
}

at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &padding)
{
    TORCH_CHECK(x.defined(), "grad_output x is not defined");
    TORCH_CHECK(padding.defined(), "padding is not defined");

    int64_t pad_top = 0;
    int64_t pad_bottom = 0;
    int64_t pad_left = 0;
    int64_t pad_right = 0;
    parse_ge_padding_eight(padding, pad_top, pad_bottom, pad_left, pad_right);

    std::vector<int64_t> pad8(8);
    {
        at::Tensor flat = padding.reshape(-1).to(at::kCPU, at::kLong).contiguous();
        auto acc = flat.accessor<int64_t, 1>();
        for (int64_t i = 0; i < 8; ++i) {
            pad8[static_cast<size_t>(i)] = acc[i];
        }
    }
    at::Tensor paddings_tensor =
        at::tensor(pad8, at::TensorOptions().dtype(at::kLong).device(at::kCPU)).contiguous();

    std::vector<int64_t> in_sizes =
        reflect_pad_grad_in_sizes(x, pad_top, pad_bottom, pad_left, pad_right);
    at::Tensor result = at::empty(in_sizes, x.options());
    TORCH_CHECK(result.defined(), "result must be defined");

    const char *mode = "reflect";
    bool paddings_contiguous = true;
    EXEC_NPU_CMD(aclnnCustomOp, x, paddings_tensor, mode, paddings_contiguous, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
