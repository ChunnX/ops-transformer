#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

static std::vector<int64_t> output_sizes_from_perm(const at::Tensor &x, const at::Tensor &perm_cpu)
{
    at::Tensor flat = perm_cpu.reshape(-1).to(at::kCPU, at::kLong).contiguous();
    TORCH_CHECK(flat.dim() == 1, "perm must be 1-D");
    const int64_t ndim = x.dim();
    TORCH_CHECK(flat.numel() == static_cast<int64_t>(ndim), "perm length must match x.dim()");
    auto acc = flat.accessor<int64_t, 1>();
    std::vector<int64_t> out_sizes(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t p = acc[i];
        TORCH_CHECK(p >= 0 && p < ndim, "perm index out of range");
        out_sizes[static_cast<size_t>(i)] = x.size(p);
    }
    return out_sizes;
}

at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &perm)
{
    TORCH_CHECK(x.defined() && perm.defined(), "x and perm must be defined");
    at::Tensor perm_cpu = perm.reshape(-1).to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> out_sizes = output_sizes_from_perm(x, perm_cpu);
    at::Tensor y = at::empty(out_sizes, x.options());
    EXEC_NPU_CMD(aclnnCustomOp, x, perm_cpu, y);
    return y;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
