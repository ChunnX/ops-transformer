#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(
    const at::Tensor &grad_out,
    std::vector<int64_t> input_sizes,
    int64_t dim,
    int64_t size,
    int64_t step)
{
    aclIntArray *sizes_acl = ConvertType(at::IntArrayRef(input_sizes));
    at::Tensor grad_in = at::empty(input_sizes, grad_out.options());
    EXEC_NPU_CMD(aclnnCustomOp, grad_out, sizes_acl, dim, size, step, grad_in);
    return grad_in;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
