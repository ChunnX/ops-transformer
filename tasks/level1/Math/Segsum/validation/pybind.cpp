#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * segsum (ExpSegsum): 输入 x (..., N)，输出 y (..., N, N)，与 module.py 中 ModelNew 的 forward 一致。
 */
at::Tensor segsum(at::Tensor& x)
{
    std::vector<int64_t> in_shape = x.sizes().vec();
    int64_t n = in_shape.back();
    std::vector<int64_t> out_shape = in_shape;
    out_shape.push_back(n);

    at::Tensor result = torch::empty(out_shape, x.options());

    EXEC_NPU_CMD(aclnnSegsum, x, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("segsum", &segsum, "");
}
