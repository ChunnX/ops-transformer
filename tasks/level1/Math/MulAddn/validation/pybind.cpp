#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::TensorList to_tensor_list(const std::vector<at::Tensor>& vec) {
    return at::TensorList(vec);
}

at::Tensor mul_addn(std::vector<at::Tensor> &x1, std::vector<at::Tensor> &x2, int64_t n)
{
    torch::IntArrayRef x1Shapes = x1[0].sizes();
    torch::IntArrayRef x2Shapes = x2[0].sizes();
    std::vector<int64_t> outShape(3, 1);
    outShape[0] = x1Shapes[0];
    outShape[1] = x1Shapes[1];
    outShape[2] = x2Shapes[2];
    at::Tensor result = torch::empty(outShape, x1[0].options());

    auto x1_list = to_tensor_list(x1);
    auto x2_list = to_tensor_list(x2);
    EXEC_NPU_CMD(aclnnMulAddn, x1_list, x2_list, n, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("mul_addn", &mul_addn, "");
}