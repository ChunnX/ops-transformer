#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor gelu_mul(at::Tensor &input, std::string approximate)
{
    torch::IntArrayRef shapes = input.sizes();
    std::vector<int64_t> outShape(shapes.begin(), shapes.end());
    outShape[outShape.size() - 1] = outShape[outShape.size() - 1] >> 1;
    at::Tensor result = torch::empty(outShape, input.options());

    const char* approximateStr = approximate.c_str();
    EXEC_NPU_CMD(aclnnGeluMul, input, approximateStr, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("gelu_mul", &gelu_mul, "");
}