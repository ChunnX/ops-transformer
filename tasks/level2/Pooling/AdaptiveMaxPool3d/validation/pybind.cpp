#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool3d(at::Tensor &x, at::IntArrayRef &output_size)
{
    if (x.dim() != 5) {
        throw std::runtime_error("The dim of x should be 5");
    }
    if (output_size.size() != 3) {
        throw std::runtime_error("The size of output_size should be 3");
    }
    std::vector<int64_t> outShape(5, 1);
    outShape[0] = x.size(0);
    outShape[1] = x.size(1);
    outShape[2] = output_size[0];
    outShape[3] = output_size[1];
    outShape[4] = output_size[2];
    at::Tensor outputOut = torch::empty(outShape, x.options());
    at::Tensor indicesOut = torch::empty(outShape, x.options().dtype(torch::kInt32));


    EXEC_NPU_CMD(aclnnAdaptiveMaxPool3d, x, output_size, outputOut, indicesOut);

    return std::make_tuple(outputOut, indicesOut);
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("adaptive_max_pool3d", &adaptive_max_pool3d, "");
}