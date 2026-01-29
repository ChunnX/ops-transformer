#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor histogram_v2(at::Tensor &input, at::Tensor &min, at::Tensor &max, int64_t bins)
{
    std::vector<int64_t> outShape(1, bins);
    at::Tensor result = torch::zeros(outShape, input.options().dtype(torch::kInt32));

    EXEC_NPU_CMD(aclnnHistogramV2, input, min, max, bins, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("histogram_v2", &histogram_v2, "");
}