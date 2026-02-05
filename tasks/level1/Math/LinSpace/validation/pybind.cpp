#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
#define GET_SCALAR_AND_RUN(TYPE)                                        \
    auto startScalar = at::Scalar(start.index({0}).item<TYPE>());     \
    auto endScalar = at::Scalar(end.index({0}).item<TYPE>());         \
    EXEC_NPU_CMD(aclnnLinspace, startScalar, endScalar, steps, result)

at::Tensor lin_space(at::Tensor &start, at::Tensor &end, int64_t steps)
{
    std::vector<int64_t> outShape(1, steps);
    at::Tensor result = torch::empty(outShape, start.options());

    if ((start.sizes().size() != 1) || (start.numel() != 1)) {
        throw std::runtime_error("start's shape should be [1]");
    }
    if ((end.sizes().size() != 1) || (end.numel() != 1)) {
        throw std::runtime_error("end's shape should be [1]");
    }

    if (start.scalar_type() == at::ScalarType::Char) {
        GET_SCALAR_AND_RUN(int8_t);
    } else if (start.scalar_type() == at::ScalarType::Byte) {
        GET_SCALAR_AND_RUN(uint8_t);
    } else if (start.scalar_type() == at::ScalarType::Short) {
        GET_SCALAR_AND_RUN(int16_t);
    } else if (start.scalar_type() == at::ScalarType::Int) {
        GET_SCALAR_AND_RUN(int32_t);
    } else if (start.scalar_type() == at::ScalarType::Float) {
        GET_SCALAR_AND_RUN(float);
    } else if (start.scalar_type() == at::ScalarType::Half) {
        GET_SCALAR_AND_RUN(at::Half);
    } else if (start.scalar_type() == at::ScalarType::BFloat16) {
        GET_SCALAR_AND_RUN(at::BFloat16);
    } else {
        throw std::runtime_error("Unsupported type for lin_space");
    }

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("lin_space", &lin_space, "");
}