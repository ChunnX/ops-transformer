#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor linear_index(at::Tensor &indices, at::Tensor &shape, int64_t axis, bool combine)
{
    // 获取输出形状（与indices相同）
    auto output_size = indices.sizes();
    
    // alloc output memory (输出类型固定为int32)
    at::Tensor result = at::empty(output_size, indices.options().dtype(at::kInt));

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnLinearIndex, indices, shape, axis, combine, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("linear_index", &linear_index, "LinearIndex operator",
          py::arg("indices"),
          py::arg("shape"),
          py::arg("axis") = -1,
          py::arg("combine") = false);
}