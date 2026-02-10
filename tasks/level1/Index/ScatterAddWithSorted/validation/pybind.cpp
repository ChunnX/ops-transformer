#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor scatter_add_with_sorted(at::Tensor &var, at::Tensor &value, at::Tensor &sorted_index, c10::optional<at::Tensor> pos)
{
    // alloc output memory
    at::Tensor result = var.clone();

    // call aclnn interface to perform the computation
    if (pos.has_value()) {
        EXEC_NPU_CMD(aclnnScatterAddWithSorted, result, value, sorted_index, pos.value(), result);
    } else {
        at::Tensor empty_pos = at::empty({0}, var.options().dtype(at::kInt));
        EXEC_NPU_CMD(aclnnScatterAddWithSorted, result, value, sorted_index, empty_pos, result);
    }
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("scatter_add_with_sorted", &scatter_add_with_sorted, "ScatterAddWithSorted operation",
          py::arg("var"), py::arg("value"), py::arg("sorted_index"), py::arg("pos") = py::none());
}