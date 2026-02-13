#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor scatter_add_with_sorted_custom(at::Tensor &var, at::Tensor &value, at::Tensor &sorted_index, c10::optional<at::Tensor> pos)
{
    char* reduction = "add";
    
    at::Tensor pos_tensor = pos.has_value() ? pos.value().to(at::kInt) : at::empty({}, at::kInt);

    EXEC_NPU_CMD(aclnnScatterAddWithSorted, var, value, sorted_index, pos_tensor, reduction);
    
    return var;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for scatter_add_with_sorted";
    m.def("scatter_add_with_sorted", &scatter_add_with_sorted_custom, "ScatterAddWithSorted custom op");
}