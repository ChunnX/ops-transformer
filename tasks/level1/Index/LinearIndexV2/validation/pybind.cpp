#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor linear_index_v2_npu(const std::vector<at::Tensor> &indices_list, 
                                const at::Tensor &stride, 
                                const at::Tensor &value_size)
{
    // 获取输出shape (与第一个索引tensor相同)
    at::Tensor result = torch::zeros_like(indices_list[0], torch::kInt32);

    // 构造TensorList
    at::TensorList indices_tensor_list(indices_list);
    
    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnLinearIndexV2, indices_tensor_list, stride, value_size, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("linear_index_v2", &linear_index_v2_npu, "LinearIndexV2 operator");
}