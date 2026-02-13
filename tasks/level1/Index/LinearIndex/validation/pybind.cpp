#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

at::Tensor linear_index_custom(at::Tensor &indices, at::Tensor &shape, int64_t axis, bool combine)
{
    // 创建输出 Tensor，数据类型为 int32 (ge::DT_INT32)
    at::Tensor result = torch::empty_like(indices, indices.options().dtype(at::kInt));

    // 调用 aclnn 接口
    // 参数顺序根据 EXEC_NPU_CMD 宏定义和算子原型推断：输入1, 输入2, 属性1, 属性2, 输出
    EXEC_NPU_CMD(aclnnLinearIndex, indices, shape, axis, combine, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("linear_index", &linear_index_custom, "LinearIndex custom op");
}