#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

// 注意：aclnnCustomOp 和 custom_pybind_api 是占位符，
// 框架在编译时自动替换为 aclnn<OpName> 和 <kernel_name>

// 如果是 OPTIONAL 输入，可改为 c10::optional<at::Tensor>
// 如果有属性参数，在输入张量和输出张量之间添加（int64_t / float / bool / at::IntArrayRef / char*）

at::Tensor custom_pybind_api(at::Tensor &input)
{
    at::Tensor result = torch::empty_like(input);
    // 参数顺序：输入张量 → 属性参数 → 输出张量（必须在最后）
    EXEC_NPU_CMD(aclnnCustomOp, input, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "kernel_gen_ops custom operator");
}
