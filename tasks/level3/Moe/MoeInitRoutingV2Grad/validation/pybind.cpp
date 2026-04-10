#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
* 使用说明：
* 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnForeachExp
* 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 foreach_exp
* 
* 替换示例：
* - aclnnCustomOp -> aclnnXXX (其中XXX为算子名，如替换为aclnnForeachExp)
* - custom_pybind_api -> YYY (其中YYY为算子名的下划线形式，如foreach_exp)
* 
* 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
*/

at::Tensor custom_pybind_api(
    const at::Tensor& grad_expanded_x,
    const at::Tensor& expanded_row_idx,
    int64_t top_k,
    int64_t drop_pad_mode,
    int64_t active_num
) {
    // 获取形状信息
    int64_t tokens_num = expanded_row_idx.size(0) / top_k;
    int64_t hidden_size = grad_expanded_x.size(-1);
    
    // 计算输出形状
    std::vector<int64_t> grad_x_shape = {tokens_num, hidden_size};
    
    // 创建输出张量
    c10::TensorOptions options = grad_expanded_x.options().dtype(grad_expanded_x.scalar_type());
    at::Tensor grad_x = at::empty(grad_x_shape, options);
    
    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnCustomOp, grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num, grad_x);
    
    return grad_x;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `moe_init_routing_v2_grad`
    m.def("moe_init_routing_v2_grad", &custom_pybind_api, "");
}
