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

std::tuple<at::Tensor, at::Tensor, at::Tensor> custom_pybind_api(
    const at::Tensor& x,
    const at::Tensor& row_idx,
    const at::Tensor& expert_idx,
    int64_t active_num
) {
    // 获取形状信息
    int64_t num_rows = x.size(0);
    int64_t h = x.size(1);
    int64_t k = expert_idx.size(1);
    
    // 计算输出形状
    int64_t output_rows = active_num > 0 ? std::min(active_num, num_rows) * k : num_rows * k;
    
    // 创建输出张量
    at::Tensor expanded_x_out = at::empty({output_rows, h}, x.options());
    at::Tensor expanded_row_idx_out = at::empty({num_rows * k}, x.options().dtype(torch::kInt32));
    at::Tensor expanded_expert_idx_out = at::empty({num_rows * k}, x.options().dtype(torch::kInt32));
    
    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnCustomOp, x, row_idx, expert_idx, active_num, 
                 expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out);
    
    return std::make_tuple(expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `moe_init_routing`
    m.def("moe_init_routing", &custom_pybind_api, "");
}
