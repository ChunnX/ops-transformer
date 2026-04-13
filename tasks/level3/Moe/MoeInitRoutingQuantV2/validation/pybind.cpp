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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> custom_pybind_api(
    const at::Tensor& x,
    const at::Tensor& expert_idx,
    const c10::optional<at::Tensor>& scale_optional,
    const c10::optional<at::Tensor>& offset_optional,
    int64_t active_num,
    int64_t expert_capacity,
    int64_t expert_num,
    int64_t drop_pad_mode,
    int64_t expert_tokens_count_or_cumsum_flag,
    bool expert_tokens_before_capacity_flag,
    int64_t quant_mode
) {
    // 获取形状信息
    int64_t num_rows = x.size(0);
    int64_t h = x.size(1);
    int64_t k = expert_idx.size(1);
    
    // 计算输出形状
    at::Tensor expanded_x_out;
    at::Tensor expanded_row_idx_out = at::empty({num_rows * k}, x.options().dtype(torch::kInt32));
    at::Tensor expert_tokens_count_or_cumsum;
    at::Tensor expert_tokens_before_capacity;
    at::Tensor dynamic_quant_scale_out;
    
    if (drop_pad_mode == 0) {
        // Dropless 模式
        if (active_num > 0) {
            // Active 场景
            int64_t active_size = std::min(active_num, num_rows * k);
            expanded_x_out = at::empty({active_size, h}, x.options().dtype(torch::kInt8));
        } else {
            // 非 Active 场景
            expanded_x_out = at::empty({num_rows * k, h}, x.options().dtype(torch::kInt8));
        }
    } else {
        // Drop/Pad 模式
        expanded_x_out = at::empty({expert_num, expert_capacity, h}, x.options().dtype(torch::kInt8));
    }
    
    // 处理可选输出
    if (expert_tokens_count_or_cumsum_flag > 0 && drop_pad_mode == 0) {
        expert_tokens_count_or_cumsum = at::empty({expert_num}, x.options().dtype(torch::kInt32));
    } else {
        expert_tokens_count_or_cumsum = at::empty({0}, x.options().dtype(torch::kInt32));
    }
    
    if (expert_tokens_before_capacity_flag && drop_pad_mode == 1) {
        expert_tokens_before_capacity = at::empty({expert_num}, x.options().dtype(torch::kInt32));
    } else {
        expert_tokens_before_capacity = at::empty({0}, x.options().dtype(torch::kInt32));
    }
    
    if (quant_mode == 1) {
        if (drop_pad_mode == 0) {
            int64_t size = active_num > 0 ? std::min(active_num, num_rows * k) : num_rows * k;
            dynamic_quant_scale_out = at::empty({size}, x.options().dtype(torch::kFloat32));
        } else {
            dynamic_quant_scale_out = at::empty({expert_num * expert_capacity}, x.options().dtype(torch::kFloat32));
        }
    } else {
        dynamic_quant_scale_out = at::empty({0}, x.options().dtype(torch::kFloat32));
    }
    
    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnCustomOp, x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, 
                 drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode, 
                 expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum, expert_tokens_before_capacity, 
                 dynamic_quant_scale_out);
    
    return std::make_tuple(expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum, 
                           expert_tokens_before_capacity, dynamic_quant_scale_out);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `moe_init_routing_quant_v2`
    m.def("moe_init_routing_quant_v2", &custom_pybind_api, "");
}
