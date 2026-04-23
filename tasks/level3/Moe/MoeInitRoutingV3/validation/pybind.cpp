#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor& x,
    const at::Tensor& expert_idx,
    const c10::optional<at::Tensor>& scale_optional,
    const c10::optional<at::Tensor>& offset_optional,
    int64_t active_num,
    int64_t expert_capacity,
    int64_t expert_num,
    int64_t drop_pad_mode,
    int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag,
    int64_t quant_mode,
    const std::vector<int64_t>& active_expert_range,
    int64_t row_idx_type)
{
    const int64_t tokens_num = x.size(0);
    const int64_t hidden_size = x.size(1);
    const int64_t topk = expert_idx.dim() == 2 ? expert_idx.size(1) : 1;

    int64_t expanded_x_out_size = tokens_num * topk;
    if (active_num > 0 && active_num < expanded_x_out_size) {
        expanded_x_out_size = active_num;
    }

    c10::TensorOptions x_options = x.options().dtype(x.scalar_type());
    // quant_mode 0/1 时算子输出为 INT8（与 validation/module.py CPU 参考一致），不能用输入 dtype 分配。
    c10::ScalarType expanded_x_dtype = x.scalar_type();
    if (quant_mode == 0 || quant_mode == 1) {
        expanded_x_dtype = torch::kInt8;
    }
    c10::TensorOptions expanded_x_options = x.options().dtype(expanded_x_dtype);
    at::Tensor expanded_x_out;
    if (drop_pad_mode == 1) {
        expanded_x_out = at::empty({expert_num, expert_capacity, hidden_size}, expanded_x_options);
    } else {
        expanded_x_out = at::empty({expanded_x_out_size, hidden_size}, expanded_x_options);
    }

    c10::TensorOptions idx_options = x.options().dtype(torch::kInt32);
    at::Tensor expanded_row_idx_out = at::empty({tokens_num * topk}, idx_options);

    at::Tensor expert_tokens_count_or_cumsum_out;
    if (expert_tokens_num_flag) {
        if (expert_tokens_num_type == 2) {
            expert_tokens_count_or_cumsum_out = at::empty({expert_num, 2}, x.options().dtype(torch::kInt64));
        } else {
            expert_tokens_count_or_cumsum_out = at::empty({expert_num}, x.options().dtype(torch::kInt64));
        }
    } else {
        expert_tokens_count_or_cumsum_out = at::empty({0}, x.options().dtype(torch::kInt64));
    }

    at::Tensor expanded_scale_out;
    if (scale_optional.has_value() && quant_mode != 0) {
        if (drop_pad_mode == 1) {
            expanded_scale_out = at::empty({expert_num, expert_capacity}, scale_optional.value().options());
        } else {
            expanded_scale_out = at::empty({expanded_x_out_size}, scale_optional.value().options());
        }
    } else {
        expanded_scale_out = at::empty({0}, x_options.dtype(torch::kFloat));
    }

    aclIntArray* active_expert_range_acl = nullptr;
    if (active_expert_range.size() == 2) {
        active_expert_range_acl = ConvertType(at::IntArrayRef(active_expert_range));
    }

    EXEC_NPU_CMD(aclnnCustomOp,
        x,
        expert_idx,
        scale_optional,
        offset_optional,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        quant_mode,
        active_expert_range_acl,
        row_idx_type,
        expanded_x_out,
        expanded_row_idx_out,
        expert_tokens_count_or_cumsum_out,
        expanded_scale_out);

    return {expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum_out, expanded_scale_out};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_init_routing_v3", &custom_pybind_api, "");
}
