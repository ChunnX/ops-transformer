#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <string>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 使用 aclnnSwiGluQuantV2（与 tiling 中 group_list_type / dst_type 一致）。
 * msopgen 对 SwiGluQuant 同时生成 v1/v2 头文件；v1 无 groupListType/dstType。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x,
    const at::Tensor &smooth_scales,
    c10::optional<at::Tensor> offsets_optional,
    c10::optional<at::Tensor> group_index_optional,
    bool activate_left,
    std::string quant_mode,
    int64_t group_list_type,
    int64_t dst_type)
{
    int64_t last = x.size(-1);
    TORCH_CHECK(last % 2 == 0, "SwiGluQuant x last dim must be even");
    std::vector<int64_t> y_sizes(x.sizes().begin(), x.sizes().end());
    y_sizes.back() = last / 2;
    at::Tensor y = at::empty(
        y_sizes,
        at::TensorOptions().dtype(at::kChar).device(x.device()));
    std::vector<int64_t> scale_sizes(x.sizes().begin(), x.sizes().end() - 1);
    at::Tensor scale = at::empty(
        scale_sizes,
        at::TensorOptions().dtype(at::kFloat).device(x.device()));

    std::vector<char> quant_mode_cstr(quant_mode.begin(), quant_mode.end());
    quant_mode_cstr.push_back('\0');
    char *quant_mode_ptr = quant_mode_cstr.data();

    EXEC_NPU_CMD(
        aclnnSwiGluQuantV2,
        x,
        smooth_scales,
        offsets_optional,
        group_index_optional,
        activate_left,
        quant_mode_ptr,
        group_list_type,
        dst_type,
        y,
        scale);
    return {y, scale};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
