#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 与 build_out/autogen/aclnn_dynamic_quant_v2.h 一致：
 * x, smoothScalesOptional, groupIndexOptional, dstType, isSymmetrical,
 * quantModeOptional, yOut, scaleOut, offsetOut。
 * ge::DT_INT8 枚举值为 2（precision-debug §8）。
 * isSymmetrical=false、quantMode="pertoken" 与 module.py 非对称 int8 golden 一致。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x,
    c10::optional<at::Tensor> smooth_scales_optional,
    c10::optional<at::Tensor> group_index_optional,
    int64_t dst_type)
{
    at::Tensor y = at::empty(
        x.sizes(),
        at::TensorOptions().dtype(at::kChar).device(x.device()));
    std::vector<int64_t> scale_sizes(x.sizes().begin(), x.sizes().end() - 1);
    at::Tensor scale = at::empty(
        scale_sizes,
        at::TensorOptions().dtype(at::kFloat).device(x.device()));
    at::Tensor offset = at::empty(
        scale_sizes,
        at::TensorOptions().dtype(at::kFloat).device(x.device()));
    bool is_symmetrical = false;
    char quant_mode[] = "pertoken";
    EXEC_NPU_CMD(
        aclnnCustomOp,
        x,
        smooth_scales_optional,
        group_index_optional,
        dst_type,
        is_symmetrical,
        quant_mode,
        y,
        scale,
        offset);
    return {y, scale, offset};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
