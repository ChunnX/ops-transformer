#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <string>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 与 build_out/autogen/aclnn_ascend_quant_v2.h 一致；EXEC_NPU_CMD 第一参须为 aclnnCustomOp，
 * batch_compile 会替换为 aclnnAscendQuantV2（与 libcust_opapi 导出符号一致）。
 * axis 与 golden 一致固定 -1；dstType 在 autogen 中为 int64_t（见 precision-debug §8）。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x,
    const at::Tensor &scale,
    c10::optional<at::Tensor> offset,
    bool sqrt_mode,
    std::string round_mode,
    int32_t dst_type)
{
    at::Tensor y = at::empty(
        x.sizes(),
        at::TensorOptions().dtype(at::kChar).device(x.device()));
    bool sqrt_mode_arg = sqrt_mode;
    int64_t dst_type_arg = static_cast<int64_t>(dst_type);
    int64_t axis_arg = static_cast<int64_t>(-1);
    std::string round_mode_storage = std::move(round_mode);
    std::vector<char> round_mode_buf(round_mode_storage.begin(), round_mode_storage.end());
    round_mode_buf.push_back('\0');
    char *round_mode_arg = round_mode_buf.data();
    EXEC_NPU_CMD(
        aclnnCustomOp,
        x,
        scale,
        offset,
        sqrt_mode_arg,
        round_mode_arg,
        dst_type_arg,
        axis_arg,
        y);
    return {y};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
