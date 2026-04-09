#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <string>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 与 build_out/autogen/aclnn_dynamic_quant_update_scatter.h 中
 * aclnnDynamicQuantUpdateScatterGetWorkspaceSize 一致（§7.1 勿多传输出位）：
 * varRef, varScaleRef, indices, updates, smoothScalesOptional, reduce, axis。
 * 输出即前两路 ref（msopgen 内 AddOutput 仍指向同一 aclTensor），不得在 EXEC_NPU_CMD 末尾再重复 varRef/varScaleRef。
 * reduce：autogen 为 char*；ConvertTypes 对 std::string 无特化，传 string 会导致与 GetWorkspaceSize ABI 错位（§8），须传以 \\0 结尾的可变缓冲区且在 EXEC_NPU_CMD 同步完成前保持有效。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &var,
    const at::Tensor &var_scale,
    const at::Tensor &indices,
    const at::Tensor &updates,
    c10::optional<at::Tensor> smooth_scales_optional,
    std::string reduce,
    int64_t axis)
{
    at::Tensor var_ref = var.clone();
    at::Tensor var_scale_ref = var_scale.clone();
    std::vector<char> reduce_cstr(reduce.begin(), reduce.end());
    reduce_cstr.push_back('\0');
    char *reduce_ptr = reduce_cstr.data();
    EXEC_NPU_CMD(
        aclnnCustomOp,
        var_ref,
        var_scale_ref,
        indices,
        updates,
        smooth_scales_optional,
        reduce_ptr,
        axis);
    return {var_ref, var_scale_ref};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
