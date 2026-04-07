#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>

#include "pytorch_npu_helper.hpp"

/**
 * 与 compile 后 build_out/autogen/aclnn_fake_quant_affine_cachemask.h 中 GetWorkspaceSize 形参表一致（§7.1）。
 * 输出 y/mask 须先 AllocTensor，勿把未初始化指针传入 EXEC_NPU_CMD。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &x,
    const at::Tensor &scale,
    const at::Tensor &zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max)
{
    at::Tensor y = at::empty_like(x);
    at::Tensor mask = at::empty(
        x.sizes(),
        at::TensorOptions().dtype(at::kBool).device(x.device()));
    EXEC_NPU_CMD(
        aclnnCustomOp,
        x,
        scale,
        zero_point,
        axis,
        quant_min,
        quant_max,
        y,
        mask);
    return {y, mask};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
