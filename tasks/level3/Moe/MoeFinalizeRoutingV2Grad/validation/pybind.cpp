#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor> custom_pybind_api(
    const at::Tensor& gradY,
    const at::Tensor& expandedRowIdx,
    const at::Tensor& expandedXOptional,
    const at::Tensor& scalesOptional,
    const at::Tensor& expertIdxOptional,
    const at::Tensor& biasOptional,
    int64_t dropPadMode,
    int64_t activeNum,
    int64_t expertNum,
    int64_t expertCapacity) {
    at::Tensor gradExpandedXOut;
    at::Tensor gradScalesOut;

    if (dropPadMode == 0) {
        if (activeNum > 0 && activeNum < expandedRowIdx.size(0)) {
            gradExpandedXOut = at::empty({activeNum, gradY.size(1)}, gradY.options());
        } else {
            gradExpandedXOut = at::empty({expandedRowIdx.size(0), gradY.size(1)}, gradY.options());
        }
    } else {
        gradExpandedXOut = at::empty({expertNum, expertCapacity, gradY.size(1)}, gradY.options());
    }
    if (scalesOptional.numel() > 0) {
        gradScalesOut = at::empty_like(scalesOptional);
    } else {
        // 与 op infershape 一致：无 scales 时 grad_scales 为 (R, 1)，R = gradY 第 0 维
        gradScalesOut = at::empty({gradY.size(0), 1}, gradY.options());
    }

    at::Tensor empty_tensor = at::Tensor();
    const at::Tensor& expandedX = expandedXOptional.numel() > 0 ? expandedXOptional : empty_tensor;
    const at::Tensor& scales = scalesOptional.numel() > 0 ? scalesOptional : empty_tensor;
    const at::Tensor& expertIdx = expertIdxOptional.numel() > 0 ? expertIdxOptional : empty_tensor;
    const at::Tensor& bias = biasOptional.numel() > 0 ? biasOptional : empty_tensor;

    EXEC_NPU_CMD(
        aclnnCustomOp,
        gradY,
        expandedRowIdx,
        expandedX,
        scales,
        expertIdx,
        bias,
        dropPadMode,
        activeNum,
        expertNum,
        expertCapacity,
        gradExpandedXOut,
        gradScalesOut);

    return std::make_tuple(gradExpandedXOut, gradScalesOut);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("moe_finalize_routing_v2_grad", &custom_pybind_api, "");
}