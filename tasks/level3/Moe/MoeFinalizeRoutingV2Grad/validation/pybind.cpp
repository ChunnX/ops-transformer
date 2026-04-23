#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::tuple<at::Tensor, at::Tensor> custom_pybind_api(
    const at::Tensor& gradY,
    const at::Tensor& expandedRowIdx,
    const c10::optional<at::Tensor>& expandedXOptional,
    const c10::optional<at::Tensor>& scalesOptional,
    const c10::optional<at::Tensor>& expertIdxOptional,
    const c10::optional<at::Tensor>& biasOptional,
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

    const bool has_scales =
        scalesOptional.has_value() && scalesOptional->defined() && scalesOptional->numel() > 0;
    if (has_scales) {
        gradScalesOut = at::empty_like(*scalesOptional);
    } else {
        gradScalesOut = at::empty({gradY.size(0), 1}, gradY.options());
    }

    EXEC_NPU_CMD(
        aclnnCustomOp,
        gradY,
        expandedRowIdx,
        expandedXOptional,
        scalesOptional,
        expertIdxOptional,
        biasOptional,
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
