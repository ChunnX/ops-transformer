#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <vector>

#include "pytorch_npu_helper.hpp"

// ge 侧 10 元 padding，与 pad_v3_grad_replication_tiling CalculateValues 中 PADDING_API_LENGTH 一致
static std::vector<int64_t> grad_replication_out_sizes(const at::Tensor &gradOut, const std::vector<int64_t> &ge10)
{
    TORCH_CHECK(ge10.size() == 10, "padding must have 10 elements");
    const int64_t nd = gradOut.dim();
    TORCH_CHECK(nd == 4 || nd == 5, "PadV3GradReplication expects 4D or 5D grad_output");
    std::vector<int64_t> o(gradOut.sizes().begin(), gradOut.sizes().end());
    o[nd - 1] -= ge10[8] + ge10[9];
    o[nd - 2] -= ge10[6] + ge10[7];
    if (nd == 5) {
        o[nd - 3] -= ge10[4] + ge10[5];
    }
    return o;
}

static std::vector<int64_t> padding_tensor_to_ge10(const at::Tensor &padding)
{
    at::Tensor flat = padding.reshape(-1).to(at::kCPU, at::kLong).contiguous();
    TORCH_CHECK(flat.dim() == 1, "padding must be 1-D (after reshape)");
    TORCH_CHECK(flat.numel() == 10, "padding must have 10 elements");
    auto acc = flat.accessor<int64_t, 1>();
    std::vector<int64_t> out(10);
    for (int64_t i = 0; i < 10; ++i) {
        out[static_cast<size_t>(i)] = acc[i];
    }
    return out;
}

at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &padding)
{
    TORCH_CHECK(x.defined(), "grad_output x is not defined");
    TORCH_CHECK(padding.defined(), "padding is not defined");

    std::vector<int64_t> ge10 = padding_tensor_to_ge10(padding);
    at::Tensor paddings_tensor =
        at::tensor(ge10, at::TensorOptions().dtype(at::kLong).device(at::kCPU)).contiguous();

    std::vector<int64_t> out_sizes = grad_replication_out_sizes(x, ge10);
    at::Tensor result = at::empty(out_sizes, x.options());
    TORCH_CHECK(result.defined(), "result must be defined");

    // 与 ops-math-dev PadV3GradReplicationAiCore：INPUT(gradOutput, paddings) OUTPUT(padV3GradOut)
    EXEC_NPU_CMD(aclnnCustomOp, x, paddings_tensor, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
