#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <vector>

#include "pytorch_npu_helper.hpp"

static std::vector<int64_t> circular_pad_out_sizes(const at::Tensor &x, const std::vector<int64_t> &padding) {
    TORCH_CHECK(padding.size() == 4 || padding.size() == 6, "padding length must be 4 (2d) or 6 (3d)");
    std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
    const int64_t nd = static_cast<int64_t>(out_sizes.size());
    if (padding.size() == 4) {
        TORCH_CHECK(nd >= 3, "circular_pad 2d expects self dim >= 3");
        out_sizes[nd - 2] += padding[2] + padding[3];
        out_sizes[nd - 1] += padding[0] + padding[1];
    } else {
        TORCH_CHECK(nd >= 4, "circular_pad 3d expects self dim >= 4");
        out_sizes[nd - 3] += padding[4] + padding[5];
        out_sizes[nd - 2] += padding[2] + padding[3];
        out_sizes[nd - 1] += padding[0] + padding[1];
    }
    return out_sizes;
}

static std::vector<int64_t> padding_tensor_to_int64_vec(const at::Tensor &padding) {
    at::Tensor flat = padding.reshape(-1).to(at::kCPU, at::kLong).contiguous();
    TORCH_CHECK(flat.dim() == 1, "padding must be 1-D (after reshape)");
    TORCH_CHECK(flat.numel() == 4 || flat.numel() == 6, "padding must have 4 or 6 elements");
    auto acc = flat.accessor<int64_t, 1>();
    std::vector<int64_t> out(static_cast<size_t>(flat.numel()));
    for (int64_t i = 0; i < flat.numel(); ++i) {
        out[static_cast<size_t>(i)] = acc[i];
    }
    return out;
}

at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &padding)
{
    TORCH_CHECK(x.defined(), "Input x is not defined");
    TORCH_CHECK(padding.defined(), "padding is not defined");

    std::vector<int64_t> padding_vec = padding_tensor_to_int64_vec(padding);
    // paddings 必须放在 Host：optiling::GetShapeAttrsInfo 在 CPU 上通过 GetData<int64_t>() 读 padding，
    // 若 tensor 在 NPU 上，GetData 返回的是设备地址，Host 解引用会 SIGSEGV（打印出的 0x48xxxxxx 即设备侧指针形态）。
    at::Tensor paddings_tensor =
        at::tensor(padding_vec, at::TensorOptions().dtype(at::kLong).device(at::kCPU)).contiguous();

    std::vector<int64_t> out_sizes = circular_pad_out_sizes(x, padding_vec);
    at::Tensor result = at::empty(out_sizes, x.options());

    TORCH_CHECK(result.defined(), "result must be defined");

    // autogen: aclnnCircularPadGetWorkspaceSize(x, paddings, out, ...)
    EXEC_NPU_CMD(aclnnCustomOp, x, paddings_tensor, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
