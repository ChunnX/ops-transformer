#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_backprop_input_v2(
    const at::Tensor &input_size,
    const at::Tensor &filter,
    const at::Tensor &out_backprop,
    const at::Tensor &input_tensor,
    at::IntArrayRef strides,
    at::IntArrayRef pads,
    at::IntArrayRef dilations,
    int64_t groups,
    const std::string &data_format,
    bool enable_hf32)
{
    auto input_size_cpu = input_size.to(torch::kCPU).to(torch::kInt).contiguous();
    int32_t* input_size_data = input_size_cpu.data_ptr<int32_t>();
    int64_t N = input_size_data[0];
    int64_t D = input_size_data[1];
    int64_t H = input_size_data[2];
    int64_t W = input_size_data[3];
    int64_t Cin = input_size_data[4];

    at::Tensor grad_input;
    if (data_format == "NDHWC") {
        grad_input = at::empty(
            {N, D, H, W, Cin},
            at::TensorOptions().dtype(out_backprop.scalar_type()).device(out_backprop.device()));
    } else {
        grad_input = at::empty(
            {N, Cin, D, H, W},
            at::TensorOptions().dtype(out_backprop.scalar_type()).device(out_backprop.device()));
    }

    at::Tensor grad_weight_placeholder;
    at::Tensor grad_bias_placeholder;

    std::vector<int64_t> bias_sizes = {filter.size(0)};
    std::vector<int64_t> output_padding = {0, 0, 0};
    std::array<bool, 3> output_mask = {true, false, false};
    int8_t cube_math_type = 1;
    bool transposed = false;

    at::IntArrayRef bias_sizes_ref(bias_sizes);
    at::IntArrayRef output_padding_ref(output_padding);

    aclIntArray* biasSizes_acl = ConvertType(bias_sizes_ref);
    aclIntArray* strides_acl = ConvertType(strides);
    aclIntArray* pads_acl = ConvertType(pads);
    aclIntArray* dilations_acl = ConvertType(dilations);
    aclIntArray* outputPads_acl = ConvertType(output_padding_ref);
    aclBoolArray* outputMask_acl = ConvertType(output_mask);

    EXEC_NPU_CMD(aclnnConvolutionBackward,
                 out_backprop, input_tensor, filter, biasSizes_acl,
                 strides_acl, pads_acl, dilations_acl,
                 transposed, outputPads_acl, groups, outputMask_acl, cube_math_type,
                 grad_input, grad_weight_placeholder, grad_bias_placeholder);

    return grad_input;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("conv3d_backprop_input_v2", &conv3d_backprop_input_v2, "");
}
