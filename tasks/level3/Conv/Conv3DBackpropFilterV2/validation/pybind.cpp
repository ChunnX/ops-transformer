#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_backprop_filter_v2(
    const at::Tensor &x,
    const at::Tensor &filter_size,
    const at::Tensor &out_backprop,
    at::IntArrayRef strides,
    at::IntArrayRef pads,
    at::IntArrayRef dilations,
    int64_t groups,
    const std::string &data_format,
    bool enable_hf32,
    const at::Tensor &filter_tensor)
{
    auto filter_size_cpu = filter_size.to(torch::kCPU).to(torch::kInt).contiguous();
    int32_t* filter_size_data = filter_size_cpu.data_ptr<int32_t>();
    int64_t dk = filter_size_data[0];
    int64_t hk = filter_size_data[1];
    int64_t wk = filter_size_data[2];
    int64_t cin = filter_size_data[3];
    int64_t cout = filter_size_data[4];

    at::Tensor x_ncdhw;
    at::Tensor out_backprop_ncdhw;
    if (data_format == "NDHWC") {
        x_ncdhw = x.permute({0, 4, 1, 2, 3}).contiguous();
        out_backprop_ncdhw = out_backprop.permute({0, 4, 1, 2, 3}).contiguous();
    } else {
        x_ncdhw = x.contiguous();
        out_backprop_ncdhw = out_backprop.contiguous();
    }

    std::vector<int64_t> weight_shape = {cout, cin / groups, dk, hk, wk};
    at::Tensor grad_weight = at::empty(
        weight_shape,
        at::TensorOptions().dtype(x.scalar_type()).device(x.device()));

    at::Tensor grad_input_placeholder;
    at::Tensor grad_bias_placeholder;

    std::vector<int64_t> bias_sizes = {cout};
    std::vector<int64_t> output_padding = {0, 0, 0};
    std::array<bool, 3> output_mask = {false, true, false};
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
                 out_backprop_ncdhw, x_ncdhw, filter_tensor, biasSizes_acl,
                 strides_acl, pads_acl, dilations_acl,
                 transposed, outputPads_acl, groups, outputMask_acl, cube_math_type,
                 grad_input_placeholder, grad_weight, grad_bias_placeholder);

    return grad_weight;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("conv3d_backprop_filter_v2", &conv3d_backprop_filter_v2, "");
}
