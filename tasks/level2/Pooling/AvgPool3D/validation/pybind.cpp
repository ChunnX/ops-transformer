#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t value_num = 0;
    if (factor == 0) {
        return value_num;
    }
    if (value % factor == 0) {
        value_num = value / factor;
    } else {
        value_num = value / factor + 1;
    }

    return value_num;
}

std::vector<int64_t> avg_pool3d_npu_output_size(const at::Tensor &x, c10::IntArrayRef kernel_size,
                                                c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                bool ceil_mode)
{
    if (x.dim() != 5) {
        throw std::runtime_error("tensor x's dimension must be 5");
    }
    if (kernel_size.size() != 3) {
        throw std::runtime_error("kernel_size length should be 3");
    }
    if (stride.size() != 3) {
        throw std::runtime_error("stride length should be 3");
    }
    if (stride[0] * stride[1] * stride[2] == 0) {
        throw std::runtime_error("stride should not contain zero");
    }
    if (padding.size() != 3) {
        throw std::runtime_error("padding length should be 3");
    }

    int x_d = x.size(-3);
    int x_h = x.size(-2);
    int x_w = x.size(-1);

    int64_t kernel_d = ceil_mode ? (CeilDiv(x_d + 2 * padding[0] - kernel_size[0], stride[0]) + 1) :
                                   ((x_d + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
    int64_t kernel_h = ceil_mode ? (CeilDiv(x_h + 2 * padding[1] - kernel_size[1], stride[1]) + 1) :
                                   ((x_h + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);
    int64_t kernel_w = ceil_mode ? (CeilDiv(x_w + 2 * padding[2] - kernel_size[2], stride[2]) + 1) :
                                   ((x_w + 2 * padding[2] - kernel_size[2]) / stride[2] + 1);
    if (kernel_d <= 0) {
        throw std::runtime_error("kernel_d has to be positive, but got " + std::to_string(kernel_d));
    }
    if (kernel_h <= 0) {
        throw std::runtime_error("kernel_h has to be positive, but got " + std::to_string(kernel_h));
    }
    if (kernel_w <= 0) {
        throw std::runtime_error("kernel_w has to be positive, but got " + std::to_string(kernel_w));
    }

    if (ceil_mode) {
        if ((kernel_d - 1) * stride[0] >= x_d + padding[0]) {
            --kernel_d;
        }

        if ((kernel_h - 1) * stride[1] >= x_h + padding[1]) {
            --kernel_h;
        }

        if ((kernel_w - 1) * stride[2] >= x_w + padding[2]) {
            --kernel_w;
        }
    }

    std::vector<int64_t> output_size = {x.size(0), x.size(1), kernel_d, kernel_h, kernel_w};

    return output_size;
}

std::vector<int64_t> calc_avg_pool3d_output_size(const at::Tensor &x, at::IntArrayRef kernel_size,
                                                 at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                                 bool count_include_pad)
{
    // generalize kernels, strides and paddings to 3D-shape
    if (kernel_size.empty()) {
        throw std::runtime_error("kernel_size must be a tuple of three ints ");
    }
    const int64_t k_d = kernel_size[0];
    const int64_t k_h = kernel_size.size() == 1 ? k_d : kernel_size[1];
    const int64_t k_w = kernel_size.size() == 1 ? k_d : kernel_size[2];
    std::vector<int64_t> kernel_sizes = {k_d, k_h, k_w};
    at::IntArrayRef kernels = at::IntArrayRef(kernel_sizes);

    const int64_t s_d = stride.empty() ? k_d : stride[0];
    const int64_t s_h = stride.empty() ? k_h : stride.size() == 1 ? s_d : stride[1];
    const int64_t s_w = stride.empty() ? k_w : stride.size() == 1 ? s_d : stride[2];
    std::vector<int64_t> stride_sizes = {s_d, s_h, s_w};
    if (s_d == 0 || s_h == 0 || s_w == 0) {
        throw std::runtime_error("stride should not be zero");
    }
    at::IntArrayRef strides = at::IntArrayRef(stride_sizes);

    const int64_t pad_d = padding[0];
    const int64_t pad_h = padding.size() == 1 ? pad_d : padding[1];
    const int64_t pad_w = padding.size() == 1 ? pad_d : padding[2];
    std::vector<int64_t> padding_sizes = {pad_d, pad_h, pad_w};
    if (pad_d < 0 || pad_h < 0 || pad_w < 0) {
        throw std::runtime_error("pad should not be less than 0");
    }
    if (pad_d > k_d / 2 || pad_h > k_h / 2 || pad_w > k_w / 2) {
        throw std::runtime_error("pad should be smaller than or equal to half of kernel size");
    }
    at::IntArrayRef paddings = at::IntArrayRef(padding_sizes);

    auto output_size = avg_pool3d_npu_output_size(x, kernels, strides, paddings, ceil_mode);

    return output_size;
}

at::Tensor avg_pool3_d(const at::Tensor &x, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                      int64_t divisor_override)
{
    auto outShape = calc_avg_pool3d_output_size(x, kernel_size, stride, padding, ceil_mode, count_include_pad);

    at::Tensor result = torch::empty(outShape, x.options());

    EXEC_NPU_CMD(aclnnAvgPool3d, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, result);

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("avg_pool3_d", &avg_pool3_d, "");
}