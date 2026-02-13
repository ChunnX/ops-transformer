#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

template <typename To, typename From>
inline constexpr To safe_downcast(const From v)
{
    if (c10::overflows<To>(v)) {
        throw std::runtime_error("Cast failed: out of range!");
    }
    return static_cast<To>(v);
}

void max_pool3d_with_indices_parameter_check(const at::Tensor &x, at::IntArrayRef kernel_size,
                                             at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation)
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
    if (padding.size() != 3) {
        throw std::runtime_error("padding length should be 3");
    }
    if (dilation.size() != 3) {
        throw std::runtime_error("dilation length should be 3");
    }
}

void pool3d_shape_check(const at::Tensor &input, int64_t nslices, int kT, int kH, int kW, int dT, int dH, int dW,
                        int pT, int pH, int pW, int dilationT, int dilationH, int dilationW,
                        int64_t itime, int64_t iheight, int64_t iwidth,
                        int64_t otime, int64_t oheight, int64_t owidth)
{
    if (kT <= 0 || kW <= 0 || kH <=0) {
        throw std::runtime_error("kernel_size should be greater than zero.");
    }
    if (dT <= 0 || dW <= 0 || dH <=0) {
        throw std::runtime_error("stride length should be 3");
    }
    if (dilationT <= 0 || dilationW <= 0 || dilationH <=0) {
        throw std::runtime_error("dilation length should be 3");
    }
    if (kT/2 < pT || kW/2 < pW || kH/2 < pH) {
        throw std::runtime_error("pad should be smaller than or equal to half of kernel size");
    }
    if (otime < 1 || owidth < 1 || oheight < 1) {
        throw std::runtime_error("output size if too small");
    }
}

template<typename T>
static inline T div_rtn(T x, T y)
{
    int q = x / y;
    int r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0)))
        --q;
    return q;
}

template<typename T>
static inline T pooling_output_shape_pad_lr(
    T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation, bool ceil_mode)
{
    T outputSize = div_rtn<T>(inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        if ((outputSize - 1) * stride >= inputSize + pad_l) {
            --outputSize;
        }
    }
    return outputSize;
}

template<typename T>
static inline T pooling_output_shape(T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode)
{
    if (stride == 0) {
        throw std::runtime_error("stride should not be 0");
    }
    if (pad < 0) {
        throw std::runtime_error("pad must be non-negative");
    }
    if (pad > kernelSize / 2) {
        throw std::runtime_error("pad should be at most half of kernel size");
    }
    return pooling_output_shape_pad_lr(inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> max_pool3d_with_argmax_v2(
    const at::Tensor &x, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode)
{
    max_pool3d_with_indices_parameter_check(x, kernel_size, stride, padding, dilation);

    const int k_D = safe_downcast<int, int64_t>(kernel_size[0]);
    const int k_H = kernel_size.size() == 1 ? k_D : safe_downcast<int, int64_t>(kernel_size[1]);
    const int k_W = kernel_size.size() == 1 ? k_D : safe_downcast<int, int64_t>(kernel_size[2]);

    // NB: stride default is not expressible as an integer constant, so we accept
    // empty stride for this case
    const int d_D = stride.empty() ? k_D : safe_downcast<int, int64_t>(stride[0]);
    const int d_H = stride.empty() ? k_H :
                    stride.size() == 1 ? d_D :
                                         safe_downcast<int, int64_t>(stride[1]);
    const int d_W = stride.empty() ? k_W :
                    stride.size() == 1 ? d_D :
                                         safe_downcast<int, int64_t>(stride[2]);

    const int pad_D = safe_downcast<int, int64_t>(padding[0]);
    const int pad_H = padding.size() == 1 ? pad_D : safe_downcast<int, int64_t>(padding[1]);
    const int pad_W = padding.size() == 1 ? pad_D : safe_downcast<int, int64_t>(padding[2]);

    const int dilation_D = safe_downcast<int, int64_t>(dilation[0]);
    const int dilation_H = dilation.size() == 1 ? dilation_D : safe_downcast<int, int64_t>(dilation[1]);
    const int dilation_W = dilation.size() == 1 ? dilation_D : safe_downcast<int, int64_t>(dilation[2]);

    const int64_t n_batch = x.ndimension() == 5 ? x.size(-5) : 1;
    const int64_t n_slices = x.size(-4);
    const int64_t input_time = x.size(-3);
    const int64_t input_height = x.size(-2);
    const int64_t input_width = x.size(-1);

    const int64_t output_time =
                  pooling_output_shape<int64_t>(input_time, k_D, pad_D, d_D, dilation_D, ceil_mode);
    const int64_t output_height =
                  pooling_output_shape<int64_t>(input_height, k_H, pad_H, d_H, dilation_H, ceil_mode);
    const int64_t output_width =
                  pooling_output_shape<int64_t>(input_width, k_W, pad_W, d_W, dilation_W, ceil_mode);

    pool3d_shape_check(x, n_slices,
                       k_D, k_H, k_W,
                       d_D, d_H, d_W,
                       pad_D, pad_H, pad_W,
                       dilation_D, dilation_H, dilation_W,
                       input_time, input_height, input_width,
                       output_time, output_height, output_width);

    std::vector<int64_t> output_size = {n_batch, n_slices, output_time, output_height, output_width};

    at::Tensor output = torch::empty(output_size, x.options());
    at::Tensor indicesOut = torch::empty(output_size, x.options().dtype(torch::kInt32));

    EXEC_NPU_CMD(aclnnMaxPool3dWithArgmax, x, kernel_size,
                 stride, padding, dilation, ceil_mode, output, indicesOut);

    return std::tuple<at::Tensor, at::Tensor>(output, indicesOut);
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("max_pool3d_with_argmax_v2", &max_pool3d_with_argmax_v2, "");
}