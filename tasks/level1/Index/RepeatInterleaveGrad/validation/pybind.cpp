#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor repeat_interleave_grad_custom(at::Tensor &y_grad, at::Tensor &repeats, int64_t axis)
{
    auto y_shape = y_grad.sizes().vec();
    int64_t dim = axis;
    if (dim < 0) dim += y_shape.size();

    int64_t out_dim_size = 0;
    
    if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.size(0) == 1)) {
        int64_t r_val = 0;
        if (repeats.scalar_type() == at::kInt) {
            r_val = repeats.item<int32_t>();
        } else {
            r_val = repeats.item<int64_t>();
        }
        
        if (r_val != 0) {
            out_dim_size = y_shape[dim] / r_val;
        } else {
             out_dim_size = 0;
        }
    } else {
        out_dim_size = repeats.size(0);
    }

    auto x_grad_shape = y_shape;
    x_grad_shape[dim] = out_dim_size;

    at::Tensor result = torch::empty(x_grad_shape, y_grad.options());

    EXEC_NPU_CMD(aclnnRepeatInterleaveGrad, y_grad, repeats, axis, result);
    
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("repeat_interleave_grad", &repeat_interleave_grad_custom, "RepeatInterleaveGrad custom op");
}