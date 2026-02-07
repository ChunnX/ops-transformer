#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
* register forward implementation for NPU device
*/

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> custom_pybind_api(
    at::Tensor &weight, 
    at::Tensor &indices, 
    at::Tensor &offsets,
    c10::optional<at::Tensor> per_sample_weights,
    std::string mode,
    bool scale_grad_by_freq,
    bool sparse,
    bool include_last_offset,
    int64_t padding_idx)
{
    // Get output shapes
    int64_t num_bags = offsets.size(0);
    if (include_last_offset) {
        num_bags -= 1;
    }
    int64_t embedding_dim = weight.size(1);
    int64_t num_indices = indices.numel();
    
    // Alloc output memory
    at::Tensor y = at::empty({num_bags, embedding_dim}, weight.options());
    at::Tensor offset2bag = at::empty({num_indices}, indices.options());
    at::Tensor bag_size = at::empty({num_bags}, indices.options());
    at::Tensor max_indices;
    
    if (mode == "max") {
        max_indices = at::empty({num_bags, embedding_dim}, indices.options());
    } else {
        max_indices = at::empty({num_bags}, indices.options());
    }
    
    // Prepare per_sample_weights tensor (optional)
    at::Tensor per_sample_weights_tensor;
    if (per_sample_weights.has_value()) {
        per_sample_weights_tensor = per_sample_weights.value();
    } else {
        per_sample_weights_tensor = at::empty({0}, weight.options());
    }
    
    // Call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnEmbeddingBag, weight, indices, offsets, per_sample_weights_tensor, 
                 mode, scale_grad_by_freq, sparse, include_last_offset, padding_idx,
                 y, offset2bag, bag_size, max_indices);
    
    return std::make_tuple(y, offset2bag, bag_size, max_indices);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "EmbeddingBag custom operation",
          py::arg("weight"),
          py::arg("indices"),
          py::arg("offsets"),
          py::arg("per_sample_weights") = py::none(),
          py::arg("mode") = "mean",
          py::arg("scale_grad_by_freq") = false,
          py::arg("sparse") = false,
          py::arg("include_last_offset") = false,
          py::arg("padding_idx") = -1);
}