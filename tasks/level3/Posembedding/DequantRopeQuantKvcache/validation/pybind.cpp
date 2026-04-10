#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

namespace {

void copy_cstr(char *dst, size_t dst_sz, const std::string &s)
{
    if (dst_sz == 0) {
        return;
    }
    std::memset(dst, 0, dst_sz);
    std::memcpy(dst, s.c_str(), std::min(s.size(), dst_sz - 1));
}

} // namespace

std::vector<at::Tensor> custom_pybind_api(
    at::Tensor x,
    at::Tensor cos,
    at::Tensor sin,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor indices,
    at::Tensor scale_k,
    at::Tensor scale_v,
    c10::optional<at::Tensor> offset_k,
    c10::optional<at::Tensor> offset_v,
    c10::optional<at::Tensor> weight_scale,
    c10::optional<at::Tensor> activation_scale,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> size_splits,
    std::string quant_mode,
    std::string layout,
    bool kv_output,
    std::string cache_mode,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v)
{
    aclIntArray *splits = ConvertType(at::IntArrayRef(size_splits));
    char qm_storage[256];
    char ly_storage[256];
    char cm_storage[256];
    copy_cstr(qm_storage, sizeof(qm_storage), quant_mode);
    copy_cstr(ly_storage, sizeof(ly_storage), layout);
    copy_cstr(cm_storage, sizeof(cm_storage), cache_mode);
    char *qm_ptr = qm_storage;
    char *ly_ptr = ly_storage;
    char *cm_ptr = cm_storage;

    EXEC_NPU_CMD(
        aclnnCustomOp,
        x,
        cos,
        sin,
        k_cache,
        v_cache,
        indices,
        scale_k,
        scale_v,
        offset_k,
        offset_v,
        weight_scale,
        activation_scale,
        bias,
        splits,
        qm_ptr,
        ly_ptr,
        kv_output,
        cm_ptr,
        q,
        k,
        v);
    return {q, k, v, k_cache, v_cache};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
