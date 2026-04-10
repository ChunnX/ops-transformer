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

/**
 * 参数顺序需与 msopgen 生成的 aclnnRopeQuantKvcacheGetWorkspaceSize 一致；
 * 编译后请核对 build_out/autogen/aclnn_rope_quant_kvcache.h。
 */
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor qkv,
    at::Tensor cos,
    at::Tensor sin,
    at::Tensor quant_scale,
    at::Tensor quant_offset,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor indices,
    std::vector<int64_t> size_splits,
    std::string layout,
    bool kv_output,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v)
{
    aclIntArray *splits = ConvertType(at::IntArrayRef(size_splits));
    char ly_storage[256];
    copy_cstr(ly_storage, sizeof(ly_storage), layout);
    char *ly_ptr = ly_storage;

    EXEC_NPU_CMD(
        aclnnCustomOp,
        qkv,
        cos,
        sin,
        quant_scale,
        quant_offset,
        k_cache,
        v_cache,
        indices,
        splits,
        ly_ptr,
        kv_output,
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
