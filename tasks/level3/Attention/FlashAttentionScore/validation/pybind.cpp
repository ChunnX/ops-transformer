#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

static constexpr int64_t kAclFormatNd = 2;

namespace {

void copy_cstr(char *dst, size_t dst_sz, const std::string &s)
{
    if (dst_sz == 0) {
        return;
    }
    std::memset(dst, 0, dst_sz);
    std::memcpy(dst, s.c_str(), std::min(s.size(), dst_sz - 1));
}

static aclTensor *AtTensorToAcl(const at::Tensor &at_tensor, int64_t format)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }
    if (!at_tensor.defined()) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(acl_data_type != ACL_DT_UNDEFINED,
                std::string(c10::toString(scalar_data_type)) + " has not been supported")
    auto itemsize = at_tensor.itemsize();
    TORCH_CHECK(itemsize > 0, "When AtTensorToAcl, tensor item size cannot be zero.");
    c10::SmallVector<int64_t, 8> storageDims;
    storageDims.push_back(at_tensor.storage().nbytes() / itemsize);
    return aclCreateTensor(
        at_tensor.sizes().data(), static_cast<int64_t>(at_tensor.sizes().size()), acl_data_type,
        at_tensor.strides().data(), at_tensor.storage_offset(), static_cast<aclFormat>(format), storageDims.data(),
        static_cast<int64_t>(storageDims.size()), const_cast<void *>(at_tensor.storage().data()));
}

} // namespace

std::vector<at::Tensor> custom_pybind_api(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor softmax_max,
    at::Tensor softmax_sum,
    at::Tensor softmax_out,
    at::Tensor attention_out,
    double scale_value,
    int64_t head_num,
    double keep_prob,
    int64_t pre_tockens,
    int64_t next_tockens,
    int64_t inner_precise,
    int64_t sparse_mode)
{
    char layout_storage[64];
    copy_cstr(layout_storage, sizeof(layout_storage), std::string("BNSD"));
    char *layout_ptr = layout_storage;

    aclTensor *a_query = AtTensorToAcl(query, kAclFormatNd);
    aclTensor *a_key = AtTensorToAcl(key, kAclFormatNd);
    aclTensor *a_val = AtTensorToAcl(value, kAclFormatNd);
    aclTensor *a_smax = AtTensorToAcl(softmax_max, kAclFormatNd);
    aclTensor *a_ssum = AtTensorToAcl(softmax_sum, kAclFormatNd);
    aclTensor *a_sout = AtTensorToAcl(softmax_out, kAclFormatNd);
    aclTensor *a_out = AtTensorToAcl(attention_out, kAclFormatNd);

    auto cleanup = [&]() {
        Release(a_query);
        Release(a_key);
        Release(a_val);
        Release(a_smax);
        Release(a_ssum);
        Release(a_sout);
        Release(a_out);
    };

    LoadCustomOpTilingLib();
    void *getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnFlashAttentionScoreGetWorkspaceSize");
    void *opApiFuncAddr = GetOpApiFuncAddr("aclnnFlashAttentionScore");
    void *initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    void *unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnFlashAttentionScore not in custom opapi");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    char softmax_out_layout[64];
    copy_cstr(softmax_out_layout, sizeof(softmax_out_layout), std::string(""));

    typedef int32_t (*GetWsFunc)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclIntArray *, const aclIntArray *, const aclIntArray *,
        const aclIntArray *, const aclIntArray *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, double, double, int64_t, int64_t, int64_t, char *,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, char *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
    auto getWs = reinterpret_cast<GetWsFunc>(getWorkspaceSizeFuncAddr);
    int32_t ws_st = getWs(
        a_query, a_key, a_val, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, scale_value, keep_prob, pre_tockens, next_tockens, head_num,
        layout_ptr, inner_precise, sparse_mode,
        static_cast<int64_t>(1),  // pse_type default
        static_cast<int64_t>(0),  // seed
        static_cast<int64_t>(0),  // offset
        static_cast<int64_t>(0),  // out_dtype
        softmax_out_layout, a_smax, a_ssum, a_sout, a_out, &workspace_size, &executor);
    TORCH_CHECK(ws_st == 0, "call aclnnFlashAttentionScoreGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

    at::Tensor workspace_holder;
    void *workspace_addr = nullptr;
    if (workspace_size != 0) {
        at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspace_holder =
            at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));
        workspace_addr = const_cast<void *>(workspace_holder.storage().data());
    }

    void *op_api_fn = opApiFuncAddr;
    auto acl_call = [workspace_addr, workspace_size, acl_stream, executor, op_api_fn]() -> int32_t {
        typedef int32_t (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(op_api_fn);
        int32_t api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        TORCH_CHECK(api_ret == 0, "call aclnnFlashAttentionScore failed, detail:", aclGetRecentErrMsg());
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnFlashAttentionScore");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();

    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    cleanup();

    return {softmax_max, softmax_sum, softmax_out, attention_out};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
