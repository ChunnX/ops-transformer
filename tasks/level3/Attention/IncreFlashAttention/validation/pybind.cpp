#include <algorithm>
#include <cstring>
#include <string>
#include <tuple>
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

at::Tensor custom_pybind_api(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    at::Tensor attention_out,
    at::Tensor actual_seq_lengths,
    int64_t num_heads,
    double scale_value,
    std::string input_layout,
    int64_t num_kv_heads)
{
    char layout_storage[64];
    copy_cstr(layout_storage, sizeof(layout_storage), input_layout);
    char *layout_ptr = layout_storage;

    aclTensor *a_query = AtTensorToAcl(query, kAclFormatNd);
    aclTensor *a_key = AtTensorToAcl(key, kAclFormatNd);
    aclTensor *a_val = AtTensorToAcl(value, kAclFormatNd);
    aclTensor *a_out = AtTensorToAcl(attention_out, kAclFormatNd);

    at::Tensor actual_cpu = actual_seq_lengths.cpu().contiguous();
    TORCH_CHECK(actual_cpu.scalar_type() == at::kLong, "actual_seq_lengths must be int64");
    static const auto aclCreateIntArrayFn = GET_OP_API_FUNC(aclCreateIntArray);
    TORCH_CHECK(aclCreateIntArrayFn != nullptr, "aclCreateIntArray unavailable");
    aclIntArray *a_actual =
        aclCreateIntArrayFn(actual_cpu.data_ptr<int64_t>(), static_cast<uint64_t>(actual_cpu.numel()));

    const aclTensor *key_ptrs[] = {a_key};
    const aclTensor *val_ptrs[] = {a_val};
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    TORCH_CHECK(aclCreateTensorList != nullptr, "aclCreateTensorList unavailable");
    aclTensorList *key_list = aclCreateTensorList(key_ptrs, 1);
    aclTensorList *val_list = aclCreateTensorList(val_ptrs, 1);

    auto cleanup = [&]() {
        Release(a_query);
        Release(a_key);
        Release(a_val);
        Release(a_out);
        Release(a_actual);
        Release(key_list);
        Release(val_list);
    };

    LoadCustomOpTilingLib();
    void *getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnIncreFlashAttentionGetWorkspaceSize");
    void *opApiFuncAddr = GetOpApiFuncAddr("aclnnIncreFlashAttention");
    void *initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    void *unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnIncreFlashAttention not in custom opapi");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    /* 与 build_out/autogen/aclnn_incre_flash_attention.cpp 中符号一致（CANN 9.0 扩展了可选输入与属性） */
    typedef int32_t (*GetWsFunc)(
        const aclTensor *, const aclTensorList *, const aclTensorList *, const aclTensor *, const aclTensor *,
        const aclIntArray *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        int64_t, double, char *, int64_t, int64_t, int64_t, const aclTensor *, uint64_t *, aclOpExecutor **);
    auto getWs = reinterpret_cast<GetWsFunc>(getWorkspaceSizeFuncAddr);
    int64_t block_size = 0;
    int64_t inner_precise = 0;
    int32_t ws_st = getWs(
        a_query, key_list, val_list, nullptr, nullptr, a_actual, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, num_heads, scale_value, layout_ptr, num_kv_heads, block_size, inner_precise, a_out,
        &workspace_size, &executor);
    TORCH_CHECK(ws_st == 0, "call aclnnIncreFlashAttentionGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

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
        TORCH_CHECK(api_ret == 0, "call aclnnIncreFlashAttention failed, detail:", aclGetRecentErrMsg());
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnIncreFlashAttention");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();

    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    cleanup();

    return attention_out;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
