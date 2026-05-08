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
    at::Tensor attention_out,
    at::Tensor topk_indices_out,
    at::Tensor actual_seq_qlen,
    at::Tensor actual_cmp_seq_kvlen,
    at::Tensor actual_sel_seq_kvlen,
    double scale_value,
    int64_t head_num,
    std::string input_layout,
    int64_t sparse_mode,
    int64_t compress_block_size,
    int64_t compress_stride,
    int64_t select_block_size,
    int64_t select_block_count)
{
    char layout_storage[64];
    copy_cstr(layout_storage, sizeof(layout_storage), input_layout);
    char *layout_ptr = layout_storage;

    aclTensor *a_query = AtTensorToAcl(query, kAclFormatNd);
    aclTensor *a_key = AtTensorToAcl(key, kAclFormatNd);
    aclTensor *a_val = AtTensorToAcl(value, kAclFormatNd);
    aclTensor *a_smax = AtTensorToAcl(softmax_max, kAclFormatNd);
    aclTensor *a_ssum = AtTensorToAcl(softmax_sum, kAclFormatNd);
    aclTensor *a_out = AtTensorToAcl(attention_out, kAclFormatNd);
    aclTensor *a_topk = AtTensorToAcl(topk_indices_out, kAclFormatNd);

    at::Tensor q_cpu = actual_seq_qlen.cpu().contiguous();
    at::Tensor k_cpu = actual_cmp_seq_kvlen.cpu().contiguous();
    at::Tensor s_cpu = actual_sel_seq_kvlen.cpu().contiguous();
    TORCH_CHECK(q_cpu.scalar_type() == at::kLong, "actual_seq_qlen must be int64");
    TORCH_CHECK(k_cpu.scalar_type() == at::kLong, "actual_cmp_seq_kvlen must be int64");
    TORCH_CHECK(s_cpu.scalar_type() == at::kLong, "actual_sel_seq_kvlen must be int64");

    static const auto aclCreateIntArrayFn = GET_OP_API_FUNC(aclCreateIntArray);
    TORCH_CHECK(aclCreateIntArrayFn != nullptr, "aclCreateIntArray unavailable");
    aclIntArray *a_qlen =
        aclCreateIntArrayFn(q_cpu.data_ptr<int64_t>(), static_cast<uint64_t>(q_cpu.numel()));
    aclIntArray *a_cmpkv =
        aclCreateIntArrayFn(k_cpu.data_ptr<int64_t>(), static_cast<uint64_t>(k_cpu.numel()));
    aclIntArray *a_selkv =
        aclCreateIntArrayFn(s_cpu.data_ptr<int64_t>(), static_cast<uint64_t>(s_cpu.numel()));

    auto cleanup = [&]() {
        Release(a_query);
        Release(a_key);
        Release(a_val);
        Release(a_smax);
        Release(a_ssum);
        Release(a_out);
        Release(a_topk);
        Release(a_qlen);
        Release(a_cmpkv);
        Release(a_selkv);
    };

    LoadCustomOpTilingLib();
    void *getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnNsaCompressAttentionGetWorkspaceSize");
    void *opApiFuncAddr = GetOpApiFuncAddr("aclnnNsaCompressAttention");
    void *initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    void *unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnNsaCompressAttention not in custom opapi");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    typedef int32_t (*GetWsFunc)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclIntArray *, const aclIntArray *, const aclIntArray *, double, int64_t, char *, int64_t, int64_t, int64_t,
        int64_t, int64_t, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *,
        aclOpExecutor **);
    auto getWs = reinterpret_cast<GetWsFunc>(getWorkspaceSizeFuncAddr);
    int32_t ws_st = getWs(
        a_query, a_key, a_val, nullptr, nullptr, a_qlen, a_cmpkv, a_selkv, scale_value, head_num, layout_ptr,
        sparse_mode, compress_block_size, compress_stride, select_block_size, select_block_count, a_smax, a_ssum,
        a_out, a_topk, &workspace_size, &executor);
    TORCH_CHECK(ws_st == 0, "call aclnnNsaCompressAttentionGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

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
        TORCH_CHECK(api_ret == 0, "call aclnnNsaCompressAttention failed, detail:", aclGetRecentErrMsg());
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnNsaCompressAttention");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();

    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    cleanup();

    return {softmax_max, softmax_sum, attention_out, topk_indices_out};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
