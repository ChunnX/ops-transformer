#include <algorithm>
#include <cstring>
#include <string>

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
    at::Tensor input,
    at::Tensor weight,
    at::Tensor slot_mapping,
    at::Tensor act_seq_len,
    at::Tensor block_table,
    std::string layout,
    int64_t compress_block_size,
    int64_t compress_stride,
    int64_t act_seq_len_type,
    int64_t page_block_size,
    at::Tensor output_cache)
{
    char layout_storage[64];
    copy_cstr(layout_storage, sizeof(layout_storage), layout);
    char *layout_ptr = layout_storage;

    aclTensor *a_in = AtTensorToAcl(input, kAclFormatNd);
    aclTensor *a_w = AtTensorToAcl(weight, kAclFormatNd);
    aclTensor *a_slot = AtTensorToAcl(slot_mapping, kAclFormatNd);
    aclTensor *a_out = AtTensorToAcl(output_cache, kAclFormatNd);
    aclTensor *a_bt = AtTensorToAcl(block_table, kAclFormatNd);

    at::Tensor act_cpu = act_seq_len.cpu().contiguous();
    TORCH_CHECK(act_cpu.scalar_type() == at::kLong, "act_seq_len must be int64");
    static const auto aclCreateIntArrayFn = GET_OP_API_FUNC(aclCreateIntArray);
    TORCH_CHECK(aclCreateIntArrayFn != nullptr, "aclCreateIntArray unavailable");
    aclIntArray *a_act = aclCreateIntArrayFn(act_cpu.data_ptr<int64_t>(), static_cast<uint64_t>(act_cpu.numel()));

    auto cleanup = [&]() {
        Release(a_in);
        Release(a_w);
        Release(a_slot);
        Release(a_out);
        Release(a_bt);
        Release(a_act);
    };

    LoadCustomOpTilingLib();
    void *getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnNsaCompressWithCacheGetWorkspaceSize");
    void *opApiFuncAddr = GetOpApiFuncAddr("aclnnNsaCompressWithCache");
    void *initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    void *unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnNsaCompressWithCache not in custom opapi");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    typedef int32_t (*GetWsFunc)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclIntArray *, const aclTensor *, char *,
        int64_t, int64_t, int64_t, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
    auto getWs = reinterpret_cast<GetWsFunc>(getWorkspaceSizeFuncAddr);
    int32_t ws_st = getWs(
        a_in, a_w, a_slot, a_act, a_bt, layout_ptr, compress_block_size, compress_stride, act_seq_len_type,
        page_block_size, a_out, &workspace_size, &executor);
    TORCH_CHECK(ws_st == 0, "call aclnnNsaCompressWithCacheGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

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
        TORCH_CHECK(api_ret == 0, "call aclnnNsaCompressWithCache failed, detail:", aclGetRecentErrMsg());
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnNsaCompressWithCache");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();

    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    cleanup();

    return output_cache;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
