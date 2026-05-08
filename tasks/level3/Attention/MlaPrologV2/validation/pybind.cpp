#include <algorithm>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/* 与 acl_base_rt.h 一致，避免直接 include CANN acl 与 torch_npu 内置 acl 冲突 */
static constexpr int64_t kAclFormatNd = 2;
static constexpr int64_t kAclFormatFractalNz = 29;

namespace {

void copy_cstr(char *dst, size_t dst_sz, const std::string &s)
{
    if (dst_sz == 0) {
        return;
    }
    std::memset(dst, 0, dst_sz);
    std::memcpy(dst, s.c_str(), std::min(s.size(), dst_sz - 1));
}

/**
 * bench 自带 ConvertType(at::Tensor) 对 2D 张量恒用 ACL_FORMAT_ND，无法表达 FRACTAL_NZ。
 * 此处按物理存储构造 aclTensor，与 torch_npu empty_with_format(29) / npu_format_cast 一致。
 */
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
    at::Tensor token_x,
    at::Tensor weight_dq,
    at::Tensor weight_uq_qr,
    at::Tensor weight_uk,
    at::Tensor weight_dkv_kr,
    at::Tensor rmsnorm_gamma_cq,
    at::Tensor rmsnorm_gamma_ckv,
    at::Tensor rope_sin,
    at::Tensor rope_cos,
    at::Tensor cache_index,
    at::Tensor kv_cache,
    at::Tensor kr_cache,
    double rmsnorm_epsilon_cq,
    double rmsnorm_epsilon_ckv,
    std::string cache_mode,
    at::Tensor query_out,
    at::Tensor query_rope_out,
    at::Tensor dequant_scale_q_nope_out)
{
    char cm_storage[64];
    copy_cstr(cm_storage, sizeof(cm_storage), cache_mode);
    char *cm_ptr = cm_storage;

    aclTensor *a_token = AtTensorToAcl(token_x, kAclFormatNd);
    aclTensor *a_wdq = AtTensorToAcl(weight_dq, kAclFormatFractalNz);
    aclTensor *a_wuq = AtTensorToAcl(weight_uq_qr, kAclFormatFractalNz);
    aclTensor *a_wuk = AtTensorToAcl(weight_uk, kAclFormatNd);
    aclTensor *a_wdkv = AtTensorToAcl(weight_dkv_kr, kAclFormatFractalNz);
    aclTensor *a_gcq = AtTensorToAcl(rmsnorm_gamma_cq, kAclFormatNd);
    aclTensor *a_gckv = AtTensorToAcl(rmsnorm_gamma_ckv, kAclFormatNd);
    aclTensor *a_sin = AtTensorToAcl(rope_sin, kAclFormatNd);
    aclTensor *a_cos = AtTensorToAcl(rope_cos, kAclFormatNd);
    aclTensor *a_idx = AtTensorToAcl(cache_index, kAclFormatNd);
    aclTensor *a_kvc = AtTensorToAcl(kv_cache, kAclFormatNd);
    aclTensor *a_krc = AtTensorToAcl(kr_cache, kAclFormatNd);
    aclTensor *a_qo = AtTensorToAcl(query_out, kAclFormatNd);
    aclTensor *a_qr = AtTensorToAcl(query_rope_out, kAclFormatNd);
    aclTensor *a_dqsnope = AtTensorToAcl(dequant_scale_q_nope_out, kAclFormatNd);

    auto cleanup = [&]() {
        Release(a_token);
        Release(a_wdq);
        Release(a_wuq);
        Release(a_wuk);
        Release(a_wdkv);
        Release(a_gcq);
        Release(a_gckv);
        Release(a_sin);
        Release(a_cos);
        Release(a_idx);
        Release(a_kvc);
        Release(a_krc);
        Release(a_qo);
        Release(a_qr);
        Release(a_dqsnope);
    };

    LoadCustomOpTilingLib();
    void *getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnMlaPrologV2WeightNzGetWorkspaceSize");
    void *opApiFuncAddr = GetOpApiFuncAddr("aclnnMlaPrologV2WeightNz");
    void *initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    void *unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnMlaPrologV2WeightNz not in custom opapi");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor **executor_addr = &executor;
    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    /* BF16 非量化：可选 dequant/quant/smooth 传空指针（与 aclnn_mla_prolog_v2_weight_nz.h 一致） */
    auto converted_params = std::make_tuple(
        a_token, a_wdq, a_wuq, a_wuk, a_wdkv, a_gcq, a_gckv, a_sin, a_cos, a_idx, a_kvc, a_krc, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, cm_ptr, a_qo, a_qr,
        a_dqsnope, workspace_size_addr, executor_addr);

    typedef int32_t (*GetWsFunc)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, aclTensor *,
        aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, double, double, char *, const aclTensor *, const aclTensor *,
        const aclTensor *, uint64_t *, aclOpExecutor **);
    auto getWs = reinterpret_cast<GetWsFunc>(getWorkspaceSizeFuncAddr);
    int32_t ws_st = getWs(
        std::get<0>(converted_params), std::get<1>(converted_params), std::get<2>(converted_params),
        std::get<3>(converted_params), std::get<4>(converted_params), std::get<5>(converted_params),
        std::get<6>(converted_params), std::get<7>(converted_params), std::get<8>(converted_params),
        std::get<9>(converted_params), std::get<10>(converted_params), std::get<11>(converted_params),
        std::get<12>(converted_params), std::get<13>(converted_params), std::get<14>(converted_params),
        std::get<15>(converted_params), std::get<16>(converted_params), std::get<17>(converted_params),
        std::get<18>(converted_params), std::get<19>(converted_params), std::get<20>(converted_params),
        std::get<21>(converted_params), std::get<22>(converted_params), std::get<23>(converted_params),
        std::get<24>(converted_params), std::get<25>(converted_params), std::get<26>(converted_params));
    TORCH_CHECK(ws_st == 0, "call aclnnMlaPrologV2WeightNzGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());

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
        TORCH_CHECK(api_ret == 0, "call aclnnMlaPrologV2WeightNz failed, detail:", aclGetRecentErrMsg());
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnMlaPrologV2WeightNz");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();

    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    cleanup();

    return {query_out, query_rope_out, kv_cache, kr_cache, dequant_scale_q_nope_out};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
