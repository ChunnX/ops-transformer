#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>

#include "pytorch_npu_helper.hpp"

namespace {

/**
 * pytorch_npu_helper 的 kATenScalarTypeToAclDataTypePair 未覆盖 UInt64，Long 会映射为 ACL_INT64；
 * 本算子 aclnn 白名单要求 out 为 DT_UINT64（precision-debug §1）。对 int64/uint64 存储用 ACL_UINT64 建 aclTensor。
 */
inline aclTensor *ConvertTensorToAclUInt64Pack(const at::Tensor &at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }
    TORCH_CHECK(at_tensor.defined(), "out tensor undefined");
    TORCH_CHECK(
        at_tensor.scalar_type() == at::ScalarType::Long ||
            at_tensor.scalar_type() == at::ScalarType::UInt64,
        "TransQuantParamV2 out must be int64 or uint64, got ",
        c10::toString(at_tensor.scalar_type()));
    const aclDataType acl_data_type = ACL_UINT64;
    c10::SmallVector<int64_t, 5> storageDims;
    const auto itemsize = at_tensor.itemsize();
    TORCH_CHECK(itemsize != 0, "tensor item size cannot be zero");
    storageDims.push_back(static_cast<int64_t>(at_tensor.storage().nbytes() / itemsize));

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    switch (dimNum) {
        case 3:
            format = ACL_FORMAT_NCL;
            break;
        case 4:
            format = ACL_FORMAT_NCHW;
            break;
        case 5:
            format = ACL_FORMAT_NCDHW;
            break;
        default:
            format = ACL_FORMAT_ND;
    }

    auto *acl_tensor = aclCreateTensor(
        at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type, at_tensor.strides().data(),
        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

} // namespace

/**
 * 与 build_out/autogen/aclnn_trans_quant_param_v2.h 中 GetWorkspaceSize 一致（§7.1）：
 * scale, offsetOptional, roundMode, out。
 * out 须以 ACL_UINT64 传入，见 ConvertTensorToAclUInt64Pack。
 */
std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor &scale,
    c10::optional<at::Tensor> offset)
{
    at::Tensor out = at::empty(scale.sizes(), at::TensorOptions().dtype(at::kLong).device(scale.device()));
    const int64_t roundMode = 0;

    do {
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnCustomOpGetWorkspaceSize");
        static const auto opApiFuncAddr = GetOpApiFuncAddr("aclnnCustomOp");
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
        TORCH_CHECK(
            getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, "aclnnCustomOp",
            " or aclnnCustomOpGetWorkspaceSize not in ", GetOpApiLibName());
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        uint64_t workspace_size = 0;
        uint64_t *workspace_size_addr = &workspace_size;
        aclOpExecutor *executor = nullptr;
        aclOpExecutor **executor_addr = &executor;
        using InitHugeMemThreadLocal = int32_t (*)(void *, bool);
        using UnInitHugeMemThreadLocal = void (*)(void *, bool);
        using ReleaseHugeMem = void (*)(void *, bool);
        auto initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
        auto unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
        if (initMemFunc) {
            initMemFunc(nullptr, false);
        }
        auto acl_scale = ConvertType(scale);
        auto acl_offset = ConvertType(offset);
        auto acl_out = ConvertTensorToAclUInt64Pack(out);
        auto converted_params =
            std::make_tuple(acl_scale, acl_offset, roundMode, acl_out, workspace_size_addr, executor_addr);
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);
        TORCH_CHECK(workspace_status == 0, "call aclnnCustomOp failed, detail:", aclGetRecentErrMsg());
        void *workspace_addr = nullptr;
        at::Tensor workspace_tensor;
        if (workspace_size != 0) {
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
            workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
        }
        auto acl_call = [converted_params, workspace_addr, workspace_size, workspace_tensor, acl_stream, executor,
                         opApiFuncAddr, releaseMemAddr]() -> int32_t {
            typedef int32_t (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
            TORCH_CHECK(api_ret == 0, "call aclnnCustomOp failed, detail:", aclGetRecentErrMsg());
            ReleaseConvertTypes(converted_params);
            auto releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
            if (releaseMemFunc) {
                releaseMemFunc(nullptr, false);
            }
            return api_ret;
        };
        at_npu::native::OpCommand cmd;
        cmd.Name("aclnnCustomOp");
        cmd.SetCustomHandler(acl_call);
        cmd.Run();
        if (unInitMemFunc) {
            unInitMemFunc(nullptr, false);
        }
    } while (false);

    return {out};
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
