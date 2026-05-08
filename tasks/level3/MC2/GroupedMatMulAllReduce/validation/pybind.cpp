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

static aclTensorList *BuildTensorList(const std::vector<at::Tensor> &ts, int64_t format,
                                      std::vector<aclTensor *> &owned)
{
    owned.clear();
    owned.reserve(ts.size());
    std::vector<const aclTensor *> ptrs;
    ptrs.reserve(ts.size());
    for (const auto &t : ts) {
        aclTensor *a = AtTensorToAcl(t, format);
        TORCH_CHECK(a != nullptr, "AtTensorToAcl failed for grouped matmul tensor");
        owned.push_back(a);
        ptrs.push_back(a);
    }
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    TORCH_CHECK(aclCreateTensorList != nullptr, "aclCreateTensorList unavailable");
    return aclCreateTensorList(ptrs.data(), static_cast<uint64_t>(ptrs.size()));
}

} // namespace

/**
 * 与 autogen aclnn_grouped_mat_mul_all_reduce 一致；streamMode 在 CANN 侧校验须为 1，commTurn 须为 0，reduceOp 须为 sum。
 */
std::vector<at::Tensor> grouped_mat_mul_all_reduce(
    const std::vector<at::Tensor> &x_list,
    const std::vector<at::Tensor> &weight_list,
    const std::vector<at::Tensor> &bias_list,
    c10::optional<at::Tensor> group_list_cpu,
    int64_t split_item,
    const std::string &group,
    const std::string &reduce_op,
    int64_t comm_turn,
    const std::vector<at::Tensor> &y_list)
{
    TORCH_CHECK(!x_list.empty(), "x_list empty");
    TORCH_CHECK(x_list.size() == weight_list.size(), "x/weight list size mismatch");
    TORCH_CHECK(x_list.size() == y_list.size(), "x/y list size mismatch");
    TORCH_CHECK(bias_list.size() == x_list.size(), "bias list size mismatch");

    char group_buf[256];
    char reduce_buf[32];
    copy_cstr(group_buf, sizeof(group_buf), group);
    copy_cstr(reduce_buf, sizeof(reduce_buf), reduce_op.empty() ? "sum" : reduce_op);
    char *group_ptr = group_buf;
    char *reduce_ptr = reduce_buf;

    std::vector<aclTensor *> own_x, own_w, own_b, own_y;
    aclTensorList *lx = BuildTensorList(x_list, kAclFormatNd, own_x);
    aclTensorList *lw = BuildTensorList(weight_list, kAclFormatNd, own_w);
    aclTensorList *ly = BuildTensorList(y_list, kAclFormatNd, own_y);

    aclTensorList *lb = BuildTensorList(bias_list, kAclFormatNd, own_b);

    aclIntArray *group_arr = nullptr;
    if (group_list_cpu.has_value() && group_list_cpu->defined() && group_list_cpu->numel() > 0) {
        at::Tensor gl_cpu = group_list_cpu->cpu().contiguous();
        TORCH_CHECK(gl_cpu.scalar_type() == at::kLong, "group_list must be int64");
        static const auto aclCreateIntArrayFn = GET_OP_API_FUNC(aclCreateIntArray);
        TORCH_CHECK(aclCreateIntArrayFn != nullptr, "aclCreateIntArray unavailable");
        group_arr = aclCreateIntArrayFn(gl_cpu.data_ptr<int64_t>(), static_cast<uint64_t>(gl_cpu.numel()));
    }

    auto cleanup_acl = [&]() {
        Release(lx);
        Release(lw);
        Release(ly);
        Release(lb);
        Release(group_arr);
        for (auto *p : own_x) {
            Release(p);
        }
        for (auto *p : own_w) {
            Release(p);
        }
        for (auto *p : own_y) {
            Release(p);
        }
        for (auto *p : own_b) {
            Release(p);
        }
    };

    LoadCustomOpTilingLib();
    void *getWsAddr = GetOpApiFuncAddr("aclnnGroupedMatMulAllReduceGetWorkspaceSize");
    void *opAddr = GetOpApiFuncAddr("aclnnGroupedMatMulAllReduce");
    void *initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    void *unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    TORCH_CHECK(getWsAddr != nullptr && opAddr != nullptr, "aclnnGroupedMatMulAllReduce not in custom opapi");

    typedef int32_t (*GetWsFunc)(const aclTensorList *, const aclTensorList *, const aclTensorList *,
                                 const aclIntArray *, int64_t, char *, char *, int64_t, const aclTensorList *,
                                 uint64_t *, aclOpExecutor **);
    auto getWs = reinterpret_cast<GetWsFunc>(getWsAddr);

    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    int32_t ws_st =
        getWs(lx, lw, lb, group_arr, split_item, group_ptr, reduce_ptr, comm_turn, ly, &workspace_size, &executor);
    TORCH_CHECK(ws_st == 0, "aclnnGroupedMatMulAllReduceGetWorkspaceSize failed: ", aclGetRecentErrMsg());

    at::Tensor workspace_holder;
    void *workspace_addr = nullptr;
    if (workspace_size != 0) {
        at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspace_holder = at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));
        workspace_addr = const_cast<void *>(workspace_holder.storage().data());
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    typedef int32_t (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opAddr);
    int32_t api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
    TORCH_CHECK(api_ret == 0, "aclnnGroupedMatMulAllReduce failed: ", aclGetRecentErrMsg());

    ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
    if (releaseMemFunc) {
        releaseMemFunc(nullptr, false);
    }
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }

    cleanup_acl();

    std::vector<at::Tensor> out;
    out.reserve(y_list.size());
    for (const auto &t : y_list) {
        out.push_back(t);
    }
    return out;
}

PYBIND11_MODULE(kernel_gen_ops, m)
{
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("grouped_mat_mul_all_reduce", &grouped_mat_mul_all_reduce, "");
}
