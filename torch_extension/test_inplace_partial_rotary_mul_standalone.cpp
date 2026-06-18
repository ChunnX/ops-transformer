#include <torch/extension.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <dlfcn.h>
#include <string>

namespace {

typedef struct aclOpExecutor aclOpExecutor;

const int BSND_DIM_NUM = 4;
static const std::unordered_map<std::string, int> mode_map = {
    {"half", 0},
    {"interleave", 1},
    {"quarter", 2},
    {"interleave-half", 3}
};

typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

inline void *GetOpApiFuncAddr(const char *apiName)
{
    void *funcAddr = nullptr;
    const char *ascend_path = std::getenv("ASCEND_HOME_PATH");
    if (ascend_path != nullptr) {
        std::string cust_path = std::string(ascend_path) + "/vendors/custom_transformer/op_api/lib/libcust_opapi.so";
        auto custHandler = dlopen(cust_path.c_str(), RTLD_LAZY);
        if (custHandler != nullptr) {
            funcAddr = dlsym(custHandler, apiName);
        }
    }
    if (funcAddr != nullptr) {
        return funcAddr;
    }
    auto custHandler = dlopen("libcust_opapi.so", RTLD_LAZY);
    if (custHandler != nullptr) {
        funcAddr = dlsym(custHandler, apiName);
    }
    if (funcAddr != nullptr) {
        return funcAddr;
    }
    static const auto opApiHandler = dlopen("libopapi.so", RTLD_LAZY);
    if (opApiHandler != nullptr) {
        funcAddr = dlsym(opApiHandler, apiName);
    }
    return funcAddr;
}

#define GET_OP_API_FUNC(apiName) \
  reinterpret_cast<OpApiFunc>(GetOpApiFuncAddr(#apiName))

#define GET_WORKSPACE_SIZE_FUNC(apiName) \
  reinterpret_cast<OpApiFunc>(GetOpApiFuncAddr(#apiName "GetWorkspaceSize"))

void inplace_partial_rotary_mul(at::Tensor &x, const at::Tensor &r1,
                                 const at::Tensor &r2, const std::string &rotary_mode,
                                 const c10::IntArrayRef &partial_slice)
{
    auto it = mode_map.find(rotary_mode);
    TORCH_CHECK(it != mode_map.end(),
        "rotary_mode must be one of 'half', 'interleave', 'quarter', 'interleave-half', got '", rotary_mode, "'.");
    int64_t mode_int = it->second;

    TORCH_CHECK(x.dim() == BSND_DIM_NUM,
        "Input tensor x's dim num should be 4, actual ", x.dim(), ".");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor **executor_addr = &executor;

    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnInplacePartialRotaryMulGetWorkspaceSize");
    static const auto opApiFuncAddr = GetOpApiFuncAddr("aclnnInplacePartialRotaryMul");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        "aclnnInplacePartialRotaryMul or aclnnInplacePartialRotaryMulGetWorkspaceSize not found.");

    static auto getWorkspaceSizeFunc = GET_WORKSPACE_SIZE_FUNC(aclnnInplacePartialRotaryMul);

    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnInplacePartialRotaryMul");

    cmd.SetCustomHandler([workspace_size, acl_stream, executor]() -> int {
        typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        void *workspace_addr = nullptr;
        at::Tensor workspace_tensor;
        if (workspace_size != 0) {
            at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
            workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
        }
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        TORCH_CHECK(api_ret == 0, "call aclnnInplacePartialRotaryMul failed");
        return api_ret;
    });

    cmd.Run();
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("inplace_partial_rotary_mul", &inplace_partial_rotary_mul, "inplace_partial_rotary_mul");
}
