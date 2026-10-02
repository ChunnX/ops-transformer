// SageAttention2 torch 扩展绑定：经 EXEC_NPU_CMD 两段式调用 aclnnSa2Attention2（参数校验与 python 层一致）
#include <dlfcn.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

using namespace pybind11::literals;

using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
// aclnn 两段式约定：第一段返回 aclnnStatus（int 枚举），第二段为固定四参入口
using aclnnStatus = int;

using aclCreateTensorFn = aclTensor *(*)(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                                         const int64_t *strides, int64_t offset, aclFormat format,
                                         const int64_t *storageDims, uint64_t storageDimsNum, void *tensorData);
using aclDestroyTensorFn = int (*)(const aclTensor *tensor);
using aclGetRecentErrMsgFn = const char *(*)();
using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

namespace {

constexpr int kMaxDimNum = 8;
constexpr int64_t kDimNum4 = 4;
constexpr int64_t kMaxHeadDim = 128;

// 依次搜索已加载符号、自定义算子包（ASCEND_CUSTOM_OPP_PATH）与标准 opapi 库，缓存库句柄
void *FindOpApiSymbol(const char *name)
{
    static std::mutex mutex;
    static std::vector<void *> handles;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (!initialized) {
        initialized = true;
        handles.push_back(RTLD_DEFAULT);
        const char *customPath = std::getenv("ASCEND_CUSTOM_OPP_PATH");
        while (customPath != nullptr && *customPath != '\0') {
            const char *sep = strchr(customPath, ':');
            size_t len = (sep != nullptr) ? static_cast<size_t>(sep - customPath) : strlen(customPath);
            handles.push_back(dlopen((std::string(customPath, len) + "/op_api/lib/libcust_opapi.so").c_str(),
                                     RTLD_LAZY | RTLD_GLOBAL));
            if (sep == nullptr) {
                break;
            }
            customPath = sep + 1;
        }
        for (const char *lib : {"libcust_opapi.so", "libopapi.so", "libopapi_transformer.so", "libopapi_math.so",
                                "libopapi_nn.so", "libascendcl.so"}) {
            handles.push_back(dlopen(lib, RTLD_LAZY | RTLD_GLOBAL));
        }
    }
    for (void *handle : handles) {
        if (handle == nullptr) {
            continue;
        }
        void *symbol = dlsym(handle, name);
        if (symbol != nullptr) {
            return symbol;
        }
    }
    return nullptr;
}

const char *GetAclErrorDetail()
{
    static aclGetRecentErrMsgFn getErrMsg =
        reinterpret_cast<aclGetRecentErrMsgFn>(FindOpApiSymbol("aclGetRecentErrMsg"));
    const char *detail = getErrMsg != nullptr ? getErrMsg() : nullptr;
    return detail != nullptr ? detail : "no acl error detail available";
}

aclTensor *ConvertType(const at::Tensor &tensor)
{
    if (!tensor.defined()) {
        return nullptr;
    }
    static aclCreateTensorFn createTensor =
        reinterpret_cast<aclCreateTensorFn>(FindOpApiSymbol("aclCreateTensor"));
    TORCH_CHECK(createTensor != nullptr, "aclCreateTensor not found in Ascend libraries.");
    aclDataType dataType = ACL_DT_UNDEFINED;
    switch (tensor.scalar_type()) {
        case at::ScalarType::Float:
            dataType = ACL_FLOAT;
            break;
        case at::ScalarType::Half:
            dataType = ACL_FLOAT16;
            break;
        case at::ScalarType::BFloat16:
            dataType = ACL_BF16;
            break;
        default:
            TORCH_CHECK(false, c10::toString(tensor.scalar_type()), " is not supported by SageAttention2.");
    }
    c10::SmallVector<int64_t, kMaxDimNum> storageDims;
    storageDims.push_back(tensor.storage().nbytes() / tensor.itemsize());
    aclFormat format = ACL_FORMAT_ND;
    switch (tensor.sizes().size()) {
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
    return createTensor(tensor.sizes().data(), tensor.sizes().size(), dataType, tensor.strides().data(),
                        tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(tensor.storage().data()));
}

const char *ConvertType(const std::string &value)
{
    return value.c_str();
}

template <typename T>
T ConvertType(T value)
{
    return value;
}

void ReleaseType(aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    static aclDestroyTensorFn destroyTensor =
        reinterpret_cast<aclDestroyTensorFn>(FindOpApiSymbol("aclDestroyTensor"));
    if (destroyTensor != nullptr) {
        destroyTensor(tensor);
    }
}

template <typename T>
void ReleaseType(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I>
void ReleaseConvertTypes(const Tuple &params, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(ReleaseType(std::get<I>(params)), 0)...};
}

template <typename Tuple, size_t... I>
int CallWorkspaceFunc(void *funcAddr, const Tuple &params, std::index_sequence<I...>)
{
    using FuncType = int (*)(typename std::tuple_element<I, Tuple>::type...);
    return reinterpret_cast<FuncType>(funcAddr)(std::get<I>(params)...);
}

template <typename Tuple>
struct ConvertedParamsGuard {
    Tuple &params;
    bool released = false;
    explicit ConvertedParamsGuard(Tuple &p) : params(p) {}
    ~ConvertedParamsGuard() {
        Release();
    }
    void Release() {
        if (!released) {
            ReleaseConvertTypes(params, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
            released = true;
        }
    }
};

// EXEC_NPU_CMD 系两段式下发：符号解析 -> GetWorkspaceSize -> workspace 分配 -> 第二段执行 -> 资源回收
// aclStream 为调用方（python 侧 torch.npu.current_stream().npu_stream）解析好的当前流句柄
//
// workspace 局部张量生命周期与异步 kernel 执行的衔接机制（无显式 aclrtSynchronize，亦无悬垂窗口）：
//   1. workspaceTensor 为宏块内的函数局部 Byte 张量，经 PrivateUse1 caching allocator 在 q 同设备
//      分配（OptionalDeviceGuard 已将当前设备切至 q.device）；
//   2. 第二段 aclnn 调用为异步语义：runFunc 仅在当前 NPU 流上入队内核后即返回；
//   3. 宏块结束时 workspaceTensor 析构、显存块归还 allocator 池；由于分配与内核消费同流，
//      allocator 的流序回收语义保证该块在同流新分配取走前，流上前序工作必已完成（跨流复用
//      由 allocator 插入事件同步），因此局部张量析构先于 kernel 实际执行并不构成悬垂；
//   4. aclTensor 描述符由 ConvertedParamsGuard 在第二段返回后释放——描述符仅宿主侧元数据，
//      发射时已被 executor 消费，不引用宿主栈内存。
#define EXEC_NPU_CMD(apiName, aclStream, ...)                                                              \
    do {                                                                                                   \
        void *wsFuncAddr = FindOpApiSymbol(#apiName "GetWorkspaceSize");                                   \
        void *runFuncAddr = FindOpApiSymbol(#apiName);                                                     \
        TORCH_CHECK(wsFuncAddr != nullptr && runFuncAddr != nullptr, #apiName,                             \
                    " or " #apiName "GetWorkspaceSize not found in Ascend op api libraries.");             \
        uint64_t workspaceSize = 0;                                                                        \
        aclOpExecutor *executor = nullptr;                                                                 \
        auto convertedParams = std::make_tuple(ConvertType(__VA_ARGS__)..., &workspaceSize, &executor);    \
        ConvertedParamsGuard<decltype(convertedParams)> guard(convertedParams);                            \
        auto wsStatus = CallWorkspaceFunc(                                                                 \
            wsFuncAddr, convertedParams, std::make_index_sequence<std::tuple_size<decltype(convertedParams)>::value>{}); \
        TORCH_CHECK(wsStatus == 0, "Call " #apiName "GetWorkspaceSize failed, detail: ", GetAclErrorDetail()); \
        void *workspaceAddr = nullptr;                                                                     \
        at::Tensor workspaceTensor;                                                                        \
        if (workspaceSize != 0) {                                                                          \
            workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)},                             \
                                        q.options().dtype(at::kByte));                                     \
            workspaceAddr = workspaceTensor.data_ptr();                                                    \
        }                                                                                                  \
        OpApiFunc runFunc = reinterpret_cast<OpApiFunc>(runFuncAddr);                                      \
        auto runStatus = runFunc(workspaceAddr, workspaceSize, executor, aclStream);                       \
        guard.Release();                                                                                   \
        TORCH_CHECK(runStatus == 0, "Call " #apiName " failed, detail: ", GetAclErrorDetail());            \
    } while (false)

// 与 python 层 _check_tensor_layout 对齐
void CheckLayout(const std::string &tensorLayout)
{
    TORCH_CHECK(tensorLayout == "HND" || tensorLayout == "NHD",
                "tensor_layout must be one of ['HND', 'NHD'], but got '", tensorLayout, "'.");
}

// 与 python 层 _check_dtype 对齐
void CheckDtype(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    for (const at::Tensor *tensor : {&q, &k, &v}) {
        TORCH_CHECK(tensor->scalar_type() == at::ScalarType::Half ||
                        tensor->scalar_type() == at::ScalarType::BFloat16,
                    "q/k/v must be torch.float16 or torch.bfloat16, but got ", c10::toString(tensor->scalar_type()),
                    ".");
    }
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
                "q/k/v must share the same dtype.");
}

// 与 python 层 _check_shape 对齐
void CheckShape(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const std::string &tensorLayout,
                bool isCausal)
{
    TORCH_CHECK(q.dim() == kDimNum4 && k.dim() == kDimNum4 && v.dim() == kDimNum4,
                "q/k/v must be 4D tensors, but got dims q=", q.dim(), ", k=", k.dim(), ", v=", v.dim(), ".");
    const bool isNhd = tensorLayout == "NHD";
    const int64_t headAxis = isNhd ? 2 : 1;
    const int64_t seqAxis = isNhd ? 1 : 2;
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q/k/v must share the same batch size, but got q=", q.size(0), ", k=", k.size(0), ", v=", v.size(0),
                ".");
    TORCH_CHECK(q.size(3) == k.size(3) && k.size(3) == v.size(3),
                "q/k/v must share the same head_dim, but got q=", q.size(3), ", k=", k.size(3), ", v=", v.size(3),
                ".");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must share the same shape, but got k=", k.sizes(),
                " and v=", v.sizes(), ".");
    TORCH_CHECK(q.size(headAxis) % k.size(headAxis) == 0,
                "num_qo_heads must be divisible by num_kv_heads, but got Hq=", q.size(headAxis),
                " and Hk=", k.size(headAxis), ".");
    TORCH_CHECK(!isCausal || q.size(seqAxis) == k.size(seqAxis),
                "is_causal is valid only when qo_len equals kv_len, but got Sq=", q.size(seqAxis),
                " and Sk=", k.size(seqAxis), ".");
}

// 与 python 层 _check_stride 对齐
void CheckStride(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    for (const at::Tensor *tensor : {&q, &k, &v}) {
        TORCH_CHECK(tensor->stride(-1) == 1, "q/k/v must have stride(-1) == 1 (contiguous last dim), but got stride ",
                    tensor->strides(), ".");
    }
}

// 与 python 层 _check_params 对齐
void CheckParams(const std::string &qkQuantGran, const std::string &pvAccumDtype, double smScale, int64_t headDim)
{
    TORCH_CHECK(qkQuantGran == "per_warp" || qkQuantGran == "per_thread",
                "qk_quant_gran must be one of ['per_warp', 'per_thread'], but got '", qkQuantGran, "'.");
    TORCH_CHECK(pvAccumDtype == "fp32" || pvAccumDtype == "fp32+fp32" || pvAccumDtype == "fp32+fp16",
                "pv_accum_dtype must be one of ['fp32', 'fp32+fp32', 'fp32+fp16'], but got '", pvAccumDtype, "'.");
    TORCH_CHECK(std::isfinite(smScale) && std::fabs(smScale) <= static_cast<double>(std::numeric_limits<float>::max()),
                "sm_scale must be finite and within the fp32 representable range, but got ", smScale, ".");
    TORCH_CHECK(headDim <= kMaxHeadDim, "head_dim must not be greater than ", kMaxHeadDim, ", but got ", headDim, ".");
}

} // namespace

// NPU 前向：q/k/v 已由 dispatch 层完成 head_dim pad；返回 (out, lse)，return_lse 为 false 时 lse 为空张量
std::tuple<at::Tensor, at::Tensor> Sa2Attention(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const std::string &tensorLayout, bool isCausal,
    const std::string &qkQuantGran, double smScale, const std::string &pvAccumDtype, bool smoothK, bool smoothV,
    bool returnLse, int64_t streamHandle)
{
    CheckLayout(tensorLayout);
    CheckDtype(q, k, v);
    CheckShape(q, k, v, tensorLayout, isCausal);
    CheckStride(q, k, v);
    CheckParams(qkQuantGran, pvAccumDtype, smScale, q.size(3));
    TORCH_CHECK(q.is_privateuseone() && k.is_privateuseone() && v.is_privateuseone(),
                "SageAttention2 forward expects NPU tensors.");
    TORCH_CHECK(q.get_device() == k.get_device() && k.get_device() == v.get_device(),
                "q, k, and v must reside on the same NPU device.");

    const bool isNhd = tensorLayout == "NHD";
    const int64_t headAxis = isNhd ? 2 : 1;
    const int64_t seqAxis = isNhd ? 1 : 2;
    at::Tensor out = at::empty(q.sizes(), q.options());
    at::Tensor lse = returnLse
                         ? at::empty({q.size(0), q.size(headAxis), q.size(seqAxis)}, q.options().dtype(at::kFloat))
                         : at::Tensor();

    c10::OptionalDeviceGuard deviceGuard(at::Device(at::kPrivateUse1, q.get_device()));
    aclrtStream aclStream = reinterpret_cast<aclrtStream>(streamHandle);

    // EXEC_NPU_CMD 实参顺序 = aclnnSa2Attention2GetWorkspaceSize 形参顺序（输入 -> 属性 -> 输出）
    EXEC_NPU_CMD(aclnnSa2Attention2, aclStream, q, k, v, tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype,
                 smoothK, smoothV, returnLse, out, lse);

    return std::make_tuple(out, lse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sa2_attention", &Sa2Attention,
          "SageAttention2 INT8 QK^T + FP8 PV forward on NPU", "q"_a, "k"_a, "v"_a, "tensor_layout"_a, "is_causal"_a,
          "qk_quant_gran"_a, "sm_scale"_a, "pv_accum_dtype"_a, "smooth_k"_a, "smooth_v"_a, "return_lse"_a,
          "stream_handle"_a);
}
