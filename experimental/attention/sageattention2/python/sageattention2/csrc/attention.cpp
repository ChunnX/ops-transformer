#include <cmath>
#include <dlfcn.h>
#include <string>
#include <tuple>
#include <torch/extension.h>

// 交付态 aclnn 两段式唯一 ABI 面（字符串属性签名，与 def.cpp 注册的 7 项属性
// 一致；移植态数值形参面已删除收敛）。经相对路径引入，编译不依赖 OPP 安装产物。
#include "../../../op_api/aclnn_sage_attention2.h"

#if defined(HAS_TORCH_NPU) || (defined(__has_include) && __has_include("torch_npu/csrc/aten/CustomFunctions.h"))
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/utils/op_api_common_base.h"
#define SAGE_NPU_AVAILABLE 1
#endif

#if SAGE_NPU_AVAILABLE
namespace sage_direct {

// 直连符号两段式执行（与 benchmarks/benchmark_aclnn.cpp 的 dlsym 直调同口径）：
// 本扩展以 -lcust_opapi 链接 op_api 库，两段式入口即链接期符号，无需经
// torch_npu GetApiFunc 字符串解析器。该解析路径要求 vendor config.ini
// load_priority 注册，实测会令内置算子（FIA 基线）的内核解析同样落到本算子
// vendor 而设备态崩溃——2026-10-04 真机定位，plog 指纹与本算子 trap 完全一致。
// 同时不走 hit_cache_ext 执行器缓存：每次调用重走 GetWorkspaceSize → TilingImpl，
// Host Tiling 输入（含调优会话的 SAGEATTN_FORCE_* 覆盖）逐调用生效。
// 余下步骤（上下文/流/确定性配置/参数转换/workspace/发射/释放）与
// EXEC_NPU_CMD_V1_EXT 逐一相同，仅符号来源与缓存行为不同。

inline void EnsureCannOpapiLoaded()
{
    static bool done = []() {
        // libopapi.so 提供 l0op::Contiguous 等 op_api 运行时符号；显式
        // RTLD_GLOBAL 预载，规避懒绑定在 libopapi 未入全局域时的
        // symbol lookup error（2026-10-04 真机实测）。
        if (dlopen("libopapi.so", RTLD_NOW | RTLD_GLOBAL) == nullptr) {
            TORCH_WARN("SageAttention2: preload libopapi.so failed: ", dlerror());
        }
        return true;
    }();
    (void)done;
}

} // namespace sage_direct

// ws_fn / run_fn 为链接期可见的两段式入口符号名（非字符串解析）
#define SAGE_EXEC_NPU_DIRECT(ws_fn, run_fn, ...)                                                           \
    do {                                                                                                   \
        sage_direct::EnsureCannOpapiLoaded();                                                              \
        InitExecCommonCtx();                                                                               \
        auto acl_stream = GetAclStream();                                                                  \
        SetExecConfig();                                                                                   \
        uint64_t workspace_size = 0;                                                                       \
        uint64_t *workspace_size_addr = &workspace_size;                                                   \
        aclOpExecutor *executor = nullptr;                                                                 \
        aclOpExecutor **executor_addr = &executor;                                                         \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);             \
        auto get_workspace_func = ConvertToOpApiFunc(converted_params, (void *)&(ws_fn));                  \
        auto workspace_status = call(get_workspace_func, converted_params);                                \
        TORCH_CHECK(workspace_status == 0, "SageAttention2 GetWorkspaceSize failed: ", workspace_status);  \
        void *workspace_addr = GetWorkSpaceAddr(workspace_size);                                           \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor, run_fn]() -> int { \
            InitExecSubTheadCtx(acl_stream);                                                               \
            auto api_ret = ExecuteApiFunc((void *)(run_fn), acl_stream, workspace_addr, workspace_size, executor); \
            ReleaseConvertTypes(converted_params);                                                         \
            ReleaseExecCommonCtx();                                                                        \
            return api_ret;                                                                                \
        };                                                                                                 \
        int api_ret = acl_call();                                                                          \
        TORCH_CHECK(api_ret == 0, "SageAttention2 aclnn launch failed: ", api_ret);                        \
        UnInitExecCommonCtx();                                                                             \
    } while (false)
#endif


namespace op_api {
namespace {
#if SAGE_NPU_AVAILABLE
constexpr int64_t ND_FORMAT = 2;
constexpr int64_t NCHW_FORMAT = 0;

void check_storage_format(const at::Tensor &tensor)
{
    auto format = at_npu::native::custom_ops::get_npu_format(tensor);
    TORCH_CHECK(format == ND_FORMAT || format == NCHW_FORMAT,
                "SageAttention2 requires base ND/NCHW storage on Ascend 950; private formats are unsupported");
}
#endif
} // namespace

// 桥接层职责收敛（P1-4）：只保留 ABI 调用必需的最小断言（NPU 设备一致性、
// forward-only 契约）；形状/dtype/布局/粒度/累加档/causal/sm_scale 的完整校验
// 唯一属于 Python 边界（error_check）与 aclnn C 入口（op_api 白名单 +
// ComputeHostTiling），此处不再复制第二份。
std::tuple<at::Tensor, c10::optional<at::Tensor>> sageattention2(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    std::string tensor_layout,
    bool is_causal,
    std::string qk_quant_gran,
    c10::optional<double> sm_scale,
    std::string pv_accum_dtype,
    bool smooth_k,
    bool smooth_v,
    bool return_lse)
{
    // ABI 必需最小断言：NPU 设备一致性与 forward-only 契约
    for (const auto &tensor : {q, k, v}) {
#if SAGE_NPU_AVAILABLE
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == q.device(),
                    "q/k/v must share an NPU device");
#endif
        TORCH_CHECK(!tensor.requires_grad(), "forward-only operator does not support autograd");
    }

    auto out = at::empty(q.sizes(), q.options());
    // 开源契约（git/SageAttention/sageattention/core.py:583-597）：km 与 lse_correction
    // 均在上层算好传入——
    // - km = k.mean(seq_dim)（输入 dtype 舍回；上游 pad_head_dim 已把 pad 通道补零），
    //   此处以 fp32 张量承载（keyMean GM 入参为 fp32 视图，kernel 按输入位型舍回消费）；
    // - lse_correction = q·km_broadcast^T（fp32 matmul，仅 return_lse 时），[B, Hq, Sq]，
    //   与布局无关（lse 输出同为该形）。设备核不再核内计算 q·km，Finalize 直读本张量。
    const int64_t head_axis = (tensor_layout == "HND") ? 1 : 2;
    const int64_t seq_axis = (tensor_layout == "HND") ? 2 : 1;
    c10::optional<at::Tensor> lse;
    if (return_lse) {
        lse = at::empty({q.size(0), q.size(head_axis), q.size(seq_axis)}, q.options().dtype(at::kFloat));
    }
    c10::optional<at::Tensor> km_tensor;
    c10::optional<at::Tensor> lse_corr_tensor;
    if (smooth_k) {
        km_tensor = k.mean(seq_axis, /*keepdim=*/false).to(at::kFloat).contiguous();
        if (return_lse) {
            const int64_t gqa = q.size(head_axis) / k.size(head_axis);
            auto q_hnd = (head_axis == 1) ? q : q.transpose(1, 2);  // [B, Hq, Sq, D] 视图
            auto km_b = km_tensor->repeat_interleave(gqa, /*dim=*/1)
                            .unsqueeze(2);  // [B, Hq, 1, D] GQA 广播
            lse_corr_tensor = at::matmul(q_hnd.to(at::kFloat), km_b.to(at::kFloat).transpose(2, 3))
                                  .squeeze(-1);
        }
    }

#if SAGE_NPU_AVAILABLE
    // sm_scale 冻结语义：None 才默认 1/sqrt(D)（显式非法值由 Python 边界与 C 入口拒绝）
    double scale = sm_scale.value_or(1.0 / std::sqrt(static_cast<double>(q.size(-1))));

    const c10::OptionalDeviceGuard device_guard(q.device());
    check_storage_format(q);
    check_storage_format(k);
    check_storage_format(v);

    // 交付态字符串属性 ABI：layout/gran/accum 直传字符串（op_api 白名单校验），
    // 禁止回退数值编码形态（与 benchmarks/benchmark_aclnn.cpp 同一口径）
    const char *layout_cstr = tensor_layout.c_str();
    const char *gran_cstr = qk_quant_gran.c_str();
    const char *accum_cstr = pv_accum_dtype.c_str();
    SAGE_EXEC_NPU_DIRECT(aclnnSageAttention2GetWorkspaceSize, aclnnSageAttention2,
                     q, k, v,
                     km_tensor, lse_corr_tensor,
                     layout_cstr, is_causal,
                     gran_cstr, scale,
                     accum_cstr,
                     smooth_k, smooth_v,
                     out, lse);
#else
    TORCH_CHECK(false, "SageAttention2 C++ extension requires Ascend NPU hardware environment.");
#endif

    return {out, lse};
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &op_api::sageattention2,
          "SageAttention2 INT8 QK / FP8 PV forward with full parameter passthrough",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("tensor_layout") = "HND",
          py::arg("is_causal") = false,
          py::arg("qk_quant_gran") = "per_thread",
          py::arg("sm_scale") = c10::nullopt,
          py::arg("pv_accum_dtype") = "fp32+fp16",
          py::arg("smooth_k") = true,
          py::arg("smooth_v") = false,
          py::arg("return_lse") = false);
}
