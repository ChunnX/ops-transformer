#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <torch/extension.h>

#include "../../../aclnn/sage_attn_proto.h"

#if defined(HAS_TORCH_NPU) || (defined(__has_include) && __has_include("torch_npu/csrc/aten/CustomFunctions.h"))
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/utils/op_api_common_base.h"
#define SAGE_NPU_AVAILABLE 1
#endif

namespace op_api {
namespace {
constexpr int64_t RANK = 4;
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr int64_t ND_FORMAT = 2;
constexpr int64_t NCHW_FORMAT = 0;
constexpr int64_t MAX_DIM = (int64_t{1} << 32) - 64;

// 接口编码与 "字符串 -> 数值编码" 映射的唯一定义位于 aclnn/sage_attn_proto.h
// （TensorLayoutCodeFromString / QuantGranCodeFromString / PvAccumCodeFromString），
// 本文件经 include 复用，禁止再引入第二套字符串协议。

#if SAGE_NPU_AVAILABLE
void check_storage_format(const at::Tensor &tensor)
{
    auto format = at_npu::native::custom_ops::get_npu_format(tensor);
    TORCH_CHECK(format == ND_FORMAT || format == NCHW_FORMAT,
                "SageAttention2 requires base ND/NCHW storage on Ascend 950; private formats are unsupported");
}
#endif
} // namespace

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
    TORCH_CHECK(tensor_layout == "HND" || tensor_layout == "NHD", "tensor_layout must be HND or NHD");
    TORCH_CHECK(qk_quant_gran == "per_thread" || qk_quant_gran == "per_warp",
                "qk_quant_gran must be per_thread or per_warp");
    TORCH_CHECK(pv_accum_dtype == "fp32" || pv_accum_dtype == "fp32+fp32" || pv_accum_dtype == "fp32+fp16",
                "pv_accum_dtype must be fp32, fp32+fp32, or fp32+fp16");

    for (const auto &tensor : {q, k, v}) {
        TORCH_CHECK(tensor.dim() == RANK, "q/k/v must have rank 4");
#if SAGE_NPU_AVAILABLE
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device() == q.device(),
                    "q/k/v must share an NPU device");
#endif
        TORCH_CHECK(tensor.scalar_type() == q.scalar_type() &&
                    (tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16),
                    "q/k/v must share FP16 or BF16 dtype");
        TORCH_CHECK(tensor.stride(-1) == 1, "last dimension must be contiguous");
        TORCH_CHECK(!tensor.requires_grad(), "forward-only operator does not support autograd");
        for (auto size : tensor.sizes()) {
            TORCH_CHECK(size > 0 && size < MAX_DIM, "invalid tensor dimension: ", size);
        }
    }

    const int64_t head_axis = (tensor_layout == "HND") ? 1 : 2;
    const int64_t seq_axis = (tensor_layout == "HND") ? 2 : 1;

    TORCH_CHECK(q.size(-1) <= MAX_HEAD_DIM, "head_dim must be in [1,128]");
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == q.size(0) && k.size(-1) == q.size(-1),
                "incompatible q/k/v shapes");
    TORCH_CHECK(q.size(head_axis) % k.size(head_axis) == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(q.size(0) * k.size(head_axis) < (int64_t{1} << 32), "batch times KV heads exceeds uint32 range");
    TORCH_CHECK(!is_causal || q.size(seq_axis) == k.size(seq_axis), "causal requires Sq == Sk");

    // sm_scale 冻结语义：None 才默认 1/sqrt(D)；显式 0/负值/非有限值一律报错
    double scale = sm_scale.value_or(1.0 / std::sqrt(static_cast<double>(q.size(-1))));
    TORCH_CHECK(scale > 0.0 && std::isfinite(scale) && scale <= static_cast<double>(std::numeric_limits<float>::max()),
                "invalid sm_scale: explicit values must be finite and > 0 (omit for default 1/sqrt(head_dim))");

    auto out = at::empty(q.sizes(), q.options());
    c10::optional<at::Tensor> lse;
    if (return_lse) {
        lse = at::empty({q.size(0), q.size(head_axis), q.size(seq_axis)}, q.options().dtype(at::kFloat));
    }

    // C++ 侧统一完成 字符串 -> proto.h 数值编码 的转换，
    // 与 aclnn 两段式 C 接口（aclnnSageAttention2GetWorkspaceSize）签名逐位对齐：
    // (q, k, v, key_mean, lse_correction, sm_scale, gran, accum, is_causal, layout, smooth_k, smooth_v, out, lse)
    // c10::optional<at::Tensor> 由 torch_npu ConvertType 转换，空 optional 即 nullptr aclTensor
    const int64_t layout_code = sage_attn::TensorLayoutCodeFromString(tensor_layout.c_str());
    const int64_t gran_code = sage_attn::QuantGranCodeFromString(qk_quant_gran.c_str());
    const int64_t accum_code = sage_attn::PvAccumCodeFromString(pv_accum_dtype.c_str());
    // smooth_k 均值与 lse_correction 修正随算子内融合，独立张量入口暂不启用
    c10::optional<at::Tensor> none_tensor;

#if SAGE_NPU_AVAILABLE
    const c10::OptionalDeviceGuard device_guard(q.device());
    check_storage_format(q);
    check_storage_format(k);
    check_storage_format(v);

    EXEC_NPU_CMD_EXT(aclnnSageAttention2,
                     q, k, v,
                     none_tensor, none_tensor,
                     scale, gran_code, accum_code,
                     is_causal, layout_code,
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
