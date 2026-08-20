// arch35 FFN PyTorch binding: torch.ops.npu.ffn_linear -> aclnnFFNV2。
// 直接接收 PyTorch Linear 布局权重 [N,K]（out,in），绕过 torch_npu.npu_ffn 的
// canonical [K,N] 布局校验；kernel 内部通过 transB（ColumnMajor）处理。
#include <cctype>
#include <torch/library.h>
#include "../op_host/ffn_layout.h"
#include "torch_aclnn_helper.h"

using namespace at;

namespace ascend_kernel {

// 与 aclnn GetActiveType 同语义：大小写不敏感、长度精确匹配（不做前缀匹配）
bool is_swiglu(const std::string &act)
{
    static constexpr char kSwiglu[] = "swiglu";
    if (act.size() != sizeof(kSwiglu) - 1) {
        return false;
    }
    for (size_t i = 0; i < act.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(act[i])) != kSwiglu[i]) {
            return false;
        }
    }
    return true;
}

// 布局识别走公共规则（../op_host/ffn_layout.h）：全方阵/均不匹配默认 canonical。
std::vector<int64_t> ffn_linear_npu_output_size(const at::Tensor &x, const at::Tensor &weight1,
                                                const at::Tensor &weight2, const std::string &activation)
{
    auto xSizes = x.sizes().vec();
    const int64_t xK = xSizes.back();
    const int64_t w1d0 = weight1.size(-2);
    const int64_t w1d1 = weight1.size(-1);
    const int64_t w2d0 = weight2.size(-2);
    const int64_t w2d1 = weight2.size(-1);
    const bool swiglu = is_swiglu(activation);
    const ffnlayout::FfnLayout layout = ffnlayout::FfnDetectLayout(w1d0, w1d1, w2d0, w2d1, xK, swiglu);
    TORCH_CHECK(layout != ffnlayout::FfnLayout::INVALID, "weight1 shape [", w1d0, ", ", w1d1,
                "] does not match x K=", xK, " (expect [K,N] canonical or [N,K] linear)");
    const bool isLinear = (layout == ffnlayout::FfnLayout::LINEAR);
    if (swiglu && !isLinear) {
        TORCH_CHECK(false, "swiglu only supports linear layout weight1 [2H,K]");
    }
    xSizes[xSizes.size() - 1] = isLinear ? w2d0 : w2d1;
    return xSizes;
}

at::Tensor ffn_linear(const at::Tensor &x,
                      const at::Tensor &weight1,
                      const at::Tensor &weight2,
                      const c10::optional<at::Tensor> &bias1_opt,
                      const c10::optional<at::Tensor> &bias2_opt,
                      c10::string_view activation,
                      int64_t inner_precise)
{
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
    TORCH_CHECK(weight1.dim() == 2 && weight2.dim() == 2, "weight1/weight2 must be 2D");

    at::Tensor x_contiguous = x.contiguous();
    at::Tensor w1_contiguous = weight1.contiguous();
    at::Tensor w2_contiguous = weight2.contiguous();

    c10::optional<at::Tensor> bias1 = (bias1_opt.has_value() && bias1_opt.value().defined())
        ? c10::optional<at::Tensor>(bias1_opt.value().contiguous()) : c10::nullopt;
    c10::optional<at::Tensor> bias2 = (bias2_opt.has_value() && bias2_opt.value().defined())
        ? c10::optional<at::Tensor>(bias2_opt.value().contiguous()) : c10::nullopt;

    c10::optional<at::IntArrayRef> expertTokens = c10::nullopt;
    c10::optional<at::Tensor> scale = c10::nullopt;
    c10::optional<at::Tensor> offset = c10::nullopt;
    c10::optional<at::Tensor> deqScale1 = c10::nullopt;
    c10::optional<at::Tensor> deqScale2 = c10::nullopt;
    c10::optional<at::Tensor> antiquantScale1 = c10::nullopt;
    c10::optional<at::Tensor> antiquantScale2 = c10::nullopt;
    c10::optional<at::Tensor> antiquantOffset1 = c10::nullopt;
    c10::optional<at::Tensor> antiquantOffset2 = c10::nullopt;

    std::string actStr(activation.data(), activation.size());
    auto output_size = ffn_linear_npu_output_size(x_contiguous, w1_contiguous, w2_contiguous, actStr);
    at::Tensor y = at_npu::native::empty_with_format(
        output_size, x_contiguous.options(),
        at_npu::native::get_npu_format(x_contiguous));

    const char *activationCStr = actStr.c_str();
    bool tokensIndexFlag = false;

    EXEC_NPU_CMD(aclnnFFNV2,
        x_contiguous, w1_contiguous, w2_contiguous,
        expertTokens,
        bias1, bias2,
        scale, offset,
        deqScale1, deqScale2,
        antiquantScale1, antiquantScale2,
        antiquantOffset1, antiquantOffset2,
        activationCStr, inner_precise, tokensIndexFlag,
        y);

    return y;
}

}  // namespace ascend_kernel

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("ffn_linear(Tensor x, Tensor weight1, Tensor weight2, Tensor? bias1=None, Tensor? bias2=None, str activation='gelu', int inner_precise=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("ffn_linear", TORCH_FN(ascend_kernel::ffn_linear));
}
