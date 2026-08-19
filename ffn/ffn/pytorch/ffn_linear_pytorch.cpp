// arch35 FFN PyTorch binding: torch.ops.npu.ffn_linear -> aclnnFFNV2。
// 直接接收 PyTorch Linear 布局权重 [N,K]（out,in），绕过 torch_npu.npu_ffn 的
// canonical [K,N] 布局校验；kernel 内部通过 transB（ColumnMajor）处理。
#include <torch/library.h>
#include "torch_aclnn_helper.h"

using namespace at;

namespace ascend_kernel {

bool is_swiglu(const std::string &act)
{
    return act.rfind("swiglu", 0) == 0 || act.rfind("Swiglu", 0) == 0 || act.rfind("SWIGLU", 0) == 0;
}

// 与 aclnn/infershape/tiling 同一套布局识别：
//  1) w1 非方阵：w1d1==xK → linear；w1d0==xK → canonical；
//  2) w1 方阵：用 w2 消歧（swiglu 隐藏宽 = w1 边长/2）；
//  3) 全方阵/均不匹配：BF16→linear，其余→canonical（写死约定）。
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
    const int64_t hiddenW = swiglu ? w1d0 / 2 : w1d0;
    const int64_t hiddenW2 = swiglu ? w1d1 / 2 : w1d1;
    bool isLinear = false;
    if (w1d1 == xK && w1d0 != xK) {
        isLinear = true;
    } else if (w1d0 == xK && w1d1 != xK) {
        isLinear = false;
    } else if (w1d0 == xK && w1d1 == xK) {
        if (w2d1 == hiddenW && w2d0 != hiddenW2) {
            isLinear = true;
        } else if (w2d0 == hiddenW2 && w2d1 != hiddenW) {
            isLinear = false;
        } else {
            isLinear = (x.scalar_type() == at::kBFloat16);
        }
    } else {
        TORCH_CHECK(false, "weight1 shape [", w1d0, ", ", w1d1,
                    "] does not match x K=", xK, " (expect [K,N] canonical or [N,K] linear)");
    }
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
